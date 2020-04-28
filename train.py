"""
Script to train model.
"""
import io
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from bedrock_client.bedrock.api import BedrockApi
from google.cloud import storage
from sklearn import metrics

from senet import se_resnext50_32x4d
from utils import ImageDataset, ToTensor, seed_torch

PROJECT = "span-production"
BUCKET = "bedrock-sample"
BASE_DIR = os.getenv("BASE_DIR")


class CustomSEResNeXt(nn.Module):

    def __init__(self, buffer, device):
        super().__init__()

        self.model = se_resnext50_32x4d(pretrained=None)
        self.model.load_state_dict(torch.load(buffer, map_location=device))
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, CFG.n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


def train_fn(model, train_loader, valid_loader, device):
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2, verbose=True, eps=1e-6)

    criterion = nn.CrossEntropyLoss()
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        model.train()
        avg_loss = 0.
        y_train = []
        y_preds = []

        optimizer.zero_grad()

        for i, data in enumerate(train_loader):
            images = data["image"].to(device)
            labels = data["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss += loss.item() / len(train_loader)

            y_train.append(labels.cpu().numpy())
            y_preds.append(logits.detach().cpu().numpy().argmax(axis=1))

        y_train = np.concatenate(y_train)
        y_preds = np.concatenate(y_preds)
        train_acc = metrics.accuracy_score(y_train, y_preds)

        model.eval()
        avg_val_loss = 0.
        y_valid = []
        y_preds = []

        for i, data in enumerate(valid_loader):
            images = data["image"].to(device)
            labels = data["label"].to(device)

            with torch.no_grad():
                logits = model(images)
            loss = criterion(logits, labels)

            avg_val_loss += loss.item() / len(valid_loader)

            y_valid.append(labels.cpu().numpy())
            y_preds.append(logits.cpu().numpy().argmax(axis=1))

        scheduler.step(avg_val_loss)

        y_valid = np.concatenate(y_valid)
        y_preds = np.concatenate(y_preds)
        valid_acc = metrics.accuracy_score(y_valid, y_preds)

        print(f"Epoch {epoch + 1}/{CFG.epochs}: elapsed time: {time.time() - start_time:.0f}s\n"
              f"  loss: {avg_loss:.4f}  train_acc: {train_acc:.4f}"
              f" - val_loss: {avg_val_loss:.4f}  val_acc: {valid_acc:.4f}")

        if avg_val_loss < best_loss:
            print(f"Epoch {epoch + 1}: val_loss improved from {best_loss:.5f} to {avg_val_loss:.5f}, "
                  f"saving model to {CFG.model_path}")
            best_loss = avg_val_loss
            torch.save(model.state_dict(), CFG.model_path)
        else:
            print(f"Epoch {epoch + 1}: val_loss did not improve from {avg_val_loss:.5f}")


def predict(model, data_loader, device):
    model.to(device)
    model.eval()

    y_probs = []
    y_actual = []
    for i, data in enumerate(data_loader):
        images = data["image"].to(device)
        labels = data["label"].to(device)

        with torch.no_grad():
            logits = model(images)

        y_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        y_actual.append(labels.cpu().numpy())

    y_probs = np.concatenate(y_probs)
    y_actual = np.concatenate(y_actual)
    return y_probs, y_actual


def compute_log_metrics(y_val, y_prob, y_pred):
    """Compute and log metrics."""
    acc = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    roc_auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)
    
    print(f"  Accuracy          = {acc:.6f}")
    print(f"  Precision         = {precision:.6f}")
    print(f"  Recall            = {recall:.6f}")
    print(f"  F1 score          = {f1_score:.6f}")
    print(f"  ROC AUC           = {roc_auc:.6f}")
    print(f"  Average precision = {avg_prc:.6f}")

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 score", f1_score)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(),
                           y_prob.flatten().tolist())

    
class CFG:
    lr = 1e-5
    batch_size = 8
    epochs = 20
    n_classes = 2
    model_path = "/artefact/train/trained_model.pth"
    

def train():
    """Train"""
    client = storage.Client(PROJECT)
    bucket = client.get_bucket(BUCKET)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print("Split train and validation data")
    proc_data = ImageDataset(
        root_dir=BUCKET,
        image_dir="proc_images",
        transform=ToTensor(),
    )

    seed_torch(seed=42)
    valid_size = int(len(proc_data) * 0.2)
    train_data, valid_data = torch.utils.data.random_split(
        proc_data, [len(proc_data) - valid_size, valid_size])

    train_loader = DataLoader(train_data, batch_size=CFG.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=CFG.batch_size, shuffle=False)

    print("Train model")
    se_model_blob = bucket.blob(os.path.join(
        BASE_DIR, "pytorch-se-resnext/se_resnext50_32x4d-a260b3a4.pth"))
    model = CustomSEResNeXt(io.BytesIO(se_model_blob.download_as_string()), device)
    train_fn(model, train_loader, valid_loader, device)

    print("Evaluate")
    y_probs, y_val = predict(model, valid_loader, device)
    y_preds = y_probs.argmax(axis=1)

    compute_log_metrics(y_val, y_probs[:, 1], y_preds)


if __name__ == "__main__":
    train()

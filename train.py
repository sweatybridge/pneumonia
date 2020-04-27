"""
Script to train model.
"""
import logging
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bedrock_client.bedrock.api import BedrockApi
from sklearn import metrics

from senet import se_resnext50_32x4d

BUCKET = os.getenv("BUCKET")


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, finding = sample['image'], sample['finding']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'finding': finding}


class ImageDataset(Dataset):
    """Class creator for the x-ray dataset."""

    def __init__(self, root_dir, csv_path="metadata.csv", image_path="proc_images", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.image_path = image_path
        self.transform = transform
        self.df = pd.read_csv(os.path.join(root_dir, csv_path))
        # If not a PA view, drop the line
        self.df.drop(self.df[self.df.view != 'PA'].index, inplace=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.df['finding'].iloc[idx] != 'COVID-19':
            finding = 0
            img_path = os.path.join(self.root_dir, self.image_path, self.df['filename'].iloc[idx])
            image = cv2.imread(img_path)
            sample = {'image': image, 'finding': finding}

            if self.transform:
                sample = self.transform(sample)

        else:
            finding = 1
            img_path = os.path.join(self.root_dir, self.image_path, self.df['filename'].iloc[idx])
            image = cv2.imread(img_path)
            sample = {'image': image, 'finding': finding}

            if self.transform:
                sample = self.transform(sample)

        return sample


class CFG:
    lr = 1e-5
    batch_size = 8
    epochs = 5
    n_classes = 2
    model_path = "models/trained_se_resnext50.pth"


class CustomSEResNeXt(nn.Module):

    def __init__(self, weights_path="pytorch-se-resnext/se_resnext50_32x4d-a260b3a4.pth"):
        super().__init__()

        self.model = se_resnext50_32x4d(pretrained=None)
        self.model.load_state_dict(torch.load(os.path.join(BUCKET, weights_path)))
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, CFG.n_classes)

    def forward(self, x):
        x = self.model(x)
        return x


def train_fn(model, train_loader, valid_loader, device):
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, eps=1e-6)

    criterion = nn.CrossEntropyLoss()
    best_score = -100
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        model.train()
        avg_loss = 0.
        y_train = []
        y_preds = []

        optimizer.zero_grad()

        for i, data in enumerate(train_loader):
            images = data['image'].to(device)
            labels = data['finding'].to(device)

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
            images = data['image'].to(device)
            labels = data['finding'].to(device)

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
              f" - val_loss: {avg_val_loss:.6f}  val_acc: {valid_acc:.4f}")

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
        images = data['image'].to(device)
        labels = data['finding'].to(device)

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
    auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)
    print("Accuracy = {:.6f}".format(acc))
    print("Precision = {:.6f}".format(precision))
    print("Recall = {:.6f}".format(recall))
    print("F1 score = {:.6f}".format(f1_score))
    print("AUC = {:.6f}".format(auc))
    print("Average precision = {:.6f}".format(avg_prc))

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 score", f1_score)
    bedrock.log_metric("AUC", auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(),
                           y_prob.flatten().tolist())


def main():
    """Train"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    seed_torch(seed=42)

    print("\tSplitting train and validation data")
    proc_data = ImageDataset(
        root_dir=BUCKET,
        transform=ToTensor(),
    )

    valid_size = int(len(proc_data) * 0.2 / CFG.batch_size)
    train_data, valid_data = torch.utils.data.random_split(
        proc_data, [len(proc_data) - valid_size, valid_size])

    train_loader = DataLoader(train_data, batch_size=CFG.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=CFG.batch_size, shuffle=False)

    print("\tTrain model")
    model = CustomSEResNeXt()
    train_fn(model, train_loader, valid_loader, device)

    print("\tEvaluating")
    y_probs, y_val = predict(model, valid_loader, device)
    y_preds = y_probs.argmax(axis=1)

    compute_log_metrics(y_val, y_probs, y_preds)


if __name__ == "__main__":
    main()

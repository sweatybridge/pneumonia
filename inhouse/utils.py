"""
Script containing commonly used functions.
"""
import os
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from skimage import transform

from pretrainedmodels import se_resnext50_32x4d


class ImageDataset(Dataset):
    """Class for X-ray dataset."""

    # pylint: disable=too-many-arguments
    def __init__(self, root_dir, image_dir, df, transform=None, bucket=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.transform = transform
        self.bucket = bucket
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = os.path.join(
            self.root_dir, self.image_dir, self.df["filename"].iloc[idx]
        )
        if self.bucket is None:
            image = cv2.imread(file_path)
        else:
            image = gcs_imread(self.bucket, file_path)

        if self.transform:
            image = self.transform(image)

        if self.df["finding"].iloc[idx] != "COVID-19":
            label = 0
        else:
            label = 1

        return {"image": image, "label": label}


def seed_torch(seed=42):
    """Set random seed."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def gcs_imread(bucket, blob_path):
    """Load image from GCS bucket."""
    blob = bucket.blob(blob_path)
    image = cv2.imdecode(
        np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), 1
    )
    return image


def gcs_imwrite(bucket, blob_path, filename, image):
    """Save image to GCS bucket."""
    cv2.imwrite(filename, image)
    bucket.blob(blob_path).upload_from_filename(filename)


class CustomSEResNeXt(nn.Module):
    """CustomSEResNeXt"""

    def __init__(self, weights_path, device, n_classes=2, save=None):
        super().__init__()

        self.model = se_resnext50_32x4d(pretrained=None)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        if save is not None:
            torch.save(self.model.state_dict(), save)

        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(
            self.model.last_linear.in_features, n_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x

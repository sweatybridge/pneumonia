"""
Script containing commonly used functions.
"""
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pretrainedmodels import se_resnext50_32x4d
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Class for X-ray dataset."""

    # pylint: disable=too-many-arguments
    def __init__(self, root_dir, transform=None, target="COVID-19"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        # taking only PA view
        meta_path = self.root_dir / "metadata.csv"
        self.df = pd.read_csv(meta_path).query("view == 'PA'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.root_dir / self.df["folder"].iloc[idx] / self.df["filename"].iloc[idx]
        image = cv2.imread(file_path)

        if self.transform:
            image = self.transform(image)

        if self.df["finding"].iloc[idx] != target:
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


class CustomSEResNeXt(nn.Module):
    """CustomSEResNeXt"""

    def __init__(self, n_classes=2, pretrained=None, weights=None):
        super().__init__()

        self.model = se_resnext50_32x4d(pretrained=pretrained)
        if weights is not None:
            self.model.load_state_dict(weights)

        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(
            self.model.last_linear.in_features, n_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x

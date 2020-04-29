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

from senet import se_resnext50_32x4d


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        return transform.resize(image, (new_h, new_w))


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        return image[top: top + new_h,
                     left: left + new_w]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
            image /= 255.
        
        return torch.from_numpy(image).float()


class ImageDataset(Dataset):
    """Class for X-ray dataset."""

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

        file_path = os.path.join(self.root_dir, self.image_dir, self.df["filename"].iloc[idx])
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
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def gcs_imread(bucket, blob_path):
    blob = bucket.blob(blob_path)
    image = cv2.imdecode(np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), 1)
    return image


def gcs_imwrite(bucket, blob_path, filename, image):
    cv2.imwrite(filename, image)
    bucket.blob(blob_path).upload_from_filename(filename)


class CustomSEResNeXt(nn.Module):

    def __init__(self, weights_path, device, n_classes=2, save=None):
        super().__init__()

        self.model = se_resnext50_32x4d(pretrained=None)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        if save is not None:
            torch.save(self.model.state_dict(), save)

        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x

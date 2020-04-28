import os
import random
import time

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from skimage import transform


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

    def __init__(self, root_dir, image_path, metadata_path="metadata.csv", transform=None):
        self.root_dir = root_dir
        self.image_path = image_path
        self.transform = transform
        self.df = (
            pd.read_csv(os.path.join(root_dir, metadata_path))
            .query("view == 'PA'")  # taking only PA view
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image = cv2.imread(os.path.join(self.root_dir, self.image_path, self.df["filename"].iloc[idx]))
        
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

"""
Script to preprocess images.
"""
import os

import cv2
import numpy as np
import pandas as pd
from skimage import transform
from torchvision import transforms

BUCKET = os.getenv("BUCKET")


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


def main():
    """Preprocess."""
    transformations = transforms.Compose([Rescale(256),
                                          RandomCrop(224)])

    df = pd.read_csv(os.path.join(BUCKET, "metadata.csv"), usecols=["filename"])

    for i, (filename,) in df.iterrows():
        image = cv2.imread(os.path.join(BUCKET, "images", filename))
        proc_image = transformations(image)
        proc_image.imwrite(os.path.join(BUCKET, "proc_images", filename))


if __name__ == "__main__":
    main()

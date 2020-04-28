"""
Script to preprocess images.
"""
import os
import time

import cv2
import numpy as np
import pandas as pd
from torchvision import transforms

from utils import Rescale, RandomCrop

BUCKET = os.getenv("BUCKET")


def preprocess():
    """Preprocess."""
    print("Transform images")
    transformations = transforms.Compose([Rescale(256),
                                          RandomCrop(224)])

    df = (
        pd.read_csv(os.path.join(BUCKET, "metadata.csv"), usecols=["filename", "view"])
        .query("view == 'PA'")  # taking only PA view
    )

    start_time = time.time()
    for filename in df["filename"].tolist():
        image = cv2.imread(os.path.join(BUCKET, "images", filename))
        proc_image = transformations(image)
        proc_image = (proc_image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(BUCKET, "proc_images", filename), proc_image)

    print(f"  Time taken = {time.time() - start_time:.2f}s")
    print(f"  Number of images processed = {df.shape[0]}")


if __name__ == "__main__":
    preprocess()

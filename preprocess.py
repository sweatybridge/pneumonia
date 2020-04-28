"""
Script to preprocess images.
"""
import os
import time

import numpy as np
import pandas as pd
from torchvision import transforms
from google.cloud import storage

from utils import Rescale, RandomCrop, gcs_imread, gcs_imwrite

PROJECT = "span-production"
BUCKET = "bedrock-sample"
BASE_DIR = os.getenv("BASE_DIR")
BASE_PATH = f"gs://{BUCKET}/{BASE_DIR}"


def preprocess():
    """Preprocess."""
    print("Transform images")
    client = storage.Client(PROJECT)
    bucket = client.get_bucket(BUCKET)

    transformations = transforms.Compose([Rescale(256),
                                          RandomCrop(224)])

    df = (
        pd.read_csv(os.path.join(BASE_PATH, "metadata.csv"), usecols=["filename", "view"])
        .query("view == 'PA'")  # taking only PA view
    )

    start_time = time.time()
    for filename in df["filename"].tolist():
        image = gcs_imread(bucket, os.path.join(BASE_DIR, "images", filename))

        proc_image = transformations(image)
        proc_image = (proc_image * 255).astype(np.uint8)

        gcs_imwrite(bucket, os.path.join(BASE_DIR, "proc_images", filename), filename, proc_image)

    print(f"  Time taken = {time.time() - start_time:.2f}s")
    print(f"  Number of images processed = {df.shape[0]}")


if __name__ == "__main__":
    preprocess()

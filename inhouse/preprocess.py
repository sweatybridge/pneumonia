"""
Script to preprocess images.
"""
import os
import time

import numpy as np
import pandas as pd
from torchvision import transforms
from google.cloud import storage

PROJECT = os.getenv('PROJECT')
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR')
RAW_BUCKET = os.getenv('RAW_BUCKET')
BUCKET = os.getenv('BUCKET')
BASE_DIR = os.getenv('BASE_DIR')
PREPROCESSED_DIR = os.getenv('PREPROCESSED_DIR')


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


def preprocess():
    """Preprocess."""
    print("Transform images")
    client = storage.Client(PROJECT)
    read_bucket = client.get_bucket(RAW_BUCKET)
    write_bucket = client.get_bucket(BUCKET)

    transformations = transforms.Compose([transforms.Rescale(256),
                                          transforms.RandomCrop(224)])

    metadata_df = (
        pd.read_csv(f"gs://{RAW_BUCKET}/{RAW_DATA_DIR}/metadata.csv", usecols=["filename", "view"])
        .query("view == 'PA'")  # taking only PA view
    )

    start_time = time.time()
    for filename in metadata_df["filename"].tolist():
        image = gcs_imread(read_bucket, os.path.join(RAW_DATA_DIR, "images", filename))

        proc_image = transformations(image)
        proc_image = (proc_image * 255).astype(np.uint8)

        gcs_imwrite(
            write_bucket, os.path.join(BASE_DIR, PREPROCESSED_DIR, filename), filename, proc_image)

    print(f"  Time taken = {time.time() - start_time:.2f}s")
    print(f"  Number of images processed = {metadata_df.shape[0]}")


if __name__ == "__main__":
    preprocess()

"""
Script for serving.
"""
import base64

import cv2
import numpy as np
import six
import torch
from torch.nn import functional as F
from torchvision import transforms
from flask import Flask, request

from utils import Rescale, RandomCrop, ToTensor, CustomSEResNeXt, seed_torch

# MODEL_DIR = "/artefact/"
MODEL_DIR = "models/"

DEVICE = torch.device("cpu")
MODEL = CustomSEResNeXt(MODEL_DIR + "pretrained_model.pth", DEVICE)
MODEL.load_state_dict(torch.load(MODEL_DIR + "finetuned_model.pth", map_location=DEVICE))
MODEL.eval()


def decode_image(field, dtype=np.uint8):
    """Decode a base64 encoded image to a list of floats.
    Args:
        field: base64 encoded string or bytes
        dtype
    Returns:
        numpy.array
    """
    if field is None:
        return None
    if not isinstance(field, bytes):
        field = six.b(field)
    array = np.frombuffer(base64.b64decode(field), dtype=dtype)
    image_array = cv2.imdecode(array, cv2.IMREAD_ANYCOLOR)
    return image_array


def predict(request_json):
    """Predict function."""
    seed_torch(seed=42)

    image = decode_image(request_json["encoded_image"])
    proc_image = transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor(),
    ])(image)
    proc_image = proc_image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(proc_image)

    return F.softmax(logits, dim=1).cpu().numpy()[0, 1].item()


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_prob():
    """Returns probability."""
    return {"prob": predict(request.json)}


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()

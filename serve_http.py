"""
Script for serving.
"""
import base64

import numpy as np
import six
import torch
from torch.nn import functional as F
from torchvision import transforms
from flask import Flask, request

from utils import Rescale, RandomCrop, ToTensor, CustomSEResNeXt, seed_torch

device = torch.device("cpu")
model = CustomSEResNeXt("/artefact/pretrained_model.pth", device)
model.load_state_dict(torch.load("/artefact/finetuned_model.pth", map_location=device))
model.eval()


def decode_image(field, dtype=np.uint8):
    """Decode a base64 encoded numpy array to a list of floats.
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
    return array


def predict(request_json):
    """Predict function."""
    seed_torch(seed=42)
    
    image = decode_image(request_json["encoded_image"]).reshape(
        request_json["image_shape"])
    
    proc_image = transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor(),
    ])(image)
    proc_image = proc_image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(proc_image)

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

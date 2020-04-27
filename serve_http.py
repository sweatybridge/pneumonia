"""
Script for serving.
"""
import json

import torch
from torch.nn import functional as F
from torchvision import transforms
from flask import Flask, request

from preprocess import Rescale, RandomCrop
from train import CustomSEResNeXt, CFG, ToTensor

BUCKET = "gs://bedrock-sample/chestxray/"


def load_model(device):
    model = CustomSEResNeXt()
    model.load_state_dict(torch.load(CFG.model_path, map_location=device))
    return model


def predict(image, model, device):
    model.to(device)
    model.eval()

    image = image.unsqueeze(0).to(device)
    proc_image = transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor(),
    ])(image)

    with torch.no_grad():
        logits = model(proc_image)

    return F.softmax(logits, dim=1).cpu().numpy()[0, 1]


# pylint: disable=invalid-name
app = Flask(__name__)
device = torch.device("cpu")
model = load_model(device)


@app.route("/", methods=["POST"])
def get_churn():
    """Returns the `churn_prob` given the subscriber features"""

    # TODO: what is the input?
    image = request.json
    result = {
        "prob": predict(image, model, device)
    }
    return result


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()

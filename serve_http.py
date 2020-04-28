"""
Script for serving.
"""
import torch
from torch.nn import functional as F
from torchvision import transforms
from flask import Flask, request

from train import CustomSEResNeXt
from utils import Rescale, RandomCrop, ToTensor

device = torch.device("cpu")
model = CustomSEResNeXt()
model.load_state_dict(torch.load("artefact/train/trained_model.pth", map_location=device))


def predict(image):
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


@app.route("/", methods=["POST"])
def get_churn():
    """Returns the `churn_prob` given the subscriber features"""

    # TODO: what is the input?
    image = request.json
    return {"prob": predict(image)}


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()

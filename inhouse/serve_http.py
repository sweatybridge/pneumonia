"""
Script for serving.
"""
import os
import logging
from pathlib import Path

import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torchvision import transforms
from flask import Flask, request
from captum.attr import GuidedGradCam, visualization

from gradcam_pytorch import GradCam
from utils import CustomSEResNeXt
from utils_image import encode_image, superimpose_heatmap

LOGGER = logging.getLogger()

DEVICE = torch.device("cpu")
MODEL_DIR = Path("models" if os.path.exists("models") else "/artefact")
WEIGHTS = torch.load(MODEL_DIR / "model.pth", map_location=DEVICE)
MODEL = CustomSEResNeXt()
MODEL.load_state_dict(WEIGHTS)
MODEL.eval()

CLASS_NAMES = ["Normal", "COVID-19"]
TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

GRAD_CAM = GradCam(
    model=MODEL.model,
    feature_module=MODEL.model.layer4,
    target_layer_names=["2"],
    use_cuda=False,
)

GUIDED_GC = GuidedGradCam(MODEL.model, MODEL.model.layer4)


def pre_process(files):
    if files["image"].filename.lower().endswith(".npy"):
        img = np.load(files["image"])
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[0] == 1):
            # Convert grayscale to rgb
            img = np.tile(img, (3, 1, 1)).transpose(1, 2, 0)
    else:
        # Use opencv to open the image file
        img = np.frombuffer(files["image"].read(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = decode_image(request.json["encoded_image"])
    return TRANSFORM(img).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict(features):
    score = MODEL(features)
    prob = F.softmax(score, dim=1).detach()[0].tolist()
    output = {k: v for k, v in zip(CLASS_NAMES, prob)}
    return output


def _get_target_index(target):
    if target.isnumeric():
        index = int(target)
        if index < len(CLASS_NAMES):
            return index
    return CLASS_NAMES.index(target)


def explain(features, target):
    """Perform XAI."""
    # Grad-CAM
    img = features.detach().numpy().squeeze().transpose((1, 2, 0))
    mask, score = GRAD_CAM(features, target_category=target)
    cam_img = superimpose_heatmap(img, mask)

    # Get the score for target class
    if target is None:
        target = score.argmax()
        prob = F.softmax(score, dim=1).detach().max().item()
    else:
        prob = F.softmax(score, dim=1).detach()[0, target].item()

    # Guided Grad-CAM
    gc_attribution = GUIDED_GC.attribute(features, target=target)
    try:
        gc_norm_attr = visualization._normalize_image_attr(
            gc_attribution.detach().squeeze().cpu().numpy().transpose((1, 2, 0)),
            sign=visualization.VisualizeSign.absolute_value.name,
        )
    except AssertionError as exc:
        LOGGER.warning("Failed to compute guided GC", exc_info=exc)
        gc_norm_attr = np.zeros(img.shape)
    gc_img = superimpose_heatmap(img, gc_norm_attr)

    return {
        "prob": prob,
        "cam_image": encode_image(cam_img),
        "gc_image": encode_image(gc_img),
    }


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
@app.route("/predict", methods=["POST"])
def get_prob():
    """Returns probability."""
    features = pre_process(request.files)
    return predict(features)


@app.route("/explain/", defaults={"target": None}, methods=["POST"])
@app.route("/explain/<target>", methods=["POST"])
def get_heatmap(target):
    features = pre_process(request.files)
    if target is not None:
        target = _get_target_index(target)
    return explain(features=features, target=target)


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()

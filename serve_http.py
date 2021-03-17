"""
Script for serving.
"""
import os

import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torchvision import transforms
from flask import Flask, request
from captum.attr import GuidedGradCam, IntegratedGradients, visualization

from gradcam_pytorch import GradCam
from utils import CustomSEResNeXt
from utils_image import (
    encode_image,
    decode_image,
    superimpose_heatmap,
    get_heatmap,
)

MODEL_DIR = "/artefact/"
if os.path.exists("models/"):
    MODEL_DIR = "models/"

DEVICE = torch.device("cpu")
MODEL = CustomSEResNeXt(MODEL_DIR + "pretrained_model.pth", DEVICE)
MODEL.load_state_dict(
    torch.load(MODEL_DIR + "finetuned_model.pth", map_location=DEVICE)
)
MODEL.eval()

PROCESS = transforms.Compose(
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
IG = IntegratedGradients(MODEL.model)


def predict(proc_image):
    """Perform XAI."""
    # Grad-CAM
    img = proc_image.detach().numpy().squeeze().transpose((1, 2, 0))
    mask, score = GRAD_CAM(proc_image)
    cam_img = superimpose_heatmap(img, mask)

    # Guided Grad-CAM
    target = score.argmax()
    gc_attribution = GUIDED_GC.attribute(proc_image, target=target)
    gc_norm_attr = visualization._normalize_image_attr(
        gc_attribution.detach().squeeze().cpu().numpy().transpose((1, 2, 0)),
        sign="absolute_value",
        outlier_perc=2,
    )
    gc_img = superimpose_heatmap(img, gc_norm_attr)

    # IntegratedGradients
    # ig_attribution = IG.attribute(proc_image, target=target, n_steps=20)
    # ig_norm_attr = visualization._normalize_image_attr(
    #     ig_attribution.detach().squeeze().cpu().numpy().transpose((1, 2, 0)),
    #     sign="absolute_value",
    #     outlier_perc=2,
    # )
    # ig_img = get_heatmap(ig_norm_attr)

    return {
        "prob": F.softmax(score, dim=1).detach().numpy()[0, 1].item(),
        "cam_image": encode_image(cam_img),
        "gc_image": encode_image(gc_img),
        # "ig_image": encode_image(ig_img),
    }


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_prob():
    """Returns probability."""
    img = np.frombuffer(request.files["image"].read(), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = decode_image(request.json["encoded_image"])
    proc_image = PROCESS(img).unsqueeze(0).to(DEVICE)
    output = predict(proc_image)
    return output


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()

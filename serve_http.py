"""
Script for serving.
"""
import os

import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision import transforms
from flask import Flask, request
from PIL import Image
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz

from gradcam_pytorch import GradCam
from utils import Rescale, RandomCrop, ToTensor, CustomSEResNeXt, seed_torch
from utils_image import encode_image, decode_image, superimpose_heatmap, fig2img

MODEL_DIR = "/artefact/"
if os.path.exists("models/"):
    MODEL_DIR = "models/"

DEVICE = torch.device("cpu")
MODEL = CustomSEResNeXt(MODEL_DIR + "pretrained_model.pth", DEVICE)
MODEL.load_state_dict(torch.load(MODEL_DIR + "finetuned_model.pth", map_location=DEVICE))
MODEL.eval()

GRAD_CAM = GradCam(model=MODEL.model,
                   feature_module=MODEL.model.layer4,
                   target_layer_names=["2"],
                   use_cuda=False)

GUIDED_GC = GuidedGradCam(MODEL.model, MODEL.model.layer4)


def get_grad_cam(proc_image, target):
    mask = GRAD_CAM(proc_image)
    img = proc_image.detach().numpy().squeeze().transpose((1, 2, 0))
    cam = superimpose_heatmap(img, mask)

    attribution = GUIDED_GC.attribute(proc_image, target=target)
    norm_attr = viz._normalize_image_attr(
        attribution.detach().squeeze().cpu().numpy().transpose((1, 2, 0)),
        sign="absolute_value", outlier_perc=2
    ).tolist()
    return cam, norm_attr


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
        prob = F.softmax(logits, dim=1).cpu().numpy()[0, 1].item()

    target = 1 if prob > 0.5 else 0
    cam, norm_attr = get_grad_cam(proc_image, target)
    cam_image = encode_image(Image.fromarray(cam))

    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(norm_attr)
    gc_image = encode_image(fig2img(fig))
    return prob, cam_image, gc_image


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_prob():
    """Returns probability."""
    prob, cam_image, gc_image = predict(request.json)
    return {
        "prob": prob,
        "cam_image": cam_image,
        "gc_image": gc_image,
    }


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()

from dataclasses import asdict
from os.path import exists
import re

import cv2
import numpy as np
import torch
from bedrock_client.bedrock.model import BaseModel
from captum.attr import GuidedGradCam, IntegratedGradients
from prometheus_client import Gauge
from torchvision.models import densenet121
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)

from gradcam_pytorch import GradCam
from utils_image import (
    encode_image,
    decode_image,
    superimpose_heatmap,
    get_heatmap,
    normalize_image_attr,
)

backlog_length_gauge = Gauge("inference_backlog_length", "Length of inference backlog")
max_batch_size = 32


class Model(BaseModel):
    PATTERN = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    CKPT_PATH = "model.pth.tar"
    N_CLASSES = 14
    CLASS_NAMES = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = densenet121(pretrained=True)
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.classifier.in_features, self.N_CLASSES),
            torch.nn.Sigmoid(),
        )

        path = f"/artefact/{self.CKPT_PATH}"
        if not exists(path):
            path = self.CKPT_PATH

        state_dict = torch.load(path, map_location=self.device)["state_dict"]
        # Rename original model weights if necessary
        state_dict = {
            k.replace("module.densenet121.", ""): v for k, v in state_dict.items()
        }
        for key in list(state_dict.keys()):
            res = self.PATTERN.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            # self.model.half()

        self.transform = Compose(
            [
                ToPILImage(),
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.grad_cam = GradCam(
            model=self.model,
            feature_module=self.model.features,
            target_layer_names=["denseblock4"],
            use_cuda=torch.cuda.is_available(),
        )
        self.guided_gc = GuidedGradCam(
            self.model, self.model.features[-2]["denselayer16"].conv2
        )
        self.ig = IntegratedGradients(self.model)

    def pre_process(self, files, http_body=None):
        img = np.frombuffer(files["image"].read(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = self.transform(img).to(self.device)
        # if self.device.type == "cuda":
        #     features = features.half()
        return features.unsqueeze(0)

    @torch.no_grad()
    def _predict(self, features):
        score = self.model(features).softmax(dim=1).max(dim=1)
        return [{"type": score.indices[0].item(), "conf": score.values[0].item()}]

    def predict(self, features):
        """Perform XAI."""
        # Grad-CAM
        img = features.detach().numpy().squeeze().transpose((1, 2, 0))
        mask, score = self.grad_cam(features)
        cam_img = superimpose_heatmap(img, mask)

        # Guided Grad-CAM
        target = score.argmax()
        gc_attribution = self.guided_gc.attribute(features, target=target)
        gc_norm_attr = normalize_image_attr(
            gc_attribution.detach().squeeze().cpu().numpy().transpose((1, 2, 0)),
            sign="absolute_value",
            outlier_perc=2,
        )
        gc_img = get_heatmap(gc_norm_attr)

        # IntegratedGradients
        ig_attribution = self.ig.attribute(features, target=target, n_steps=20)
        ig_norm_attr = normalize_image_attr(
            ig_attribution.detach().squeeze().cpu().numpy().transpose((1, 2, 0)),
            sign="absolute_value",
            outlier_perc=2,
        )
        ig_img = get_heatmap(ig_norm_attr)

        return [
            {
                "prob": score.max().item(),
                "cam_image": encode_image(cam_img),
                "gc_image": encode_image(gc_img),
                "ig_image": encode_image(ig_img),
            }
        ]

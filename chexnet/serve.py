from dataclasses import asdict
from logging import getLogger
from os.path import exists
import re

import cv2
import numpy as np
import torch
from bedrock_client.bedrock.model import BaseModel
from captum.attr import GuidedGradCam, visualization
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
)


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

    def __init__(self, logger):
        self.logger = logger or getLogger()
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

        self.transform = Compose(
            [ToPILImage(), Resize(256), CenterCrop(224), ToTensor()]
        )
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.grad_cam = GradCam(
            model=self.model,
            feature_module=self.model.features,
            target_layer_names=["denseblock4"],
            use_cuda=torch.cuda.is_available(),
        )
        self.guided_gc = GuidedGradCam(
            self.model, self.model.features[-2]["denselayer16"].conv2
        )

    def pre_process(self, files, http_body=None):
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
        features = self.transform(img).to(self.device)
        return features.unsqueeze(0)

    @torch.no_grad()
    def predict(self, features):
        score = self.model(self.norm(features))
        output = {k: v for k, v in zip(self.CLASS_NAMES, score[0].tolist())}
        return [output]

    def _get_target_index(self, target):
        if target.isnumeric():
            index = int(target)
            if index < len(self.CLASS_NAMES):
                return index
        return self.CLASS_NAMES.index(target)

    def explain(self, features, target):
        """Perform XAI."""
        if target is not None:
            target = self._get_target_index(target)

        # Grad-CAM
        img = features.detach().numpy().squeeze().transpose((1, 2, 0))
        features = self.norm(features)
        mask, score = self.grad_cam(features, target_category=target)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        cam_img = superimpose_heatmap(img, mask)

        # Get the score for target class
        if target is None:
            target = score.argmax()
            prob = score.max().item()
        else:
            prob = score[0][target].item()

        # Guided Grad-CAM
        gc_attribution = self.guided_gc.attribute(features, target=target)
        try:
            gc_norm_attr = visualization._normalize_image_attr(
                gc_attribution.detach().squeeze().cpu().numpy().transpose((1, 2, 0)),
                sign=visualization.VisualizeSign.absolute_value.name,
            )
        except AssertionError as exc:
            self.logger.warning("Failed to compute guided GC", exc_info=exc)
            gc_norm_attr = np.zeros(img.shape)
        gc_img = superimpose_heatmap(img, gc_norm_attr)

        return [
            {
                "prob": prob,
                "cam_image": encode_image(cam_img),
                "gc_image": encode_image(gc_img),
            }
        ]

from logging import getLogger
from os.path import exists

import cv2
import numpy as np
import tensorflow as tf
from bedrock_client.bedrock.model import BaseModel

from utils_image import encode_image, superimpose_heatmap


class Model(BaseModel):
    CKPT_PATH = "model.pth.tar"
    N_CLASSES = 3
    CLASS_NAMES = [
        "LOS>2",
        "Mortality_Risk",
        "ICU_Admission",
    ]

    def __init__(self, logger=None):
        self.logger = logger or getLogger()

        # TODO: replace with CAPE model weights
        self.model = tf.keras.applications.xception.Xception(classifier_activation=None)
        self.transform = tf.keras.applications.xception.preprocess_input
        self.decode = tf.math.sigmoid
        # self.decode = keras.applications.xception.decode_predictions

        path = f"/artefact/{self.CKPT_PATH}"
        if not exists(path):
            path = self.CKPT_PATH

        self.grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer("block14_sepconv2_act").output, self.model.output]
        )

    def pre_process(self, files, http_body=None):
        image = files["image"]
        if hasattr(image, "filename") and image.filename.lower().endswith(".npy"):
            img = np.load(image)
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[0] == 1):
                # Convert grayscale to rgb
                img = np.tile(img, (3, 1, 1)).transpose(1, 2, 0)
        else:
            # Use opencv to open the image file
            img = np.frombuffer(image.read(), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_ANYCOLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # features = self.transform(img).to(self.device)
        # return features.unsqueeze(0)
        features = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(features, (299, 299))[tf.newaxis, ...]

    def predict(self, features):
        logits = self.model(self.transform(features))
        score = self.decode(logits)
        output = {k: v.item() for k, v in zip(self.CLASS_NAMES, score[0].numpy())}
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

        # Grad-CAM: https://github.com/keras-team/keras-io/blob/master/examples/vision/grad_cam.py#L72
        img = features[0].numpy()
        features = self.transform(features)
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = self.grad_model(features)
            if target is None:
                target = tf.argmax(preds[0])
            class_channel = preds[:, target]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        mask = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
        cam_img = superimpose_heatmap(img, mask)

        # Get the score for target class
        prob = self.decode(preds[0, target]).numpy().item()

        # Guided Grad-CAM
        # gc_attribution = self.guided_gc.attribute(features, target=target)
        # try:
        #     gc_norm_attr = visualization._normalize_image_attr(
        #         gc_attribution.detach().squeeze().cpu().numpy().transpose((1, 2, 0)),
        #         sign=visualization.VisualizeSign.absolute_value.name,
        #     )
        # except AssertionError as exc:
        #     self.logger.warning("Failed to compute guided GC", exc_info=exc)
        #     gc_norm_attr = np.zeros(img.shape)
        # gc_img = superimpose_heatmap(img, gc_norm_attr)

        return [
            {
                "prob": prob,
                "cam_image": encode_image(cam_img),
                "gc_image": encode_image(cam_img),
            }
        ]

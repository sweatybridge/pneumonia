"""
Functions for encoding and decoding images.
"""
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def encode_image(image):
    """Encode an image to base64 encoded bytes.
    Args:
        image: PIL.PngImagePlugin.PngImageFile
    Returns:
        base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="png")
    base64_bytes = base64.b64encode(buffered.getvalue())
    return base64_bytes.decode("utf-8")


def decode_image(field):
    """Decode a base64 encoded image to a list of floats.
    Args:
        field: base64 encoded string
    Returns:
        numpy.array
    """
    array = np.frombuffer(base64.b64decode(field), dtype=np.uint8)
    image_array = cv2.imdecode(array, cv2.IMREAD_ANYCOLOR)
    # output_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_array


def superimpose_heatmap(img, mask):
    """Superimpose mask heatmap on image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image."""
    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-8)
    img = img * 0.05
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.clip(img * 255, 0, 255).astype('uint8')

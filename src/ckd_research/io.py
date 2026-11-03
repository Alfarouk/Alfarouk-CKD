"""Checked local image I/O and overlays; no input-file overwrites."""
from pathlib import Path
from PIL import Image
import numpy as np
from .contracts import COLORS, MAX_PIXELS, validate_mask

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def load_image(path):
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image exceeds16 million pixels")
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Normalize image and mask orientation together before analysis")
        return np.array(image.convert("L"))


def load_mask(path):
    with Image.open(path) as image:
        if image.mode not in {"L", "P", "I", "I;16"}:
            raise ValueError("Use an integer grayscale or indexed label mask, not a color overlay")
        return validate_mask(np.array(image))


def render_overlay(image, mask, alpha=0.48):
    mask = validate_mask(mask)
    if image.shape != mask.shape:
        raise ValueError("Image and mask shapes differ")
    if not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must be between zero and one")
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    foreground = mask > 0
    rgb[foreground] = np.round((1 - alpha) * rgb[foreground] + alpha * COLORS[mask[foreground]]).astype(np.uint8)
    return rgb


def save_png(array, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array).astype(np.uint8)).save(path, format="PNG")

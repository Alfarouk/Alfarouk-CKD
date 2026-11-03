"""A generated engineering fixture, never a patient scan or trained prediction."""
import numpy as np
from .contracts import combine_source_masks


def fixture(size=256, seed=17):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    x, y = (xx - size * .52) / size, (yy - size * .52) / size
    whole = ((x / .29) ** 2 + (y / .19) ** 2 < 1)
    inside = ((x / .23) ** 2 + (y / .13) ** 2 < 1)
    center = ((x / .13) ** 2 + (y / .075) ** 2 < 1)
    regions = np.zeros((size, size), dtype=np.uint8)
    regions[whole] = 3
    regions[inside] = 2
    regions[center] = 1
    regions[whole & ~(((x / .28) ** 2 + (y / .18) ** 2) < 1)] = 0
    labels = combine_source_masks(whole.astype(np.uint8), regions)
    image = np.clip(24 + whole * 64 + inside * 34 + center * 45 + rng.normal(0, 13, whole.shape), 0, 255).astype(np.uint8)
    return image, labels

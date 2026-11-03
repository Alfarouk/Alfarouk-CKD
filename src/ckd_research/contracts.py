"""Explicit label and calibration contracts shared by every workflow."""
from dataclasses import dataclass
import math
import numpy as np

LABELS = {0: "background", 1: "region_1", 2: "region_2", 3: "region_3", 4: "capsule_only"}
MASK_SCHEMAS = {
    "corrected-five": LABELS,
    "source-four": {0: "background", 1: "source_class_1", 2: "source_class_2", 3: "source_class_3"},
}
COLORS = np.array([[0, 0, 0], [237, 158, 67], [65, 178, 153], [122, 119, 225], [76, 149, 224]], dtype=np.uint8)
MAX_PIXELS = 16_000_000


@dataclass(frozen=True)
class Spacing:
    """Physical row/column spacing for the actual array being measured."""
    row_mm: float
    column_mm: float

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (self.row_mm, self.column_mm)):
            raise ValueError("Pixel spacing must be finite and greater than zero")

    def resized(self, old_shape, new_shape):
        return Spacing(self.row_mm * old_shape[0] / new_shape[0], self.column_mm * old_shape[1] / new_shape[1])


def validate_mask(mask, *, label_ids=None):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or mask.size > MAX_PIXELS:
        raise ValueError("A non-empty two-dimensional label mask is required (maximum16 million pixels)")
    allowed = list(LABELS if label_ids is None else label_ids)
    if not np.issubdtype(mask.dtype, np.integer) or not np.isin(mask, allowed).all():
        raise ValueError("Label mask IDs do not match the selected annotation schema")
    return mask.astype(np.uint8)


def combine_source_masks(capsule, regions):
    """Preserve source region IDs; fill only the unlabelled capsule interior.

    Source capsule is a whole-organ binary mask, not an outer-rim class.
    Anatomy names remain neutral until the source annotator's schema is known.
    """
    capsule, regions = np.asarray(capsule), np.asarray(regions)
    if capsule.ndim != 2 or capsule.shape != regions.shape:
        raise ValueError("Capsule and region masks must have the same two-dimensional shape")
    if not np.isin(capsule, [0, 1, 255]).all() or not np.isin(regions, [0, 1, 2, 3]).all():
        raise ValueError("Expected binary capsule mask and region IDs 0, 1, 2, 3")
    if np.any((regions > 0) & (capsule == 0)):
        raise ValueError("Region labels extend outside the capsule; inspect the source annotations")
    result = regions.astype(np.uint8).copy()
    result[(capsule > 0) & (regions == 0)] = 4
    return result

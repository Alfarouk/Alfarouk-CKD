"""Mask geometry and reproducible overlap metrics, without diagnosis rules."""
import math
import cv2
import numpy as np
from .contracts import LABELS, Spacing, validate_mask


def overlap(prediction, reference):
    prediction, reference = validate_mask(prediction), validate_mask(reference)
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference shapes differ; no silent metric resize")
    values = {}
    for label, name in LABELS.items():
        a, b = prediction == label, reference == label
        union, total = int(np.count_nonzero(a | b)), int(a.sum() + b.sum())
        values[name] = {"dice": None if not total else float(2 * np.count_nonzero(a & b) / total),
                        "iou": None if not union else float(np.count_nonzero(a & b) / union),
                        "reference_pixels": int(b.sum())}
    present = [v["dice"] for k, v in values.items() if k != "background" and v["dice"] is not None]
    return {"per_class": values, "foreground_macro_dice": float(np.mean(present)) if present else None,
            "absent_class_rule": "Both masks absent: null, excluded from macro average"}


def measure(mask, spacing: Spacing | None = None, depth_mm: float | None = None):
    mask = validate_mask(mask)
    foreground = (mask > 0).astype(np.uint8)
    count = int(foreground.sum())
    per_class = {name: int(np.count_nonzero(mask == label)) for label, name in LABELS.items()}
    result = {"label_pixels": per_class, "foreground_pixels": count,
              "coverage_fraction": count / mask.size, "units": "pixels", "calibration": None,
              "major_axis": None, "minor_axis": None, "area": 0.0,
              "volume_mm3": None, "diagnosis": None,
              "interpretation": "Research geometry only; no disease or severity inference"}
    if depth_mm is not None and (not math.isfinite(depth_mm) or depth_mm <= 0):
        raise ValueError("An independently measured positive depth is required for volume")
    if not count:
        result["status"] = "empty_mask"
        return result
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.concatenate(contours).astype(np.float32)
    if spacing:
        points[:, :, 0] *= spacing.column_mm
        points[:, :, 1] *= spacing.row_mm
        result.update(units="mm", calibration={"row_mm": spacing.row_mm, "column_mm": spacing.column_mm})
    _, (width, height), _ = cv2.minAreaRect(points)
    major, minor = max(width, height), min(width, height)
    result.update(status="measured", major_axis=float(major), minor_axis=float(minor),
                  area=float(count * spacing.row_mm * spacing.column_mm if spacing else count),
                  area_units="mm2" if spacing else "pixels2",
                  component_count=len(contours),
                  axis_method="Minimum rotated rectangle of all foreground contour points")
    if depth_mm is not None:
        if spacing is None:
            raise ValueError("Volume requires calibrated image spacing as well as measured depth")
        result["volume_mm3"] = float(math.pi / 6 * major * minor * depth_mm)
        result["volume_method"] = "Ellipsoid approximation using independently supplied third dimension"
    return result

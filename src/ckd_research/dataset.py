"""Patient-separated data manifests, shared preprocessing and source import."""
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .contracts import combine_source_masks
from .io import load_image, load_mask, save_png


@dataclass(frozen=True)
class Sample:
    split: str
    patient_id: str
    image: Path
    mask: Path


def read_manifest(path, *, require_training=False):
    path = Path(path).resolve()
    base = path.parent
    samples, groups, image_hashes = [], {}, {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["split", "patient_id", "image", "mask"]:
            raise ValueError("Manifest columns must be split,patient_id,image,mask in that order")
        for row in reader:
            split, patient = row["split"], row["patient_id"].strip()
            if split not in {"train", "val", "test"} or not patient:
                raise ValueError("Every sample needs an explicit split and patient/group ID")
            if patient in groups and groups[patient] != split:
                raise ValueError("Patient/group occurs in more than one split")
            groups[patient] = split
            paths = [(base / row[key]).resolve() for key in ["image", "mask"]]
            if any(not p.is_relative_to(base) or not p.is_file() for p in paths):
                raise ValueError("Manifest files must exist inside its dataset directory")
            image, mask = load_image(paths[0]), load_mask(paths[1])
            if image.shape != mask.shape:
                raise ValueError("Every image and label mask must have identical dimensions")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in image_hashes and image_hashes[digest] != split:
                raise ValueError("Duplicate image content occurs across data splits")
            image_hashes[digest] = split
            samples.append(Sample(split, patient, *paths))
    if not samples:
        raise ValueError("Dataset manifest is empty")
    if require_training and not {"train", "val"}.issubset({s.split for s in samples}):
        raise ValueError("Training requires explicit non-empty train and val splits")
    return samples


def preprocess_image(image, size):
    return np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def import_triplets(images, capsules, regions, destination):
    """Combine source masks without changing their spatial resolution or labels."""
    images, capsules, regions, destination = map(Path, [images, capsules, regions, destination])
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    if not files:
        raise ValueError("No supported source images found")
    rows = []
    for image in files:
        candidates = []
        for folder in [capsules, regions]:
            matches = [p for p in folder.iterdir() if p.stem == image.stem and p.suffix.lower() == ".png"]
            if len(matches) != 1:
                raise ValueError(f"Expected one PNG mask for {image.name} in {folder}")
            with Image.open(matches[0]) as mask:
                candidates.append(np.array(mask))
        combined = combine_source_masks(*candidates)
        raw = load_image(image)
        if raw.shape != combined.shape:
            raise ValueError("Source image and masks differ in size")
        if any((destination / folder / (image.stem + ".png")).exists() for folder in ["images", "masks"]):
            raise ValueError("Destination contains this image; choose an empty output directory")
        save_png(raw, destination / "images" / (image.stem + ".png"))
        save_png(combined, destination / "masks" / (image.stem + ".png"))
        rows.append(image.stem)
    return rows

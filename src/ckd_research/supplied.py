"""Hash-bound checkpoints and explicit research inference.

Model architecture is checked strictly. A tensor match does not establish class
meanings, preprocessing, parameterless operations or clinical performance.
"""
from contextlib import contextmanager
import gc
import hashlib
from importlib.resources import files
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading

import cv2
import numpy as np
from PIL import Image

from .contracts import MAX_PIXELS

MODEL_IDS = {"source-swin-four", "biopsy-efficientnet", "sclerosis-convnext"}
_LOCK = threading.Lock()
PROFILES = {
    "source-clahe-256": {
        "title": "Grayscale + CLAHE · 256 × 256",
        "size": 256, "channels": 1, "decoder": "OpenCV grayscale",
        "resize": "stretch, OpenCV linear", "clahe": {"clip_limit": 2.0, "tile_grid": [8, 8]},
        "mean": [0.0], "std": [1.0],
        "provenance": "Grayscale and CLAHE inference profile; differs from the grayscale training profile.",
    },
    "source-training-256": {
        "title": "Grayscale · 256 × 256",
        "size": 256, "channels": 1, "decoder": "Pillow grayscale",
        "resize": "stretch, Pillow bicubic", "clahe": None,
        "mean": [0.0], "std": [1.0],
        "provenance": "Grayscale training profile; checkpoint data identity and annotation provenance remain unverified.",
    },
    "research-rgb-224": {
        "title": "Research choice: RGB, 224 × 224, ImageNet normalization",
        "size": 224, "channels": 3, "decoder": "Pillow RGB",
        "resize": "stretch, Pillow bicubic", "clahe": None,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "provenance": "Explicit engineering input choice. Original classifier resize, crop and normalization are unknown.",
    },
}
PROFILES["research-rgb-380"] = {
    **PROFILES["research-rgb-224"], "size": 380,
    "title": "Research choice: RGB, 380 × 380, ImageNet normalization",
}
PROFILES["research-rgb-224-unit"] = {
    **PROFILES["research-rgb-224"], "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
    "title": "Research choice: RGB, 224 × 224, unit range only",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions():
    result = {}
    for name in ["torch", "torchvision", "transformers", "safetensors", "numpy", "Pillow", "opencv-python-headless"]:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def registry():
    value = json.loads(files("ckd_research").joinpath("model_registry.json").read_text())
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or not isinstance(value.get("models"), list) or len(value["models"]) != 3
            or {s.get("id") for s in value["models"] if isinstance(s, dict)} != MODEL_IDS):
        raise ValueError("Unsupported model registry")
    for spec in value["models"]:
        for key, extension in [("original", ".pth"), ("weights", ".safetensors")]:
            asset = spec.get(key)
            if (not isinstance(asset, dict) or not isinstance(asset.get("name"), str)
                    or not re.fullmatch(r"[A-Za-z0-9_.-]+", asset["name"])
                    or not asset["name"].endswith(extension)
                    or not re.fullmatch(r"[a-f0-9]{64}", str(asset.get("sha256")))):
                raise ValueError("Registry assets need plain file names and exact SHA256 hashes")
        if (spec.get("task") not in {"segmentation", "classification"}
                or spec.get("class_count") != (4 if spec["id"] == "source-swin-four" else 2)
                or not isinstance(spec.get("input_profiles"), list)
                or not spec["input_profiles"] or any(p not in PROFILES for p in spec["input_profiles"])):
            raise ValueError("Registry task/input contract is incomplete")
    return value


def spec_for(model_id):
    try:
        return next(s for s in registry()["models"] if s["id"] == model_id)
    except StopIteration as exc:
        raise ValueError("Unknown model") from exc


def model_status(directory):
    directory = Path(directory)
    return [{**spec, "installed": (directory / spec["weights"]["name"]).is_file(),
             "profiles": [{"id": name, **PROFILES[name]} for name in spec["input_profiles"]]}
            for spec in registry()["models"]]


def build_model(model_id):
    """Offline, explicit library runtime configurations; never from_pretrained."""
    import torch
    if model_id == "source-swin-four":
        from .model import SourceSwinV2Segmentation
        model = SourceSwinV2Segmentation()
    elif model_id == "biopsy-efficientnet":
        from torchvision.models import efficientnet_b4
        model = efficientnet_b4(weights=None)
        # The checkpoint names a nested head. Its parameterless operation is
        # not encoded. Identity is an explicit research runtime assumption.
        model.classifier[1] = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(1792, 2))
    elif model_id == "sclerosis-convnext":
        from transformers import ConvNextConfig, ConvNextForImageClassification
        model = ConvNextForImageClassification(ConvNextConfig(
            num_labels=2, depths=[3, 3, 9, 3], hidden_sizes=[96, 192, 384, 768],
            hidden_act="gelu", layer_norm_eps=1e-12, layer_scale_init_value=1e-6))
    else:
        raise ValueError("Unknown model")
    return model.cpu().eval()


@contextmanager
def one_model():
    if not _LOCK.acquire(blocking=False):
        raise ValueError("Another model operation is running; try again when it finishes")
    try:
        import torch
        torch.set_num_threads(2)
        yield
    finally:
        gc.collect()
        _LOCK.release()


def _restricted_original(path, spec):
    import torch
    if file_hash(path) != spec["original"]["sha256"]:
        raise ValueError("Checkpoint hash differs from the registered asset")
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if spec["id"] == "source-swin-four":
        if unsafe != {"numpy.core.multiarray.scalar", "numpy.dtype"}:
            raise ValueError("Unexpected globals in the hash-bound Swin checkpoint")
        # Static inspection found only a NumPy float64 val_dice scalar.
        # No project function/class is trusted and unrestricted pickle is never used.
        allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
                   np.dtype, np.dtypes.Float64DType]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        return checkpoint["model_state_dict"]
    if unsafe:
        raise ValueError("Unexpected globals in tensor-only checkpoint")
    return torch.load(path, weights_only=True, map_location="cpu", mmap=True)


def _validate_tensors(state):
    import torch
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint must contain a non-empty tensor state dictionary")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Only named tensors are allowed in operational weights")
        if tensor.layout != torch.strided or not torch.isfinite(tensor).all().item():
            raise ValueError("Weights must be dense and finite")


def _publish_new_file(staged, target):
    """Same-filesystem hard-link creation is atomic and never replaces a name."""
    try:
        os.link(staged, target)
    except FileExistsError as exc:
        raise ValueError("Another file appeared at the model destination; it was preserved. Rerun to verify it or choose another folder.") from exc
    except OSError as exc:
        raise ValueError("This folder cannot publish a model without overwriting data; use a local filesystem that supports hard links") from exc


def convert_original(model_id, source, directory):
    """Explicit, hash-bound conversion. Originals and optimizer data stay intact."""
    from safetensors.torch import save_file
    spec = spec_for(model_id)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec["weights"]["name"]
    if target.resolve() == Path(source).resolve():
        raise ValueError("Conversion cannot overwrite its source")
    if target.exists():
        if spec["weights"].get("sha256") and file_hash(target) == spec["weights"]["sha256"]:
            return {"model_id": model_id, "path": str(target), "sha256": file_hash(target), "already_present": True}
        raise ValueError("Output already exists with different or unverified bytes")
    with tempfile.TemporaryDirectory(prefix="ckd-convert-", dir=directory) as temporary:
        staged = Path(temporary) / spec["weights"]["name"]
        with one_model():
            state = _restricted_original(Path(source), spec)
            _validate_tensors(state)
            model = build_model(model_id)
            try:
                model.load_state_dict(state, strict=True)
                save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(staged))
            finally:
                del model, state
        actual = file_hash(staged)
        if spec["weights"].get("sha256") and actual != spec["weights"]["sha256"]:
            raise ValueError("Converted weight hash differs from the registry")
        _publish_new_file(staged, target)
    return {"model_id": model_id, "path": str(target), "sha256": actual, "bytes": target.stat().st_size,
            "strict_tensor_match": True, "finite_weights": True, "unrestricted_pickle_used": False}


def _checked_image(content):
    if not isinstance(content, bytes) or not content or len(content) > 20_000_000:
        raise ValueError("Supply an image smaller than 20 MB")
    with Image.open(BytesIO(content)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image exceeds 16 million pixels")
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Normalize image orientation before model inference")
        image.load()
        return image.convert("RGB")


def prepare_input(content, profile_id):
    """The chosen input settings are recorded with every output."""
    if profile_id not in PROFILES:
        raise ValueError("Choose a supported explicit input profile")
    image = _checked_image(content)
    p = PROFILES[profile_id]
    if profile_id == "source-clahe-256":
        gray = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("OpenCV could not decode this source-profile image")
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        array = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        array = array[None]
    else:
        mode = "L" if p["channels"] == 1 else "RGB"
        array = np.asarray(image.convert(mode).resize((p["size"], p["size"]), Image.Resampling.BICUBIC),
                           dtype=np.float32) / 255.0
        array = array[None] if mode == "L" else array.transpose(2, 0, 1)
    mean = np.asarray(p["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(p["std"], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((array - mean) / std), np.asarray(image)


def run_model(model_id, content, directory, *, profile_id):
    from .pipeline import analyze, png_data
    spec = spec_for(model_id)
    if profile_id not in spec["input_profiles"]:
        raise ValueError("That input profile does not belong to this model")
    path = Path(directory) / spec["weights"]["name"]
    if not path.is_file():
        raise FileNotFoundError("Model files are not installed here; follow the README model setup steps")
    if file_hash(path) != spec["weights"]["sha256"]:
        raise ValueError("Operational model weight hash mismatch")
    array, rgb = prepare_input(content, profile_id)
    import torch
    from safetensors import SafetensorError
    from safetensors.torch import load_file
    with one_model():
        try:
            state = load_file(str(path), device="cpu")
        except SafetensorError as exc:
            raise ValueError("Operational tensor file is unreadable") from exc
        _validate_tensors(state)
        model = build_model(model_id)
        try:
            model.load_state_dict(state, strict=True)
            with torch.inference_mode():
                output = model(torch.from_numpy(array).unsqueeze(0))
                logits = output if spec["task"] == "segmentation" else getattr(output, "logits", output)
                if not torch.isfinite(logits).all().item():
                    raise ValueError("Model returned non-finite logits")
                if spec["task"] == "segmentation":
                    if list(logits.shape) != [1, 4, 256, 256]:
                        raise ValueError("Source segmentation output contract differs")
                    small = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    mask = np.asarray(Image.fromarray(small).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST))
                    gray = np.asarray(Image.fromarray(rgb).convert("L"))
                    result = analyze(gray, mask, source="supplied_four_class_model_output", label_schema_id="source-four")
                    result["report"]["logits_shape"] = list(logits.shape)
                    result["report"]["class_mean_softmax_scores"] = logits.softmax(1).mean((0, 2, 3)).tolist()
                else:
                    if list(logits.shape) != [1, 2]:
                        raise ValueError("Classifier output must have two raw indices")
                    result = {"image": png_data(rgb), "report": {
                        "source": "supplied_classifier_model_output", "task": "classification",
                        "raw_logits": logits[0].tolist(), "softmax_index_scores": logits.softmax(1)[0].tolist(),
                        "index_names": ["index_0_meaning_unknown", "index_1_meaning_unknown"],
                        "score_scope": "Relative model scores under the selected input profile; not calibrated probabilities of disease.",
                        "scope": "Research execution only; no diagnosis or clinical validation"}}
                result["report"].update(model_id=model_id, model_name=spec["title"],
                    weights_sha256=spec["weights"]["sha256"], original_sha256=spec["original"]["sha256"],
                    input_profile={"id": profile_id, **PROFILES[profile_id]},
                    runtime_config=spec["runtime_config"], semantic_limits=spec["semantic_limits"],
                    software_versions=software_versions(),
                    input_tensor_shape=[1, *array.shape], actual_checkpoint_executed=True)
                return result
        except RuntimeError as exc:
            raise ValueError("Operational weights/runtime do not match the recorded model contract") from exc
        finally:
            del model, state


def fetch_models(directory, *, include_originals=False):
    """Use the caller's gh authentication; never read or embed its credentials."""
    catalog = registry()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fetched = []
    for spec in catalog["models"]:
        for asset in [spec["weights"], *([spec["original"]] if include_originals else [])]:
            target = directory / asset["name"]
            if target.exists():
                if file_hash(target) != asset["sha256"]:
                    raise ValueError("Existing asset differs; choose an empty folder instead of overwriting it")
                fetched.append(str(target))
                continue
            with tempfile.TemporaryDirectory(prefix="ckd-model-", dir=directory) as temporary:
                process = subprocess.run(["gh", "release", "download", catalog["release"], "--repo", catalog["repository"],
                    "--pattern", asset["name"], "--dir", temporary], capture_output=True, text=True, timeout=600)
                if process.returncode:
                    raise ValueError("Private model download failed. Check gh sign-in and access to the existing private repository.")
                downloaded = Path(temporary) / asset["name"]
                if not downloaded.is_file() or file_hash(downloaded) != asset["sha256"]:
                    raise ValueError("Downloaded model asset hash mismatch")
                _publish_new_file(downloaded, target)
                fetched.append(str(target))
    return fetched

"""Swin V2 model and safe, explicit checkpoint manifests.

The default model preserves the source Swin V2 tiny backbone/five-stage
upsampling design. The corrected dataset contract has five output labels.
No download or unpickling is triggered merely by importing this module.
"""
import hashlib
import json
import re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Swinv2Config, Swinv2Model
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from .contracts import LABELS


def default_architecture():
    return {"image_size": 256, "embed_dim": 96, "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24], "window_size": 8,
            "decoder_channels": [512, 256, 128, 64, 32], "num_classes": 5}


def validate_architecture(value, *, expected_num_classes=5):
    expected = set(default_architecture())
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Unsupported architecture fields")
    size, embed = value["image_size"], value["embed_dim"]
    if type(size) is not int or size < 32 or size > 512 or size % 32:
        raise ValueError("Model image size must be a multiple of32 between32 and512")
    if (expected_num_classes not in {4, 5} or type(embed) is not int
            or not 12 <= embed <= 96 or value["num_classes"] != expected_num_classes):
        raise ValueError("Model output count does not match its explicit task contract")
    depths, heads, decoder = value["depths"], value["num_heads"], value["decoder_channels"]
    if (not all(isinstance(v, list) for v in [depths, heads, decoder])
            or len(depths) != 4 or any(type(n) is not int or not 1 <= n <= 6 for n in depths)
            or sum(depths) > 12 or len(heads) != 4
            or any(type(n) is not int or n < 1 or n > 24 or (embed * 2 ** i) % n for i, n in enumerate(heads))
            or len(decoder) != 5 or any(type(n) is not int or not 1 <= n <= 512 for n in decoder)
            or sum(decoder) > 1000 or type(value["window_size"]) is not int or not 2 <= value["window_size"] <= 8):
        raise ValueError("Unsupported or unbounded backbone/decoder configuration")
    return value


class _SwinV2Architecture(nn.Module):
    def __init__(self, architecture, *, num_classes, pretrained_backbone=None):
        super().__init__()
        self.architecture = validate_architecture(architecture, expected_num_classes=num_classes)
        a = self.architecture
        config = Swinv2Config(image_size=a["image_size"], embed_dim=a["embed_dim"], depths=a["depths"],
                              num_heads=a["num_heads"], window_size=a["window_size"], drop_path_rate=0.0)
        if pretrained_backbone:
            self.backbone = Swinv2Model.from_pretrained(pretrained_backbone, use_safetensors=True)
            if any(getattr(self.backbone.config, key) != getattr(config, key)
                   for key in ["image_size", "embed_dim", "depths", "num_heads", "window_size"]):
                raise ValueError("Pretrained backbone differs from the recorded architecture")
        else:
            self.backbone = Swinv2Model(config)
        channels = config.hidden_size
        blocks = []
        for out_channels in a["decoder_channels"]:
            blocks.append(nn.Sequential(nn.ConvTranspose2d(channels, out_channels, 2, 2),
                                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                         nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
            channels = out_channels
        self.decoder = nn.ModuleList(blocks)
        self.classifier = nn.Conv2d(channels, a["num_classes"], 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] not in {1, 3}:
            raise ValueError("Model expects NCHW grayscale or RGB inputs")
        spatial = x.shape[-2:]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Use the backbone's spatial output, never assume sqrt(token count).
        features = self.backbone(x, output_hidden_states=True).reshaped_hidden_states[-1]
        for block in self.decoder:
            features = block(features)
        logits = self.classifier(features)
        return F.interpolate(logits, size=spatial, mode="bilinear", align_corners=False)


class SwinV2Segmentation(_SwinV2Architecture):
    """Corrected five-label experiment contract; unchanged training API."""
    def __init__(self, architecture=None, *, pretrained_backbone=None):
        super().__init__(architecture or default_architecture(), num_classes=5,
                         pretrained_backbone=pretrained_backbone)


class SourceSwinV2Segmentation(_SwinV2Architecture):
    """Separate source-compatible four-output model, never a five-label alias."""
    def __init__(self):
        super().__init__({**default_architecture(), "num_classes": 4}, num_classes=4)


def save_checkpoint(model, directory, *, training_record):
    validate_architecture(model.architecture)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "model.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(weights))
    manifest = {"format_version": 1, "architecture": model.architecture, "labels": LABELS,
                "preprocessing": "grayscale_bilinear_resize_divide255", "weights": weights.name,
                "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "training_record": training_record, "clinical_validation": False}
    (directory / "model.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return directory / "model.json"


def load_checkpoint(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("Five-label experiment manifest is missing. Use run-model for the separately supplied checkpoints.")
    manifest = json.loads(manifest_path.read_text())
    if (not isinstance(manifest, dict) or manifest.get("format_version") != 1
            or manifest.get("preprocessing") != "grayscale_bilinear_resize_divide255"):
        raise ValueError("Unsupported model/preprocessing manifest")
    if manifest.get("labels") != {str(k): v for k, v in LABELS.items()}:
        raise ValueError("Checkpoint labels do not match the current five-class contract; no silent remapping")
    if (not isinstance(manifest.get("weights"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", str(manifest.get("sha256")))
            or not isinstance(manifest.get("training_record"), dict)):
        raise ValueError("Model manifest needs weight name/hash and a training record")
    validate_architecture(manifest.get("architecture"))
    weights = (manifest_path.parent / manifest["weights"]).resolve()
    if not weights.is_relative_to(manifest_path.parent) or weights.suffix != ".safetensors":
        raise ValueError("Weights must be a local safetensors file inside the model directory")
    if weights.stat().st_size > 1_000_000_000:
        raise ValueError("Checkpoint exceeds the one-gigabyte local model limit")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    model = SwinV2Segmentation(manifest["architecture"])
    try:
        model.load_state_dict(load_file(str(weights)), strict=True)
    except (SafetensorError, RuntimeError, ValueError) as exc:
        # These are user-supplied tensor bytes/parameter names, not code to run.
        raise ValueError("Checkpoint tensor data is unreadable or incompatible with its architecture") from exc
    return model.eval(), manifest


def import_legacy_checkpoint(path, destination, *, expected_sha256, architecture, training_record):
    """Explicit opt-in conversion of a trusted state dictionary, without pickle code.

    A source-era four-class checkpoint cannot be silently converted into the
    corrected five-class task. It must be retained separately and retrained.
    """
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Legacy checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = SwinV2Segmentation(architecture)
    model.load_state_dict(state, strict=True)
    return save_checkpoint(model, destination, training_record=training_record)

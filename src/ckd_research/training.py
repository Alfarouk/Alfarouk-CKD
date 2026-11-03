"""Bounded, patient-separated training; no advertised medical performance."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from .analysis import overlap
from .dataset import read_manifest, preprocess_image
from .io import load_image, load_mask
from .model import SwinV2Segmentation, save_checkpoint


class SegmentationDataset(Dataset):
    def __init__(self, samples, size):
        self.samples, self.size = samples, size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = preprocess_image(load_image(sample.image), self.size)
        mask = np.asarray(Image.fromarray(load_mask(sample.mask)).resize((self.size, self.size), Image.Resampling.NEAREST), dtype=np.int64).copy()
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask)


def focal_loss(logits, targets, gamma=2.0):
    """Source-era focal-loss formulation, retained as an explicit option."""
    ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    return ((1 - torch.exp(-ce)) ** gamma * ce).mean()


def train(manifest, output, *, epochs=1, batch_size=1, architecture=None, pretrained_backbone=None, loss_name="focal"):
    if not 1 <= epochs <= 1000 or not 1 <= batch_size <= 16:
        raise ValueError("Use 1–1000 epochs and batch size 1–16")
    if loss_name not in {"focal", "cross-entropy"}:
        raise ValueError("Choose focal or cross-entropy loss")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    samples = read_manifest(manifest, require_training=True)
    model = SwinV2Segmentation(architecture, pretrained_backbone=pretrained_backbone)
    size = model.architecture["image_size"]
    loaders = {split: DataLoader(SegmentationDataset([s for s in samples if s.split == split], size),
                                batch_size=batch_size, shuffle=split == "train", num_workers=0)
               for split in ["train", "val"]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    history = []
    for epoch in range(epochs):
        metrics = {"epoch": epoch + 1}
        for split, loader in loaders.items():
            model.train(split == "train")
            losses, scores = [], []
            for image, mask in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split == "train"):
                    logits = model(image)
                    loss = focal_loss(logits, mask) if loss_name == "focal" else torch.nn.functional.cross_entropy(logits, mask)
                    if split == "train":
                        loss.backward()
                        optimizer.step()
                losses.append(float(loss.detach()))
                for predicted, truth in zip(logits.detach().argmax(1).numpy(), mask.numpy()):
                    score = overlap(predicted, truth)["foreground_macro_dice"]
                    if score is not None:
                        scores.append(score)
            metrics[split + "_loss"] = float(np.mean(losses))
            metrics[split + "_foreground_macro_dice"] = float(np.mean(scores)) if scores else None
        history.append(metrics)
    record = {"manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "epochs": epochs, "seed": 42, "history": history,
              "loss": loss_name, "focal_gamma": 2.0 if loss_name == "focal" else None,
              "split_method": "Explicit patient/group-disjoint manifest; no hidden random split",
              "initialization": pretrained_backbone or "random; requires adequate research training",
              "performance_scope": "This run only. No clinical or source-era accuracy claim."}
    checkpoint = save_checkpoint(model.eval(), output, training_record=record)
    (Path(output) / "training.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return checkpoint

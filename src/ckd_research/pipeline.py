"""One analysis path for CLI, uploads, demo and model predictions."""
import base64
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .analysis import measure, overlap
from .contracts import Spacing, validate_mask
from .dataset import preprocess_image
from .demo import fixture
from .io import load_image, render_overlay, save_png


def png_data(array):
    stream = BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def analyze(image, mask, *, spacing=None, reference=None, source="supplied_label_mask", depth_mm=None):
    mask = validate_mask(mask)
    if image.shape != mask.shape:
        raise ValueError("Image and label mask sizes must match")
    report = {"source": source, "shape": list(mask.shape), "labels_are_prediction": source == "checkpoint_model_output",
              "measurements": measure(mask, spacing, depth_mm),
              "evaluation": overlap(mask, reference) if reference is not None else None,
              "scope": "Research software; no diagnosis, disease staging or clinical validation"}
    return {"report": report, "image": png_data(image), "mask": png_data(mask),
            "overlay": png_data(render_overlay(image, mask))}


def demo_result(*, calibrated=False):
    image, mask = fixture()
    result = analyze(image, mask, spacing=Spacing(.4, .4) if calibrated else None,
                     source="generated_engineering_fixture_not_a_patient_scan_or_model_prediction")
    result["report"]["fixture_calibration"] = "Illustrative 0.4 mm/pixel, generated fixture only" if calibrated else None
    return result


def predict(model_manifest, image_path, *, spacing=None):
    import torch
    from .model import load_checkpoint
    torch.set_num_threads(2)
    model, metadata = load_checkpoint(model_manifest)
    image = load_image(image_path)
    array = preprocess_image(image, model.architecture["image_size"])
    with torch.inference_mode():
        predicted = model(torch.from_numpy(array).unsqueeze(0).unsqueeze(0)).argmax(1)[0].numpy().astype(np.uint8)
    # Measure on original resolution so supplied spacing does not silently drift.
    mask = np.array(Image.fromarray(predicted).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST))
    result = analyze(image, mask, spacing=spacing, source="checkpoint_model_output")
    result["report"]["model_manifest"] = str(Path(model_manifest).name)
    result["report"]["model_weights_sha256"] = metadata["sha256"]
    result["report"]["training_record"] = metadata["training_record"]
    return result


def write_result(result, output, stem="analysis", *, protected=()):
    output = Path(output)
    targets = {output / f"{stem}.{key}.png" for key in ["image", "mask", "overlay"]} | {output / f"{stem}.json"}
    if {p.resolve() for p in targets} & {Path(p).resolve() for p in protected if p is not None}:
        raise ValueError("Output would overwrite an input file; choose a separate result directory")
    output.mkdir(parents=True, exist_ok=True)
    for key in ["image", "mask", "overlay"]:
        data = base64.b64decode(result[key].split(",", 1)[1], validate=True)
        (output / f"{stem}.{key}.png").write_bytes(data)
    (output / f"{stem}.json").write_text(json.dumps(result["report"], indent=2, allow_nan=False) + "\n")

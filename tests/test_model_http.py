import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image
import pytest

from ckd_research.pipeline import png_data
from ckd_research.server import Handler


@pytest.fixture
def model_server(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.models_directory = Path(os.environ.get("CKD_MODEL_DIR", tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=90)


def test_model_inventory_and_invalid_model_request(model_server):
    with urllib.request.urlopen(model_server + "/api/models") as response:
        models = json.load(response)["models"]
    assert len(models) == 3 and all(s["profiles"] for s in models)
    image = png_data(np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(model_server + "/api/run-model", {"model_id": "not-a-model", "profile_id": "research-rgb-224", "image": image})
    assert exc.value.code == 400
    assert "Unknown supplied model" in exc.value.read().decode()


def test_manual_source_four_schema_survives_HTTP(model_server):
    mask = np.full((32, 32), 3, dtype=np.uint8)
    with post(model_server + "/api/analyze", {
        "image": png_data(mask * 30), "mask": png_data(mask), "label_schema_id": "source-four"}) as response:
        result = json.load(response)
    assert result["report"]["label_schema_id"] == "source-four"
    assert "capsule_only" not in result["report"]["measurements"]["label_pixels"]


@pytest.mark.parametrize("model_id,profile,task", [
    ("source-swin-four", "source-training-256", "segmentation"),
    ("biopsy-efficientnet", "research-rgb-224", "classification"),
    ("sclerosis-convnext", "research-rgb-224", "classification"),
])
def test_real_supplied_checkpoint_HTTP_flow(model_server, model_id, profile, task):
    if not os.environ.get("CKD_MODEL_DIR"):
        pytest.skip("Set CKD_MODEL_DIR to the verified private assets for real checkpoint integration checks")
    pytest.importorskip("torch")
    y, x = np.mgrid[:64, :64]
    image = np.stack([x * 3, y * 3, (x + y)], axis=-1).astype(np.uint8)
    with post(model_server + "/api/run-model", {
        "image": png_data(image), "model_id": model_id, "profile_id": profile}) as response:
        result = json.load(response)
    report = result["report"]
    assert report["task"] == task and report["actual_checkpoint_executed"] is True
    assert report["input_profile"]["id"] == profile
    assert len(report["weights_sha256"]) == 64
    if task == "classification":
        assert len(report["raw_logits"]) == len(report["softmax_index_scores"]) == 2
        assert "mask" not in result and "overlay" not in result
    else:
        assert report["shape"] == [64, 64] and report["label_schema_id"] == "source-four"
        assert len(report["labels"]) == 4 and "mask" in result


def test_real_classifier_CLI_exports_neutral_logits(tmp_path):
    directory = os.environ.get("CKD_MODEL_DIR")
    if not directory:
        pytest.skip("Real private model assets not supplied")
    pytest.importorskip("torch")
    image = tmp_path / "input.png"
    Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8)).save(image)
    output = tmp_path / "result"
    process = subprocess.run([sys.executable, "-m", "ckd_research.cli", "run-model",
        "--id", "sclerosis-convnext", "--profile", "research-rgb-224", "--image", str(image),
        "--models-dir", directory, "--output", str(output)], capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stderr
    report = json.loads((output / "model-output.json").read_text())
    assert report["actual_checkpoint_executed"] is True and len(report["raw_logits"]) == 2
    assert (output / "model-output.image.png").exists()
    assert not (output / "model-output.mask.png").exists()

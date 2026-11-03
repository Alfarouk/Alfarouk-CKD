import base64
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from ckd_research import supplied
from ckd_research.analysis import overlap
from ckd_research.pipeline import analyze, write_result


def image_bytes(mode="RGB", size=64):
    y, x = np.mgrid[:size, :size]
    image = np.stack([(x * 4) % 256, (y * 4) % 256, ((x + y) * 2) % 256], axis=-1).astype(np.uint8)
    stream = BytesIO()
    Image.fromarray(image).convert(mode).save(stream, format="PNG")
    return stream.getvalue()


def test_registry_has_hash_bound_assets_and_no_invented_class_meanings():
    value = supplied.registry()
    assert value["repository"] == "Alfarouk/Alfarouk-CKD"
    assert value["release"] == "v1.1.0"
    for model in value["models"]:
        assert len(model["weights"]["sha256"]) == len(model["original"]["sha256"]) == 64
        assert model["weights"]["name"].endswith(".safetensors")
        assert "id2label" not in model and "class_names" not in model
        assert model["semantic_limits"]


@pytest.mark.parametrize("profile,shape", [
    ("source-clahe-256", (1, 256, 256)), ("source-training-256", (1, 256, 256)),
    ("research-rgb-224", (3, 224, 224)), ("research-rgb-380", (3, 380, 380)),
    ("research-rgb-224-unit", (3, 224, 224)),
])
def test_explicit_input_profiles_record_geometry_channels_and_normalization(profile, shape):
    tensor, image = supplied.prepare_input(image_bytes(), profile)
    assert tensor.shape == shape and tensor.dtype == np.float32
    assert tensor.flags.c_contiguous and np.isfinite(tensor).all()
    assert image.shape == (64, 64, 3)
    if profile == "research-rgb-224-unit":
        assert tensor.min() >= 0 and tensor.max() <= 1


def test_original_classifier_preprocessing_is_not_assumed():
    for key in ["research-rgb-224", "research-rgb-380", "research-rgb-224-unit"]:
        assert "Original classifier" in supplied.PROFILES[key]["provenance"]
    unit, _ = supplied.prepare_input(image_bytes(), "research-rgb-224-unit")
    normalized, _ = supplied.prepare_input(image_bytes(), "research-rgb-224")
    assert not np.array_equal(unit, normalized)
    with pytest.raises(ValueError, match="explicit input profile"):
        supplied.prepare_input(image_bytes(), "unknown")


def test_four_class_results_do_not_inherit_five_label_meanings():
    image = np.zeros((16, 16), dtype=np.uint8)
    mask = np.zeros_like(image)
    mask[3:12, 4:11] = 3
    result = analyze(image, mask, source="supplied_four_class_model_output", label_schema_id="source-four")
    assert result["report"]["label_schema_id"] == "source-four"
    assert result["report"]["labels_are_prediction"] is True
    assert set(result["report"]["measurements"]["label_pixels"]) == {
        "background", "source_class_1", "source_class_2", "source_class_3"}
    assert "capsule_only" not in json.dumps(result["report"])
    mask[0, 0] = 4
    with pytest.raises(ValueError, match="schema"):
        analyze(image, mask, label_schema_id="source-four")


def test_complete_foreground_miss_cannot_be_hidden_by_absent_classes():
    target = np.zeros((8, 8), dtype=np.uint8)
    target[2:6, 2:6] = 3
    prediction = np.zeros_like(target)
    report = overlap(prediction, target)
    assert report["foreground_macro_dice"] == 0.0
    assert report["per_class"]["region_1"]["dice"] is None
    assert report["per_class"]["region_2"]["dice"] is None


def test_classifier_export_does_not_require_a_fake_segmentation_mask(tmp_path):
    data = base64.b64encode(image_bytes()).decode()
    write_result({"image": "data:image/png;base64," + data, "report": {"task": "classification"}}, tmp_path)
    assert (tmp_path / "analysis.image.png").is_file()
    assert not (tmp_path / "analysis.mask.png").exists()


def test_classifier_cannot_leave_stale_segmentation_artifacts(tmp_path):
    array = np.zeros((16, 16), dtype=np.uint8)
    previous = analyze(array, array)
    write_result(previous, tmp_path, "model-output")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    image = "data:image/png;base64," + base64.b64encode(image_bytes()).decode()
    with pytest.raises(ValueError, match="different result type"):
        write_result({"image": image, "report": {"task": "classification"}}, tmp_path, "model-output")
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_missing_or_corrupt_weights_stop_before_model_build(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(supplied, "build_model", lambda _: pytest.fail("Must not construct a model"))
    with pytest.raises(FileNotFoundError, match="not installed"):
        supplied.run_model("source-swin-four", image_bytes(), tmp_path, profile_id="source-clahe-256")
    (tmp_path / "source-swin-four.safetensors").write_bytes(b"not weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        supplied.run_model("source-swin-four", image_bytes(), tmp_path, profile_id="source-clahe-256")


def test_wrong_task_profile_rejected_before_loading(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="does not belong"):
        supplied.run_model("source-swin-four", image_bytes(), tmp_path, profile_id="research-rgb-224")


def test_download_is_private_hash_checked_and_does_not_replace_existing_data(tmp_path, monkeypatch):
    content = b"test dense tensor asset"
    import hashlib
    sha = hashlib.sha256(content).hexdigest()
    catalog = {"repository": "Alfarouk/Alfarouk-CKD", "release": "v1.1.0", "models": [
        {"weights": {"name": "test.safetensors", "sha256": sha}}]}
    monkeypatch.setattr(supplied, "registry", lambda: catalog)
    calls = []
    def download(args, **kwargs):
        calls.append(args)
        assert args[:4] == ["gh", "release", "download", "v1.1.0"]
        assert args[args.index("--repo") + 1] == "Alfarouk/Alfarouk-CKD"
        (Path(args[args.index("--dir") + 1]) / "test.safetensors").write_bytes(content)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(supplied.subprocess, "run", download)
    assert len(supplied.fetch_models(tmp_path)) == 1
    assert len(supplied.fetch_models(tmp_path)) == 1 and len(calls) == 1
    (tmp_path / "test.safetensors").write_bytes(b"user file")
    with pytest.raises(ValueError, match="instead of overwriting"):
        supplied.fetch_models(tmp_path)
    assert (tmp_path / "test.safetensors").read_bytes() == b"user file"


def test_failed_download_does_not_publish_partial_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(supplied.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match="gh sign-in"):
        supplied.fetch_models(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_preserves_a_destination_created_during_network_wait(tmp_path, monkeypatch):
    import hashlib
    content = b"verified staged asset"
    monkeypatch.setattr(supplied, "registry", lambda: {
        "repository": "Alfarouk/Alfarouk-CKD", "release": "v1.1.0", "models": [
            {"weights": {"name": "test.safetensors", "sha256": hashlib.sha256(content).hexdigest()}}]})
    def download(args, **_kwargs):
        (Path(args[args.index("--dir") + 1]) / "test.safetensors").write_bytes(content)
        (tmp_path / "test.safetensors").write_bytes(b"new user work")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(supplied.subprocess, "run", download)
    with pytest.raises(ValueError, match="appeared"):
        supplied.fetch_models(tmp_path)
    assert (tmp_path / "test.safetensors").read_bytes() == b"new user work"
    assert len(list(tmp_path.iterdir())) == 1


def test_conversion_preserves_a_destination_created_during_serialization(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import safetensors.torch
    target = tmp_path / "converted.safetensors"
    spec = {"id": "source-swin-four", "weights": {"name": target.name, "sha256": None}}
    monkeypatch.setattr(supplied, "spec_for", lambda _: spec)
    monkeypatch.setattr(supplied, "_restricted_original", lambda *_: {"weight": torch.ones(1)})
    monkeypatch.setattr(supplied, "build_model", lambda _: SimpleNamespace(load_state_dict=lambda *_args, **_kwargs: None))
    save = safetensors.torch.save_file
    def serialize(state, path):
        save(state, path)
        target.write_bytes(b"new user work")
    monkeypatch.setattr(safetensors.torch, "save_file", serialize)
    with pytest.raises(ValueError, match="appeared"):
        supplied.convert_original("source-swin-four", tmp_path / "original.pth", tmp_path)
    assert target.read_bytes() == b"new user work"
    assert len(list(tmp_path.iterdir())) == 1


def test_tensor_validation_rejects_nonfinite_and_nontensor_payloads():
    torch = pytest.importorskip("torch")
    for state in [{"x": torch.tensor([float("nan")])}, {"x": "not a tensor"}, {}]:
        with pytest.raises(ValueError):
            supplied._validate_tensors(state)


def test_one_model_lock_blocks_concurrent_model_loads():
    pytest.importorskip("torch")
    with supplied.one_model():
        with pytest.raises(ValueError, match="Another model operation"):
            with supplied.one_model():
                pytest.fail("Concurrent model load accepted")


def test_restricted_import_rejects_unexpected_source_hash(tmp_path):
    pytest.importorskip("torch")
    source = tmp_path / "wrong.pth"
    source.write_bytes(b"not the supplied checkpoint")
    with pytest.raises(ValueError, match="hash differs"):
        supplied.convert_original("source-swin-four", source, tmp_path / "converted")


@pytest.mark.parametrize("model_id,shape", [
    ("source-swin-four", (1, 4, 256, 256)),
    ("biopsy-efficientnet", (1, 2)), ("sclerosis-convnext", (1, 2)),
])
def test_offline_architecture_forward_contract(model_id, shape, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    torch.set_num_threads(2)
    model = supplied.build_model(model_id)
    size = 256 if model_id == "source-swin-four" else 224
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, size, size)))
        logits = getattr(output, "logits", output)
    assert tuple(logits.shape) == shape and torch.isfinite(logits).all().item()

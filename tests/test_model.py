import hashlib
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from ckd_research.model import SwinV2Segmentation, save_checkpoint, load_checkpoint, validate_architecture
from ckd_research.training import train, focal_loss
from ckd_research.pipeline import predict
from test_workbench import make_dataset

TINY = {"image_size": 64, "embed_dim": 12, "depths": [1, 1, 1, 1],
        "num_heads": [3, 3, 6, 12], "window_size": 2,
        "decoder_channels": [32, 16, 12, 8, 4], "num_classes": 5}


def test_offline_swin_forward_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    torch.set_num_threads(2)
    model = SwinV2Segmentation(TINY).eval()
    x = torch.zeros((1, 1, 64, 64))
    with torch.inference_mode():
        expected = model(x)
    assert expected.shape == (1, 5, 64, 64)
    path = save_checkpoint(model, tmp_path, training_record={"kind": "untrained_engineering_fixture"})
    recovered, metadata = load_checkpoint(path)
    with torch.inference_mode():
        assert torch.equal(expected, recovered(x))
    assert metadata["clinical_validation"] is False


def test_hash_and_label_mismatch_are_rejected(tmp_path):
    torch.set_num_threads(2)
    path = save_checkpoint(SwinV2Segmentation(TINY).eval(), tmp_path, training_record={})
    metadata = json.loads(path.read_text())
    metadata["sha256"] = "0" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path)
    metadata["labels"].pop("4")
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="labels"):
        load_checkpoint(path)


def test_real_training_save_and_prediction_pipeline(tmp_path):
    manifest, _ = make_dataset(tmp_path)
    checkpoint = train(manifest, tmp_path / "checkpoint", epochs=1, architecture=TINY)
    result = predict(checkpoint, tmp_path / "image2.png")
    assert result["report"]["labels_are_prediction"] is True
    assert result["report"]["shape"] == [64, 64]
    record = result["report"]["training_record"]
    assert record["epochs"] == 1
    assert record["history"][0]["train_loss"] > 0
    assert "This run only" in record["performance_scope"]


@pytest.mark.parametrize("field,value", [("image_size", 100000), ("embed_dim", 100000), ("num_classes", 4), ("decoder_channels", [10000] * 5)])
def test_unbounded_or_legacy_model_configuration_rejected(field, value):
    config = dict(TINY, **{field: value})
    with pytest.raises(ValueError):
        validate_architecture(config)


def test_default_source_architecture_forward():
    torch.set_num_threads(2)
    model = SwinV2Segmentation().eval()
    with torch.inference_mode():
        logits = model(torch.zeros((1, 1, 256, 256)))
    assert logits.shape == (1, 5, 256, 256)


def test_focal_gamma_zero_matches_cross_entropy():
    logits = torch.tensor([[[[1., 2.]], [[3., 4.]]]], requires_grad=True)
    target = torch.tensor([[[0, 1]]])
    loss = focal_loss(logits, target, gamma=0)
    assert torch.equal(loss, torch.nn.functional.cross_entropy(logits, target))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_malformed_manifest_is_an_actionable_error(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="manifest"):
        load_checkpoint(path)


def test_malformed_architecture_sequences_are_rejected():
    with pytest.raises(ValueError):
        validate_architecture(dict(TINY, depths=None))

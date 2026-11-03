import base64
import csv
from io import BytesIO
import json
from pathlib import Path
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
import numpy as np
from PIL import Image
import pytest

from ckd_research.analysis import measure, overlap
from ckd_research.cli import main
from ckd_research.contracts import Spacing, combine_source_masks, validate_mask
from ckd_research.dataset import read_manifest, import_triplets
from ckd_research.demo import fixture
from ckd_research.io import load_image, load_mask, save_png, render_overlay
from ckd_research.pipeline import demo_result, png_data, write_result
from ckd_research.server import Handler


def make_dataset(path):
    rows = []
    for n, split in enumerate(["train", "val", "test"]):
        image, mask = fixture(64, seed=n)
        save_png(image, path / f"image{n}.png")
        save_png(mask, path / f"mask{n}.png")
        rows.append([split, f"patient{n}", f"image{n}.png", f"mask{n}.png"])
    manifest = path / "manifest.csv"
    write_manifest(manifest, rows)
    return manifest, rows


def write_manifest(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "patient_id", "image", "mask"])
        writer.writerows(rows)


def test_source_mask_import_preserves_regions():
    capsule = np.ones((3, 3), dtype=np.uint8)
    regions = np.array([[0, 1, 2], [3, 3, 0], [2, 1, 0]], dtype=np.uint8)
    result = combine_source_masks(capsule, regions)
    assert np.array_equal(result[regions > 0], regions[regions > 0])
    assert (result[regions == 0] == 4).all()


def test_region_outside_capsule_rejected():
    with pytest.raises(ValueError, match="outside"):
        combine_source_masks(np.zeros((2, 2), np.uint8), np.ones((2, 2), np.uint8))


@pytest.mark.parametrize("mask", [np.array([[255]], np.uint8), np.ones((2, 2), float), np.zeros((0, 0), np.uint8), np.zeros((2, 2, 3), np.uint8)])
def test_invalid_labels_fail(mask):
    with pytest.raises(ValueError):
        validate_mask(mask)


@pytest.mark.parametrize("a,b", [(0, 1), (-1, 2), (float("nan"), 1), (1, float("inf"))])
def test_invalid_scale_fails(a, b):
    with pytest.raises(ValueError):
        Spacing(a, b)


def test_scale_follows_resize():
    resized = Spacing(.2, .4).resized((100, 200), (50, 50))
    assert resized == Spacing(.4, 1.6)


def test_geometry_uses_separate_row_column_scale():
    mask = np.zeros((40, 60), np.uint8)
    mask[5:25, 10:40] = 1
    raw = measure(mask)
    scaled = measure(mask, Spacing(2, 3), depth_mm=10)
    assert raw["area"] == 600
    assert raw["units"] == "pixels" and raw["volume_mm3"] is None
    assert scaled["area"] == 3600
    assert scaled["major_axis"] == pytest.approx(87)
    assert scaled["minor_axis"] == pytest.approx(38)
    assert scaled["diagnosis"] is None


def test_volume_without_scale_fails():
    with pytest.raises(ValueError, match="calibrated"):
        measure(np.ones((20, 20), np.uint8), depth_mm=5)


def test_empty_mask_is_explicit():
    result = measure(np.zeros((10, 10), np.uint8))
    assert result["status"] == "empty_mask"
    assert result["major_axis"] is None


def test_overlap_absent_classes_are_not_perfect_scores():
    _, mask = fixture(64)
    result = overlap(mask, mask)
    assert result["foreground_macro_dice"] == 1
    empty = overlap(np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8))
    assert empty["foreground_macro_dice"] is None
    assert empty["per_class"]["region_1"]["dice"] is None


def test_metric_does_not_silently_resize():
    with pytest.raises(ValueError, match="shapes"):
        overlap(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))


def test_manifest_preserves_explicit_splits(tmp_path):
    manifest, _ = make_dataset(tmp_path)
    assert [s.split for s in read_manifest(manifest, require_training=True)] == ["train", "val", "test"]


def test_patient_leakage_rejected(tmp_path):
    manifest, rows = make_dataset(tmp_path)
    rows[1][1] = rows[0][1]
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="Patient"):
        read_manifest(manifest)


def test_duplicate_image_leakage_rejected(tmp_path):
    manifest, rows = make_dataset(tmp_path)
    (tmp_path / rows[1][2]).write_bytes((tmp_path / rows[0][2]).read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest)


def test_escaping_data_paths_rejected(tmp_path):
    manifest, rows = make_dataset(tmp_path)
    rows[0][2] = "../outside.png"
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="inside"):
        read_manifest(manifest)


def test_empty_manifest_rejected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [])
    with pytest.raises(ValueError, match="empty"):
        read_manifest(manifest)


def test_color_mask_not_accepted_as_class_labels(tmp_path):
    save_png(np.zeros((10, 10, 3), np.uint8), tmp_path / "color.png")
    with pytest.raises(ValueError, match="integer"):
        load_mask(tmp_path / "color.png")


def test_overlay_rejects_different_shape():
    with pytest.raises(ValueError, match="shapes"):
        render_overlay(np.zeros((3, 3), np.uint8), np.zeros((4, 4), np.uint8))


def test_demo_exports_truthful_result(tmp_path):
    result = demo_result()
    write_result(result, tmp_path)
    assert len(list(tmp_path.iterdir())) == 4
    report = json.loads((tmp_path / "analysis.json").read_text())
    assert report["labels_are_prediction"] is False
    assert "generated_engineering_fixture" in report["source"]
    assert report["measurements"]["units"] == "pixels"


def test_output_cannot_overwrite_an_input(tmp_path):
    source = tmp_path / "analysis.image.png"
    source.write_bytes(b"original")
    with pytest.raises(ValueError, match="overwrite"):
        write_result(demo_result(), tmp_path, protected=[source])
    assert source.read_bytes() == b"original"


def test_exif_rotation_requires_pair_normalization(tmp_path):
    image = Image.new("L", (10, 10))
    exif = Image.Exif()
    exif[274] = 6
    image.save(tmp_path / "rotated.jpg", exif=exif)
    with pytest.raises(ValueError, match="orientation"):
        load_image(tmp_path / "rotated.jpg")


def test_triplet_import_keeps_class_ids(tmp_path):
    for name in ["images", "capsules", "regions"]:
        (tmp_path / name).mkdir()
    image, mask = fixture(64)
    save_png(image, tmp_path / "images/sample.jpg")
    save_png((mask > 0).astype(np.uint8), tmp_path / "capsules/sample.png")
    regions = mask.copy()
    regions[regions == 4] = 0
    save_png(regions, tmp_path / "regions/sample.png")
    import_triplets(tmp_path / "images", tmp_path / "capsules", tmp_path / "regions", tmp_path / "imported")
    assert np.array_equal(load_mask(tmp_path / "imported/masks/sample.png"), mask)


def test_cli_demo_analyze_batch_and_missing_checkpoint(tmp_path, capsys):
    assert main(["demo", "--output", str(tmp_path / "demo")]) == 0
    assert main(["analyze", "--image", str(tmp_path / "demo/engineering-fixture.image.png"),
                 "--mask", str(tmp_path / "demo/engineering-fixture.mask.png"), "--output", str(tmp_path / "analysis")]) == 0
    manifest, _ = make_dataset(tmp_path)
    assert main(["batch", "--manifest", str(manifest), "--output", str(tmp_path / "batch")]) == 0
    assert (tmp_path / "batch/summary.csv").is_file()
    assert main(["predict", "--model", str(tmp_path / "missing.json"), "--image", "missing.png", "--output", str(tmp_path / "predict")]) == 2
    assert "missing" in capsys.readouterr().err.lower()


@pytest.fixture
def server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def post(url, body, **headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    return urllib.request.urlopen(req, timeout=10)


def test_http_health_assets_demo_and_upload(server):
    with urllib.request.urlopen(server + "/api/health") as response:
        assert json.load(response)["research_only"]
    for asset in ["/", "/style.css", "/app.js"]:
        with urllib.request.urlopen(server + asset) as response:
            assert response.status == 200
    with post(server + "/api/demo", {"calibrated": True}) as response:
        result = json.load(response)
    assert result["report"]["measurements"]["units"] == "mm"
    image, mask = fixture(64)
    with post(server + "/api/analyze", {"image": png_data(image), "mask": png_data(mask)}) as response:
        assert json.load(response)["report"]["shape"] == [64, 64]


def test_http_rejects_bad_upload_and_cross_origin(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        post(server + "/api/analyze", {"image": "bad", "mask": "bad"})
    assert error.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as error:
        post(server + "/api/demo", {}, Origin="https://example.org")
    assert error.value.code == 403

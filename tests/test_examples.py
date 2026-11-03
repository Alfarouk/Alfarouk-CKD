from importlib.resources import files
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from PIL import Image
import pytest

from ckd_research.server import Handler


@pytest.fixture
def example_server(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.models_directory = tmp_path
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.mark.parametrize("name", [
    "ultrasound-output.jpg", "biopsy-sclerosed-output.jpg", "biopsy-normal-output.jpg",
])
def test_project_figure_HTTP_route_serves_intact_packaged_JPEG(example_server, name):
    expected = files("ckd_research").joinpath("static", "examples", name).read_bytes()
    with urllib.request.urlopen(example_server + "/examples/" + name) as response:
        assert response.headers["Content-Type"] == "image/jpeg"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        content = response.read()
    assert content == expected
    with Image.open(BytesIO(content)) as image:
        assert image.format == "JPEG"
        assert image.width >= 1000 and image.height >= 400
        image.verify()


@pytest.mark.parametrize("path", [
    "/examples/", "/examples/unknown.jpg", "/examples/../model_registry.json",
    "/examples/%2e%2e/model_registry.json", "/examples/ultrasound-output.jpg/extra",
])
def test_figure_routes_do_not_expose_other_resources(example_server, path):
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(example_server + path)
    assert error.value.code == 404


def test_examples_boot_and_selection_preserve_live_result_identity():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the isolated frontend state check")
    script = Path(__file__).with_name("frontend_examples.cjs")
    run = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["default_view"] == "examples"
    assert result["figures"] == 3
    assert result["initial_requests"] == ["/api/models"]
    assert result["saved_figures_keep_live_report"] is True
    assert result["model_busy_controls"] is True

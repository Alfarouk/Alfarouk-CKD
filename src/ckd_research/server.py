"""Small local workbench. No uploads, data or inference leave this computer."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from io import BytesIO
import json
from pathlib import Path
import threading
from urllib.parse import urlparse
import numpy as np
from PIL import Image
from .contracts import Spacing, MAX_PIXELS, validate_mask
from .pipeline import analyze, demo_result

MAX_BODY = 4_000_000


def decode_png(data, *, label=False):
    if not isinstance(data, str) or not data.startswith("data:image/"):
        raise ValueError("Supply a PNG or JPEG data URL")
    content = base64.b64decode(data.split(",", 1)[1], validate=True)
    with Image.open(BytesIO(content)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image exceeds16 million pixels")
        if label:
            if image.mode not in {"L", "P", "I", "I;16"}:
                raise ValueError("Label masks must be grayscale/indexed integer images")
            return validate_mask(np.array(image))
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Normalize image and mask orientation together before analysis")
        return np.array(image.convert("L"))


class Handler(BaseHTTPRequestHandler):
    slots = threading.BoundedSemaphore(2)

    def log_message(self, *_):
        pass

    def send(self, status, content, content_type="application/json"):
        if not isinstance(content, bytes):
            content = json.dumps(content, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            from .supplied import model_status
            count = sum(s["installed"] for s in model_status(getattr(self.server, "models_directory", Path("models"))))
            return self.send(200, {"status": "ok", "model_checkpoint_supplied": count > 0,
                                   "installed_supplied_models": count, "research_only": True})
        if path == "/api/models":
            from .supplied import model_status
            return self.send(200, {"models": model_status(getattr(self.server, "models_directory", Path("models")))})
        mapping = {
            "/": (("index.html",), "text/html; charset=utf-8"),
            "/app.js": (("app.js",), "text/javascript"),
            "/style.css": (("style.css",), "text/css"),
            "/examples/ultrasound-output.jpg": (("examples", "ultrasound-output.jpg"), "image/jpeg"),
            "/examples/biopsy-sclerosed-output.jpg": (("examples", "biopsy-sclerosed-output.jpg"), "image/jpeg"),
            "/examples/biopsy-normal-output.jpg": (("examples", "biopsy-normal-output.jpg"), "image/jpeg"),
        }
        if path not in mapping:
            return self.send(404, {"error": "Not found"})
        parts, content_type = mapping[path]
        self.send(200, files("ckd_research").joinpath("static", *parts).read_bytes(), content_type)

    def do_POST(self):
        if self.headers.get("Origin") and self.headers["Origin"] != "http://" + self.headers.get("Host", ""):
            return self.send(403, {"error": "Cross-origin requests are not accepted"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_BODY:
                return self.send(413, {"error": "Request must be between 1 byte and 4 MB"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send(415, {"error": "Use application/json"})
            if not self.slots.acquire(blocking=False):
                return self.send(503, {"error": "Two analyses are already running; try again"})
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object")
                if self.path == "/api/demo":
                    result = demo_result(calibrated=payload.get("calibrated") is True)
                elif self.path == "/api/analyze":
                    image, mask = decode_png(payload["image"]), decode_png(payload["mask"], label=True)
                    spacing = payload.get("spacing")
                    spacing = Spacing(float(spacing[0]), float(spacing[1])) if spacing else None
                    result = analyze(image, mask, spacing=spacing,
                                     label_schema_id=payload.get("label_schema_id", "corrected-five"))
                elif self.path == "/api/run-model":
                    from .supplied import run_model
                    image = payload["image"]
                    if not isinstance(image, str) or not image.startswith("data:image/"):
                        raise ValueError("Supply an image data URL")
                    content = base64.b64decode(image.split(",", 1)[1], validate=True)
                    result = run_model(payload["model_id"], content,
                        getattr(self.server, "models_directory", Path("models")), profile_id=payload["profile_id"])
                else:
                    return self.send(404, {"error": "Not found"})
                return self.send(200, result)
            finally:
                self.slots.release()
        except (ValueError, KeyError, TypeError, IndexError, OSError, ImportError, Image.DecompressionBombError) as exc:
            return self.send(400, {"error": str(exc)})


def serve(host="127.0.0.1", port=8782, models_directory=Path("models")):
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("This research workbench binds to localhost only")
    server = ThreadingHTTPServer((host, port), Handler)
    server.models_directory = Path(models_directory).resolve()
    server.daemon_threads = True
    print(f"CKD research workbench: http://{host}:{server.server_port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

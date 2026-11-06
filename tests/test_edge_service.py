import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from edgeplant.config import ROOT
from plantdoctor_edge.engine import Engine, ImageRejected, decode
from plantdoctor_edge.server import is_loopback, make_handler, validate_binding

MODEL = ROOT / "models" / "plant-doctor-int8.onnx"
LABELS = ROOT / "models" / "labels.json"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="deployed model not present")


def jpeg(size=(320, 240), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (*size[::-1], 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def engine():
    return Engine(MODEL, LABELS)


@pytest.fixture
def server(engine):
    def start(token):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, token))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd

    started = []

    def factory(token=None):
        httpd = start(token)
        started.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    yield factory
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


def post(url, body, headers=None):
    req = urllib.request.Request(url + "/predict", data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_engine_returns_three_ordered_predictions(engine):
    result = engine.classify(jpeg())
    probs = [p["probability"] for p in result["predictions"]]
    assert len(probs) == 3 and probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0001
    assert {"crop", "condition", "healthy"} <= set(result["predictions"][0])


def test_decode_rejects_garbage_and_oversized_images():
    with pytest.raises(ImageRejected):
        decode(b"not an image")
    buf = io.BytesIO()
    Image.new("L", (7000, 7000)).save(buf, "PNG")
    with pytest.raises(ImageRejected):
        decode(buf.getvalue())


def test_health_and_predict(server):
    url = server()
    with urllib.request.urlopen(url + "/health", timeout=10) as res:
        assert json.loads(res.read())["status"] == "ok"
    status, body = post(url, jpeg())
    assert status == 200 and len(body["predictions"]) == 3


def test_bad_inputs_are_refused_without_stack_traces(server):
    url = server()
    assert post(url, b"")[0] == 400
    status, body = post(url, b"not an image")
    assert status == 400 and "Traceback" not in json.dumps(body)


def test_oversized_body_is_refused(server):
    url = server()
    status, _ = post(url, b"x" * (10 * 1024 * 1024 + 1))
    assert status == 413


def test_token_is_enforced_when_configured(server):
    url = server("s3cret")
    assert post(url, jpeg())[0] == 401
    assert post(url, jpeg(), {"Authorization": "Bearer wrong"})[0] == 401
    assert post(url, jpeg(), {"Authorization": "Bearer s3cret"})[0] == 200


def test_public_bind_requires_a_token():
    assert is_loopback("127.0.0.1") and is_loopback("::1") and not is_loopback("0.0.0.0")
    validate_binding("127.0.0.1", None)
    validate_binding("0.0.0.0", "token")
    with pytest.raises(SystemExit):
        validate_binding("0.0.0.0", None)

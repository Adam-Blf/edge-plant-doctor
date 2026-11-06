"""The browser pipeline (app/src/preprocess.ts) must match the Python reference."""

import shutil
import struct
import subprocess

import numpy as np
import pytest

from edgeplant.config import ROOT
from edgeplant.preprocess import preprocess

SCRIPT = ROOT / "app" / "scripts" / "parity.ts"
pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not SCRIPT.exists(), reason="node or the web app is missing"
)


def web_preprocess(rgb: np.ndarray, tmp_path) -> np.ndarray:
    h, w = rgb.shape[:2]
    rgba = np.dstack([rgb, np.full((h, w), 255, dtype=np.uint8)])
    src, dst = tmp_path / "in.bin", tmp_path / "out.bin"
    src.write_bytes(struct.pack("<II", w, h) + rgba.tobytes())
    subprocess.run(["node", str(SCRIPT), str(src), str(dst)], check=True, cwd=ROOT / "app")
    return np.frombuffer(dst.read_bytes(), dtype=np.float32).reshape(1, 3, 224, 224)


def gradient(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.dstack([x * 255 // w, y * 255 // h, (x + y) * 255 // (w + h)]).astype(np.uint8)


@pytest.mark.parametrize("shape", [(256, 256), (300, 480), (900, 640), (896, 896), (150, 150)])
def test_random_images_match(shape, tmp_path):
    rng = np.random.default_rng(shape[0])
    rgb = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    np.testing.assert_allclose(web_preprocess(rgb, tmp_path), preprocess(rgb), atol=1e-4)


def test_smooth_image_matches(tmp_path):
    rgb = gradient(480, 640)
    np.testing.assert_allclose(web_preprocess(rgb, tmp_path), preprocess(rgb), atol=1e-4)

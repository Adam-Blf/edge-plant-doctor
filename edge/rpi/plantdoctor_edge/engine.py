"""Inference engine for the Raspberry Pi: ONNX Runtime plus the shared preprocessing."""

import io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

from edgeplant.preprocess import preprocess

MAX_PIXELS = 40_000_000


class ImageRejectedError(ValueError):
    """The bytes are not a usable image (corrupt, unsupported or too large)."""


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def decode(data: bytes) -> np.ndarray:
    """Image bytes to an RGB uint8 array with EXIF orientation applied."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > MAX_PIXELS:
                raise ImageRejectedError("image too large")
            return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageRejectedError("not a readable image") from exc


class Engine:
    def __init__(self, model: Path, labels: Path, threads: int = 0):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.labels = json.loads(labels.read_text(encoding="utf-8"))
        self.model_name = model.name

    def classify(self, data: bytes, k: int = 3) -> dict:
        started = time.perf_counter()
        tensor = preprocess(decode(data))
        prepared = time.perf_counter()
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        done = time.perf_counter()
        probs = softmax(logits)
        top = np.argsort(probs)[::-1][:k]
        return {
            "predictions": [
                {**self.labels[i], "probability": round(float(probs[i]), 4)} for i in top
            ],
            "preprocess_ms": round((prepared - started) * 1000, 1),
            "inference_ms": round((done - prepared) * 1000, 1),
            "model": self.model_name,
        }

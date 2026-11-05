"""Deterministic preprocessing shared by training, benchmarks and the web app.

The browser cannot reproduce PIL resampling, so the pipeline is defined once as a
plain algorithm: centre crop to a square, area-weighted resize to 224x224 (each
output pixel is the exact average of the source region it covers), then
normalisation. `app/src/preprocess.ts` implements the same steps.
"""

import numpy as np

from .config import INPUT_SIZE, MEAN, STD


def area_weights(src: int, dst: int) -> np.ndarray:
    """Matrix (dst, src) whose rows average the source span covered by each output pixel."""
    scale = src / dst
    w = np.zeros((dst, src), dtype=np.float64)
    for i in range(dst):
        lo, hi = i * scale, (i + 1) * scale
        first, last = int(np.floor(lo)), min(int(np.ceil(hi)), src)
        for j in range(first, last):
            w[i, j] = min(hi, j + 1) - max(lo, j)
        w[i] /= w[i].sum()
    return w


def center_crop_square(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return rgb[top : top + side, left : left + side]


def resize_area(rgb: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Square RGB uint8 (s, s, 3) to float32 (size, size, 3) in the 0..255 range."""
    side = rgb.shape[0]
    wy = area_weights(side, size)
    rows = (wy @ rgb.astype(np.float64).reshape(side, side * 3)).reshape(size, side, 3)
    cols = wy @ rows.transpose(1, 0, 2).reshape(side, size * 3)
    out = cols.reshape(size, size, 3).transpose(1, 0, 2)
    return out.astype(np.float32)


def to_uint8_square(rgb: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Cache format used by training: crop and resize, rounded back to uint8."""
    out = resize_area(center_crop_square(rgb), size)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def normalize(chw_source: np.ndarray) -> np.ndarray:
    """(size, size, 3) 0..255 to NCHW float32 normalised tensor with batch of one."""
    x = chw_source.astype(np.float32) / 255.0
    x = (x - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Full inference pipeline: RGB uint8 (h, w, 3) of any size to (1, 3, 224, 224) float32."""
    return normalize(resize_area(center_crop_square(rgb)))

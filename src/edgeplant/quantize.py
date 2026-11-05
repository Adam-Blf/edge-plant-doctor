"""Post-training quantization with ONNX Runtime: dynamic and static int8 (QDQ).

Variants, named after what stays in float:

- `dynamic_gemm`: only the classifier layers are quantized.
- `dynamic_all`, `static_all_minmax`: every Conv and Gemm, the naive recipe. On
  MobileNetV3-Small it collapses to near-chance accuracy, and it is kept as a measured
  negative result.
- `dynamic_mixed`, `static_mixed_*`: the stem convolution and the depthwise convolutions
  stay in float, the 40 pointwise convolutions and the classifier go to int8. Measured
  one family at a time, quantizing either the stem or the depthwise layers is what breaks
  the network.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from .config import CHECKPOINT_DIR, SEED
from .data import load_split
from .preprocess import normalize

CALIBRATION_IMAGES = 200
LINEAR_OPS = ["Conv", "Gemm", "MatMul"]
STATIC_MIXED = {
    "static_mixed_minmax": CalibrationMethod.MinMax,
    "static_mixed_percentile": CalibrationMethod.Percentile,
    "static_mixed_entropy": CalibrationMethod.Entropy,
}


def calibration_images(count: int = CALIBRATION_IMAGES) -> np.ndarray:
    """A fixed random sample of training images (uint8, NHWC)."""
    x, _ = load_split("train")
    idx = np.sort(np.random.default_rng(SEED).choice(len(x), count, replace=False))
    return np.stack([x[i] for i in idx])


class Calibration(CalibrationDataReader):
    """Feeds calibration images to ONNX Runtime, one at a time."""

    def __init__(self, input_name: str, images: np.ndarray):
        self._batches = iter([{input_name: normalize(img)} for img in images])

    def get_next(self):
        return next(self._batches, None)


def float_islands(model_path: Path) -> list[str]:
    """Names of the stem convolution and of every depthwise convolution."""
    convs = [n for n in onnx.load(str(model_path)).graph.node if n.op_type == "Conv"]
    depthwise = [n.name for n in convs if any(a.name == "group" and a.i > 1 for a in n.attribute)]
    return [convs[0].name, *depthwise]


def quantize_all(
    fp32: Path,
    out_dir: Path,
    images: np.ndarray,
    static_methods: dict[str, CalibrationMethod] | None = None,
) -> dict[str, Path]:
    """Create every int8 variant of `fp32` and return name to path."""
    prepared = out_dir / f"{fp32.stem}.prep.onnx"
    quant_pre_process(str(fp32), str(prepared))
    input_name = ort.InferenceSession(str(fp32)).get_inputs()[0].name
    keep_float = float_islands(prepared)
    outputs: dict[str, Path] = {}

    def path_for(name: str) -> Path:
        outputs[name] = out_dir / f"{fp32.stem}.{name}.onnx"
        return outputs[name]

    dynamic = {
        "dynamic_gemm": {"op_types_to_quantize": ["MatMul", "Gemm"]},
        "dynamic_all": {"per_channel": True},
        "dynamic_mixed": {"per_channel": True, "nodes_to_exclude": keep_float},
    }
    for name, options in dynamic.items():
        quantize_dynamic(str(prepared), str(path_for(name)), weight_type=QuantType.QInt8, **options)

    def static(name: str, method: CalibrationMethod, exclude: list[str]) -> None:
        quantize_static(
            str(prepared),
            str(path_for(name)),
            Calibration(input_name, images),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=LINEAR_OPS,
            nodes_to_exclude=exclude,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=method,
        )

    static("static_all_minmax", CalibrationMethod.MinMax, [])
    for name, method in (static_methods or STATIC_MIXED).items():
        static(name, method, keep_float)
    prepared.unlink()
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize an exported FP32 ONNX model")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    fp32 = CHECKPOINT_DIR / f"{args.tag}.onnx"
    for name, path in quantize_all(fp32, CHECKPOINT_DIR, calibration_images()).items():
        print(name, path.stat().st_size)


if __name__ == "__main__":
    main()

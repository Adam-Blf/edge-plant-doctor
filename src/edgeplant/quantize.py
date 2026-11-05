"""Post-training quantization with ONNX Runtime: dynamic and static int8 (QDQ)."""

import argparse
from pathlib import Path

import numpy as np
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
STATIC_VARIANTS = {
    "static_minmax": CalibrationMethod.MinMax,
    "static_percentile": CalibrationMethod.Percentile,
    "static_entropy": CalibrationMethod.Entropy,
}


class Calibration(CalibrationDataReader):
    """Feeds calibration images from the training split, one at a time."""

    def __init__(self, input_name: str, count: int = CALIBRATION_IMAGES):
        x, _ = load_split("train")
        idx = np.sort(np.random.default_rng(SEED).choice(len(x), count, replace=False))
        self._batches = iter([{input_name: normalize(x[i])} for i in idx])

    def get_next(self):
        return next(self._batches, None)


def quantize_all(fp32: Path, out_dir: Path) -> dict[str, Path]:
    """Create every int8 variant of `fp32` and return name to path."""
    prepared = out_dir / f"{fp32.stem}.prep.onnx"
    quant_pre_process(str(fp32), str(prepared))
    input_name = ort.InferenceSession(str(fp32)).get_inputs()[0].name
    outputs = {}
    dyn = out_dir / f"{fp32.stem}.dynamic.onnx"
    quantize_dynamic(str(prepared), str(dyn), weight_type=QuantType.QInt8)
    outputs["dynamic"] = dyn
    for name, method in STATIC_VARIANTS.items():
        path = out_dir / f"{fp32.stem}.{name}.onnx"
        quantize_static(
            str(prepared),
            str(path),
            Calibration(input_name),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=method,
        )
        outputs[name] = path
    prepared.unlink()
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize an exported FP32 ONNX model")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    fp32 = CHECKPOINT_DIR / f"{args.tag}.onnx"
    for name, path in quantize_all(fp32, CHECKPOINT_DIR).items():
        print(name, path.stat().st_size)


if __name__ == "__main__":
    main()

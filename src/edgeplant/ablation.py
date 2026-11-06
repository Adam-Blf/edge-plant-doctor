"""Which layer family breaks int8? Quantize one family at a time and measure.

Families of the student's linear layers: the stem convolution, the depthwise
convolutions, the pointwise convolutions and the classifier (Gemm/MatMul). Each run
quantizes exactly one family (static QDQ, per-channel, percentile calibration, the
recipe of the shipped model) and keeps everything else in float. The validation split
is used, never the test split, because this study chose the float islands.

    python -m edgeplant.ablation --tag student_distilled
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .config import CHECKPOINT_DIR, RESULTS_DIR
from .data import load_split
from .evaluate import onnx_logits
from .quantize import LINEAR_OPS, Calibration, calibration_images


def families(model_path: Path) -> dict[str, list[str]]:
    """Partition the linear nodes of the graph into stem, depthwise, pointwise, classifier."""
    nodes = [n for n in onnx.load(str(model_path)).graph.node if n.op_type in LINEAR_OPS]
    convs = [n for n in nodes if n.op_type == "Conv"]

    def grouped(n) -> bool:
        return any(a.name == "group" and a.i > 1 for a in n.attribute)

    return {
        "stem": [convs[0].name],
        "depthwise": [n.name for n in convs[1:] if grouped(n)],
        "pointwise": [n.name for n in convs[1:] if not grouped(n)],
        "classifier": [n.name for n in nodes if n.op_type != "Conv"],
    }


def run(tag: str, out: Path) -> dict:
    fp32 = CHECKPOINT_DIR / f"{tag}.onnx"
    work = CHECKPOINT_DIR / "ablation"
    work.mkdir(exist_ok=True)
    prepared = work / f"{tag}.prep.onnx"
    quant_pre_process(str(fp32), str(prepared))
    input_name = ort.InferenceSession(str(fp32)).get_inputs()[0].name
    fams = families(prepared)
    every = [name for names in fams.values() for name in names]
    images = calibration_images()
    x, y = load_split("val")
    reference = onnx_logits(fp32, x).argmax(1)

    result = {
        "tag": tag,
        "split": "val",
        "images": int(len(y)),
        "fp32_accuracy": float((reference == y).mean()),
        "families": {},
    }
    for family, names in fams.items():
        path = work / f"{tag}.only_{family}.onnx"
        quantize_static(
            str(prepared),
            str(path),
            Calibration(input_name, images),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=LINEAR_OPS,
            nodes_to_exclude=[n for n in every if n not in names],
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.Percentile,
        )
        pred = onnx_logits(path, x).argmax(1)
        result["families"][family] = {
            "nodes": len(names),
            "accuracy": float((pred == y).mean()),
            "agreement_with_fp32": float((pred == reference).mean()),
        }
        print(family, len(names), result["families"][family])
        path.unlink()
    prepared.unlink()
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="student_distilled")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "ablation_families.json")
    args = ap.parse_args()
    np.random.seed(0)
    run(args.tag, args.out)


if __name__ == "__main__":
    main()

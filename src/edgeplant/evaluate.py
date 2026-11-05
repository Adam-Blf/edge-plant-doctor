"""Task metrics and size of every ONNX variant on the held-out test split."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import CHECKPOINT_DIR, RESULTS_DIR
from .data import load_split
from .preprocess import normalize
from .train import metrics


def onnx_logits(path: Path, x: np.ndarray, batch: int = 32) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(x), batch):
        stacked = np.concatenate([normalize(img) for img in x[i : i + batch]])
        out.append(sess.run(None, {name: stacked})[0])
    return np.concatenate(out)


def evaluate_models(paths: list[Path]) -> dict[str, dict]:
    x, y = load_split("test")
    results, reference = {}, {}
    for path in paths:
        logits = onnx_logits(path, x)
        base = path.stem.split(".")[0]
        if path.stem == base:
            reference[base] = logits
        results[path.stem] = {
            **metrics(logits, y),
            "size_mb": path.stat().st_size / 1e6,
            "_logits": logits,
        }
    for name, row in results.items():
        ref = reference.get(name.split(".")[0])
        row["top1_agreement_with_fp32"] = (
            float((row["_logits"].argmax(1) == ref.argmax(1)).mean()) if ref is not None else None
        )
        del row["_logits"]
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ONNX models on the test split")
    ap.add_argument("models", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "accuracy.json")
    args = ap.parse_args()
    paths = args.models or sorted(CHECKPOINT_DIR.glob("student_*.onnx"))
    results = evaluate_models(sorted(paths))
    args.out.parent.mkdir(exist_ok=True)
    merged = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    merged.update(results)
    args.out.write_text(json.dumps(dict(sorted(merged.items())), indent=1), encoding="utf-8")
    for name, row in results.items():
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()

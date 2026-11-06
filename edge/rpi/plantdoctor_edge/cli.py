"""Command line: classify one or more photos, optionally time the model."""

import argparse
import os
import statistics
import time
from pathlib import Path

from .engine import Engine, ImageRejectedError


def main() -> None:
    root = Path(os.environ.get("PLANT_DOCTOR_HOME", "."))
    ap = argparse.ArgumentParser(description="Diagnose leaf photos with the edge model")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--model", type=Path, default=root / "models" / "plant-doctor-int8.onnx")
    ap.add_argument("--labels", type=Path, default=root / "models" / "labels.json")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--bench", type=int, default=0, help="repeat each image N times, print timing")
    args = ap.parse_args()
    engine = Engine(args.model, args.labels)
    for path in args.images:
        try:
            data = path.read_bytes()
            result = engine.classify(data, args.top)
        except (OSError, ImageRejectedError) as exc:
            print(f"{path}: {exc}")
            continue
        print(path)
        for p in result["predictions"]:
            print(f"  {p['probability'] * 100:5.1f} %  {p['crop']}, {p['condition']}")
        if args.bench:
            times = []
            for _ in range(args.bench):
                start = time.perf_counter()
                engine.classify(data, args.top)
                times.append((time.perf_counter() - start) * 1000)
            print(f"  {args.bench} runs: median {statistics.median(times):.1f} ms")


if __name__ == "__main__":
    main()

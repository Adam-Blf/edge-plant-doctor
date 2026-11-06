"""Single-image latency of ONNX models. Needs only numpy and onnxruntime, so the same
module runs on the x86 workstation, inside the arm64 container and on a Raspberry Pi.
"""

import argparse
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

INPUT_SHAPE = (1, 3, 224, 224)


def session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def time_runs(sess: ort.InferenceSession, runs: int, warmup: int) -> list[float]:
    """Milliseconds for each of `runs` inferences after `warmup` discarded ones."""
    x = np.random.default_rng(0).normal(size=INPUT_SHAPE).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: x})
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, {name: x})
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def percentile(samples: list[float], q: float) -> float:
    return float(np.percentile(samples, q))


def benchmark(
    models: list[Path], threads: int, runs: int, warmup: int, rounds: int
) -> dict[str, dict]:
    """Interleave models over several rounds so a noisy neighbour hits all of them alike."""
    sessions = {m.stem: session(m, threads) for m in models}
    per_round: dict[str, list[dict]] = {name: [] for name in sessions}
    order = list(sessions)
    for _ in range(rounds):
        random.shuffle(order)
        for name in order:
            s = time_runs(sessions[name], runs, warmup)
            per_round[name].append({"p50": percentile(s, 50), "p95": percentile(s, 95)})
    return {
        name: {
            "p50_ms": statistics.median(r["p50"] for r in rows),
            "p95_ms": statistics.median(r["p95"] for r in rows),
            "p50_spread_ms": [min(r["p50"] for r in rows), max(r["p50"] for r in rows)],
            "rounds": rounds,
            "runs_per_round": runs,
        }
        for name, rows in per_round.items()
    }


def host_info(label: str, emulated: bool) -> dict:
    return {
        "label": label,
        "emulated": emulated,
        "machine": platform.machine(),
        "system": platform.platform(),
        "onnxruntime": ort.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure ONNX single-image latency")
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--label", default="x86 CPU")
    ap.add_argument("--emulated", action="store_true", help="QEMU or other emulation")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = {
        "host": host_info(args.label, args.emulated),
        "threads": {
            str(t): benchmark(args.models, t, args.runs, args.warmup, args.rounds)
            for t in args.threads
        },
    }
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

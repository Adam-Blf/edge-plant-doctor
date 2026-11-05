"""PyTorch checkpoint to ONNX (opset 17, dynamic batch)."""

import argparse
from pathlib import Path

import torch

from .config import CHECKPOINT_DIR, INPUT_SIZE
from .models import BUILDERS


def export_onnx(arch: str, checkpoint: Path, out: Path) -> Path:
    net = BUILDERS[arch](pretrained=False)
    net.load_state_dict(torch.load(checkpoint))
    net.eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        net,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a checkpoint to ONNX")
    ap.add_argument("--arch", choices=sorted(BUILDERS), required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out = export_onnx(
        args.arch, CHECKPOINT_DIR / f"{args.tag}.pt", CHECKPOINT_DIR / f"{args.tag}.onnx"
    )
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()

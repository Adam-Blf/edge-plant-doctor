"""Fine-tuning loop shared by the baseline, the teacher and the distilled student."""

import argparse
import json
import os
import time
from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .config import CHECKPOINT_DIR, MEAN, RESULTS_DIR, SEED, STD
from .data import load_split
from .models import BUILDERS

LossFn = Callable[[torch.Tensor, torch.Tensor, np.ndarray], torch.Tensor]
_MEAN = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(STD).view(1, 3, 1, 1)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(xb: np.ndarray) -> torch.Tensor:
    """uint8 NHWC batch to normalised float NCHW."""
    x = torch.from_numpy(np.ascontiguousarray(xb)).permute(0, 3, 1, 2).float() / 255.0
    return ((x - _MEAN) / _STD).contiguous(memory_format=torch.channels_last)


def augment(xb: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
    """Flips, quarter turns, small translation and brightness/contrast jitter."""
    out = np.empty_like(xb)
    for i, img in enumerate(xb):
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1]
        img = np.rot90(img, rng.integers(4))
        dy, dx = rng.integers(-16, 17, size=2)
        out[i] = np.roll(img, (dy, dx), axis=(0, 1))
    x = torch.from_numpy(out).permute(0, 3, 1, 2).float() / 255.0
    n = len(x)
    contrast = torch.from_numpy(rng.uniform(0.8, 1.2, (n, 1, 1, 1))).float()
    bright = torch.from_numpy(rng.uniform(-0.1, 0.1, (n, 1, 1, 1))).float()
    x = ((x - 0.5) * contrast + 0.5 + bright).clamp(0, 1)
    return ((x - _MEAN) / _STD).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, batch: int = 128) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    return np.concatenate(
        [
            model(to_tensor(x[i : i + batch]).to(device)).cpu().numpy()
            for i in range(0, len(x), batch)
        ]
    )


def metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(1)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def cross_entropy(logits: torch.Tensor, y: torch.Tensor, _idx: np.ndarray) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, y, label_smoothing=0.1)


def fit(
    model: nn.Module,
    tag: str,
    epochs: int,
    lr: float,
    loss_fn: LossFn = cross_entropy,
    batch: int = 32,
) -> dict:
    """Train with AdamW and one-cycle schedule, keep the best validation checkpoint."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    device = pick_device()
    model = model.to(device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    history, best, start = [], -1.0, time.time()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(xtr))
        total = 0.0
        for s in range(len(xtr) // batch):
            idx = np.sort(order[s * batch : (s + 1) * batch])
            xb = augment(xtr[idx], rng).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            loss = loss_fn(model(xb), yb, idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        val = metrics(predict_logits(model, xva), yva)
        history.append({"epoch": epoch + 1, "loss": total / (len(xtr) // batch), **val})
        print(json.dumps(history[-1]), f"{time.time() - start:.0f}s", flush=True)
        if val["accuracy"] > best:
            best = val["accuracy"]
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(state, CHECKPOINT_DIR / f"{tag}.pt")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "tag": tag,
        "epochs": epochs,
        "lr": lr,
        "seconds": time.time() - start,
        "history": history,
    }
    (RESULTS_DIR / f"train_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a model on the cached subset")
    ap.add_argument("--arch", choices=sorted(BUILDERS), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    fit(BUILDERS[args.arch](), args.tag, args.epochs, args.lr)


if __name__ == "__main__":
    main()

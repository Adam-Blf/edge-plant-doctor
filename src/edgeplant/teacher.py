"""EfficientNet-B0 teacher: the first stages stay frozen and their output is cached.

Fine-tuning the whole network on a shared CPU is out of budget, so the teacher keeps
`features[:FROZEN]` from ImageNet, caches their activations once, and fine-tunes the
remaining stages (about 95% of the parameters) plus the classifier on those caches.
The same code runs on a GPU in the Colab notebook with the full dataset.
"""

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from .config import CHECKPOINT_DIR, RESULTS_DIR, SEED
from .data import load_split
from .models import teacher
from .train import metrics, pick_device, to_tensor

FROZEN = 5


def tail(net: nn.Module) -> nn.Sequential:
    return nn.Sequential(net.features[FROZEN:], net.avgpool, nn.Flatten(), net.classifier)


@torch.no_grad()
def cache_features(net: nn.Module, split: str, device: torch.device, batch: int = 64) -> np.ndarray:
    """Activations after the frozen stages, stored as float16 (N, C, H, W)."""
    x, _ = load_split(split)
    head = net.features[:FROZEN].eval().to(device)
    chunks = [
        head(to_tensor(x[i : i + batch]).to(device)).half().cpu().numpy()
        for i in range(0, len(x), batch)
    ]
    out = np.concatenate(chunks)
    np.save(CHECKPOINT_DIR / f"teacher_feats_{split}.npy", out)
    return out


@torch.no_grad()
def tail_logits(net: nn.Module, feats: np.ndarray, device: torch.device) -> np.ndarray:
    model = tail(net).eval().to(device)
    return np.concatenate(
        [
            model(torch.from_numpy(feats[i : i + 256]).float().to(device)).cpu().numpy()
            for i in range(0, len(feats), 256)
        ]
    )


def flip(batch: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    if rng.random() < 0.5:
        batch = batch.flip(3)
    if rng.random() < 0.5:
        batch = batch.flip(2)
    return batch


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the cached-feature teacher")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    device = pick_device()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    net = teacher()
    start = time.time()
    feats = {s: cache_features(net, s, device) for s in ("train", "val", "test")}
    labels = {s: load_split(s)[1] for s in feats}
    print(f"features cached in {time.time() - start:.0f}s", flush=True)
    model = tail(net).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    batch, n = 64, len(feats["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * (n // batch), pct_start=0.2
    )
    history, best = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(n)
        for s in range(n // batch):
            idx = np.sort(order[s * batch : (s + 1) * batch])
            xb = flip(torch.from_numpy(feats["train"][idx]).float().to(device), rng)
            yb = torch.from_numpy(labels["train"][idx]).to(device)
            loss = nn.functional.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        val = metrics(tail_logits(net, feats["val"], device), labels["val"])
        history.append({"epoch": epoch + 1, **val})
        print(json.dumps(history[-1]), f"{time.time() - start:.0f}s", flush=True)
        if val["accuracy"] > best:
            best = val["accuracy"]
            state = {k: v.cpu() for k, v in net.state_dict().items()}
            torch.save(state, CHECKPOINT_DIR / "teacher.pt")
    net.load_state_dict(torch.load(CHECKPOINT_DIR / "teacher.pt"))
    np.save(CHECKPOINT_DIR / "teacher_logits_train.npy", tail_logits(net, feats["train"], device))
    test = metrics(tail_logits(net, feats["test"], device), labels["test"])
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "frozen_stages": FROZEN,
        "history": history,
        "test": test,
        "seconds": time.time() - start,
    }
    (RESULTS_DIR / "train_teacher.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(test))


if __name__ == "__main__":
    main()

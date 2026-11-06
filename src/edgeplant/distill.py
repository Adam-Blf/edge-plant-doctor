"""Knowledge distillation: the EfficientNet-B0 teacher guides the MobileNetV3 student."""

import argparse
import os

import numpy as np
import torch
from torch import nn

from .config import CHECKPOINT_DIR
from .models import student
from .train import LossFn, cross_entropy, fit


def distillation_loss(teacher_logits: np.ndarray, temperature: float, alpha: float) -> LossFn:
    """alpha * T^2 * KL(teacher || student) + (1 - alpha) * cross entropy on hard labels.

    Teacher logits are precomputed once per training image (see `teacher.py`), indexed
    by the position of each image in the training split.
    """
    table = torch.from_numpy(teacher_logits).float()

    def loss(logits: torch.Tensor, y: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
        soft = nn.functional.softmax(table[idx].to(logits.device) / temperature, dim=1)
        kd = nn.functional.kl_div(
            nn.functional.log_softmax(logits / temperature, dim=1), soft, reduction="batchmean"
        )
        return alpha * temperature**2 * kd + (1 - alpha) * cross_entropy(logits, y, idx)

    return loss


def main() -> None:
    ap = argparse.ArgumentParser(description="Distil the teacher checkpoint into the student")
    ap.add_argument("--tag", default="student_distilled")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    logits = np.load(CHECKPOINT_DIR / "teacher_logits_train.npy")
    loss = distillation_loss(logits, args.temperature, args.alpha)
    fit(student(), args.tag, args.epochs, args.lr, loss_fn=loss)


if __name__ == "__main__":
    main()

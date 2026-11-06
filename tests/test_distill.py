import numpy as np
import torch

from edgeplant.distill import distillation_loss


def test_pure_distillation_is_zero_when_student_matches_teacher():
    teacher_logits = np.random.default_rng(0).normal(size=(8, 38)).astype(np.float32)
    loss = distillation_loss(teacher_logits, temperature=4.0, alpha=1.0)
    student_logits = torch.from_numpy(teacher_logits[[2, 5]])
    value = loss(student_logits, torch.tensor([0, 1]), np.array([2, 5]))
    assert abs(value.item()) < 1e-5


def test_distillation_penalises_a_wrong_student():
    teacher_logits = np.eye(38, dtype=np.float32)[:8] * 10
    loss = distillation_loss(teacher_logits, temperature=2.0, alpha=0.7)
    y = torch.tensor([0, 1])
    idx = np.array([0, 1])
    good = loss(torch.from_numpy(teacher_logits[idx]), y, idx)
    bad = loss(torch.from_numpy(teacher_logits[[3, 4]]), y, idx)
    assert bad.item() > good.item()

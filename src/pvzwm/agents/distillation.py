from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def masked_policy_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    legal_masks: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL-distill a legality-masked teacher policy into a student policy."""

    if not (
        student_logits.shape == teacher_logits.shape == legal_masks.shape
    ):
        raise ValueError("Student, teacher, and legal-mask tensors must align")
    if student_logits.ndim != 2:
        raise ValueError("Policy distillation expects [batch, actions] tensors")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not bool(legal_masks.any(dim=1).all()):
        raise ValueError("Every distillation state must have a legal action")

    scaled_student = student_logits.masked_fill(~legal_masks, -torch.inf) / temperature
    scaled_teacher = teacher_logits.masked_fill(~legal_masks, -torch.inf) / temperature
    teacher_log_probability = F.log_softmax(scaled_teacher, dim=-1).detach()
    teacher_probability = teacher_log_probability.exp()
    student_log_probability = F.log_softmax(scaled_student, dim=-1)
    divergence = teacher_probability * (
        teacher_log_probability - student_log_probability
    ).nan_to_num(0.0)
    return divergence.sum(dim=-1).mean() * temperature**2


def balanced_level_batch_indices(
    level_sizes: list[int],
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sample a batch with level counts differing by at most one."""

    if not level_sizes or any(size < 1 for size in level_sizes):
        raise ValueError("Every teacher level must contain at least one state")
    if batch_size < len(level_sizes):
        raise ValueError("batch_size must be at least the number of levels")
    quotient, remainder = divmod(batch_size, len(level_sizes))
    extra_levels = set(
        int(index) for index in rng.permutation(len(level_sizes))[:remainder]
    )
    return [
        rng.integers(0, size, size=quotient + int(index in extra_levels))
        for index, size in enumerate(level_sizes)
    ]

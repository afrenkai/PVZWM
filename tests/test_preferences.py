import pytest
import torch

from pvzwm.agents.preferences import (
    contrastive_opsd_loss,
    error_localized_window,
    segment_log_probabilities,
    trajectory_dpo_loss,
)


def test_segment_log_probabilities_ignore_padding_and_normalize_length() -> None:
    values = torch.tensor([[-1.0, -3.0, -100.0], [-2.0, -2.0, -2.0]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    assert segment_log_probabilities(values, valid).tolist() == pytest.approx(
        [-2.0, -2.0]
    )


def test_dpo_gradient_increases_preferred_margin() -> None:
    chosen = torch.tensor([0.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    loss = trajectory_dpo_loss(
        chosen,
        rejected,
        torch.zeros(1),
        torch.zeros(1),
        beta=1.0,
    )
    loss.backward()
    assert chosen.grad is not None and chosen.grad.item() < 0.0
    assert rejected.grad is not None and rejected.grad.item() > 0.0


def test_contrastive_opsd_cancels_identical_privileged_context_effects() -> None:
    student = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True)
    teacher = torch.tensor([[2.0, -1.0, 0.5]])
    legal = torch.ones_like(student, dtype=torch.bool)
    loss = contrastive_opsd_loss(student, teacher, teacher, legal)
    loss.backward()
    assert student.grad is not None
    assert student.grad.abs().max().item() < 1e-6


def test_error_localization_centers_on_first_action_divergence() -> None:
    assert error_localized_window([1, 2, 3, 4], [1, 2, 8, 9], radius=1) == (1, 4)

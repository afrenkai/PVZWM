from __future__ import annotations

import torch
import torch.nn.functional as F

from pvzwm.agents.ppo import PolicyModel


def masked_action_log_probabilities(
    model: PolicyModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    logits, _ = model(states)
    logits = logits.masked_fill(~legal_masks, -torch.inf)
    return torch.distributions.Categorical(logits=logits).log_prob(actions)


def segment_log_probabilities(
    token_log_probabilities: torch.Tensor,
    valid_steps: torch.Tensor,
    *,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Reduce action log-probabilities over fixed shared-prefix continuation chunks."""

    if token_log_probabilities.shape != valid_steps.shape:
        raise ValueError("Log probabilities and valid-step masks must align")
    if token_log_probabilities.ndim != 2:
        raise ValueError("Segment tensors must have shape [batch, time]")
    weights = valid_steps.to(token_log_probabilities.dtype)
    totals = (token_log_probabilities * weights).sum(dim=1)
    if not length_normalize:
        return totals
    return totals / weights.sum(dim=1).clamp_min(1.0)


def trajectory_dpo_loss(
    chosen_policy_log_probability: torch.Tensor,
    rejected_policy_log_probability: torch.Tensor,
    chosen_reference_log_probability: torch.Tensor,
    rejected_reference_log_probability: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """DPO over matched continuation segments, not unmatched whole MDP episodes."""

    shapes = {
        value.shape
        for value in (
            chosen_policy_log_probability,
            rejected_policy_log_probability,
            chosen_reference_log_probability,
            rejected_reference_log_probability,
        )
    }
    if len(shapes) != 1:
        raise ValueError("All DPO log-probability tensors must align")
    if beta <= 0.0 or not 0.0 <= label_smoothing < 0.5:
        raise ValueError("Invalid DPO beta or label smoothing")
    policy_margin = chosen_policy_log_probability - rejected_policy_log_probability
    reference_margin = chosen_reference_log_probability - rejected_reference_log_probability
    logits = beta * (policy_margin - reference_margin)
    positive = -F.logsigmoid(logits)
    negative = -F.logsigmoid(-logits)
    return ((1.0 - label_smoothing) * positive + label_smoothing * negative).mean()


def contrastive_opsd_loss(
    student_logits: torch.Tensor,
    positive_teacher_logits: torch.Tensor,
    negative_teacher_logits: torch.Tensor,
    legal_masks: torch.Tensor,
    *,
    contrast_clip: float = 5.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Distill only the teacher shift unique to a correct privileged future.

    Shared positive/negative privileged-context effects cancel, preventing a
    generic teacher sharpening or action-frequency shift from becoming the signal.
    Teacher tensors are detached; only the deployment-time student is optimized.
    """

    if not (
        student_logits.shape
        == positive_teacher_logits.shape
        == negative_teacher_logits.shape
        == legal_masks.shape
    ):
        raise ValueError("Student, teachers, and legal masks must align")
    if contrast_clip <= 0.0 or temperature <= 0.0:
        raise ValueError("Contrast clipping and temperature must be positive")
    masked_student = student_logits.masked_fill(~legal_masks, -torch.inf)
    positive_log_probability = F.log_softmax(
        positive_teacher_logits.masked_fill(~legal_masks, -torch.inf) / temperature,
        dim=-1,
    )
    negative_log_probability = F.log_softmax(
        negative_teacher_logits.masked_fill(~legal_masks, -torch.inf) / temperature,
        dim=-1,
    )
    contrast = (positive_log_probability - negative_log_probability).nan_to_num(0.0)
    contrast = contrast.clamp(-contrast_clip, contrast_clip).detach()
    target_logits = masked_student.detach() / temperature + contrast
    target = F.softmax(target_logits, dim=-1)
    student_log_probability = F.log_softmax(masked_student / temperature, dim=-1)
    return -(target * student_log_probability.nan_to_num(0.0)).sum(dim=-1).mean()


def error_localized_window(
    preferred_actions: list[int],
    rejected_actions: list[int],
    *,
    radius: int = 8,
) -> tuple[int, int]:
    """Return a compact training window around the first divergent decision."""

    if radius < 0:
        raise ValueError("radius cannot be negative")
    limit = min(len(preferred_actions), len(rejected_actions))
    divergence = next(
        (
            index
            for index in range(limit)
            if preferred_actions[index] != rejected_actions[index]
        ),
        limit,
    )
    return max(0, divergence - radius), min(
        max(len(preferred_actions), len(rejected_actions)), divergence + radius + 1
    )

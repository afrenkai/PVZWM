from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_reduce


@dataclass(slots=True)
class VICRegComponents:
    total: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor


def _samples(embeddings: torch.Tensor, name: str) -> torch.Tensor:
    if embeddings.ndim < 2:
        raise ValueError(f"{name} expects at least [samples, features]")
    flattened = embeddings.reshape(-1, embeddings.shape[-1]).float()
    if flattened.shape[0] < 2:
        raise ValueError(f"{name} requires at least two samples")
    return flattened


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_regularizer(
    embeddings: torch.Tensor,
    *,
    eps: float = 1e-4,
    variance_weight: float = 1.0,
    covariance_weight: float = 1.0,
) -> VICRegComponents:
    """VICReg variance and covariance terms from Bardes, Ponce, and LeCun."""
    z = _samples(embeddings, "VICReg")
    centered = z - z.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    variance = F.relu(1.0 - standard_deviation).mean()
    covariance_matrix = centered.T @ centered / max(z.shape[0] - 1, 1)
    covariance = _off_diagonal(covariance_matrix).square().sum() / z.shape[1]
    total = variance_weight * variance + covariance_weight * covariance
    return VICRegComponents(total, variance, covariance)


def vicreg_objective(
    predicted: torch.Tensor,
    target: torch.Tensor,
    regularization_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not 0.0 <= regularization_weight <= 1.0:
        raise ValueError("regularization_weight must be in [0, 1]")
    prediction = F.mse_loss(predicted, target)
    predicted_regularizer = vicreg_regularizer(predicted)
    target_regularizer = vicreg_regularizer(target)
    variance = 0.5 * (
        predicted_regularizer.variance + target_regularizer.variance
    )
    covariance = 0.5 * (
        predicted_regularizer.covariance + target_regularizer.covariance
    )
    regularizer = variance + covariance
    total = (
        (1.0 - regularization_weight) * prediction
        + regularization_weight * regularizer
    )
    return total, {
        "prediction": prediction,
        "vicreg": regularizer,
        "variance": variance,
        "covariance": covariance,
    }


def _distributed_average(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        return all_reduce(value, op=dist.ReduceOp.SUM) / dist.get_world_size()
    return value


def sigreg_regularizer(
    embeddings: torch.Tensor,
    *,
    slices: int = 64,
    integration_points: int = 17,
    t_max: float = 3.0,
) -> torch.Tensor:
    """LeJEPA SIGReg using sliced Epps–Pulley Gaussian tests."""
    z = _samples(embeddings, "SIGReg")
    if slices < 1:
        raise ValueError("slices must be positive")
    if integration_points < 3 or integration_points % 2 == 0:
        raise ValueError("integration_points must be odd and at least three")

    directions = torch.randn(
        z.shape[-1], slices, device=z.device, dtype=z.dtype
    )
    directions = F.normalize(directions, dim=0)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(directions, src=0)
    projected = z @ directions
    integration = torch.linspace(
        0.0, t_max, integration_points, device=z.device, dtype=z.dtype
    )
    gaussian_cf = torch.exp(-0.5 * integration.square())
    step = t_max / (integration_points - 1)
    weights = torch.full_like(integration, 2.0 * step)
    weights[0] = step
    weights[-1] = step
    weights = weights * gaussian_cf

    arguments = projected.unsqueeze(-1) * integration
    cosine_mean = _distributed_average(torch.cos(arguments).mean(dim=0))
    sine_mean = _distributed_average(torch.sin(arguments).mean(dim=0))
    error = (cosine_mean - gaussian_cf).square() + sine_mean.square()
    sample_count = z.shape[0] * (dist.get_world_size() if dist.is_initialized() else 1)
    return ((error @ weights) * sample_count).mean()


def sigreg_objective(
    predicted: torch.Tensor,
    target: torch.Tensor,
    regularization_weight: float = 0.05,
    slices: int = 64,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not 0.0 <= regularization_weight <= 1.0:
        raise ValueError("regularization_weight must be in [0, 1]")
    prediction = F.mse_loss(predicted, target)
    # LeJEPA/LeWM regularize encoder embeddings, not predictor outputs. The
    # predictor is trained through the invariance/prediction term.
    regularizer = sigreg_regularizer(target, slices=slices)
    total = (
        (1.0 - regularization_weight) * prediction
        + regularization_weight * regularizer
    )
    return total, {
        "prediction": prediction,
        "sigreg": regularizer,
    }

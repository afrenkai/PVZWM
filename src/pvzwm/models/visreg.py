from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class VISRegComponents:
    total: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor
    shape: torch.Tensor


def _normal_quantiles(count: int, reference: torch.Tensor) -> torch.Tensor:
    # erfinv form avoids constructing a distribution object and works on MPS.
    u = torch.arange(1, count + 1, device=reference.device, dtype=reference.dtype)
    u = u / (count + 1)
    return torch.sqrt(reference.new_tensor(2.0)) * torch.erfinv(2.0 * u - 1.0)


def visreg_regularizer(
    embeddings: torch.Tensor,
    slices: int = 64,
    eps: float = 1e-6,
    center_weight: float = 1.0,
    scale_weight: float = 1.0,
    shape_weight: float = 1.0,
) -> VISRegComponents:
    """VISReg from Wu, Balestriero, and Levine (2026).

    A three-dimensional input follows the released ``[views, batch, features]``
    contract and computes the distribution constraints independently per view.
    A matrix is treated as one view; higher-rank spatial tensors are flattened
    into one sample axis. Float32 is retained for MPS numerical stability.
    """
    if embeddings.ndim < 2:
        raise ValueError("VISReg expects at least [samples, features]")
    if embeddings.ndim == 2:
        z = embeddings.unsqueeze(0).float()
    elif embeddings.ndim == 3:
        z = embeddings.float()
    else:
        z = embeddings.reshape(1, -1, embeddings.shape[-1]).float()
    if z.shape[1] < 2:
        raise ValueError("VISReg requires at least two samples")
    if slices < 1:
        raise ValueError("slices must be positive")

    mean = z.mean(dim=1, keepdim=True)
    center_loss = mean.square().mean()
    centered = z - mean
    # Match the released implementation exactly: an RMS over the batch axis,
    # with a small additive floor retained in scale and normalization terms.
    std = centered.norm(dim=1).div(z.shape[1] ** 0.5) + eps
    scale_loss = (1.0 - std).square().mean()

    normalized = centered / std.detach().unsqueeze(1)
    directions = torch.randn(
        normalized.shape[-1], slices, device=normalized.device, dtype=normalized.dtype
    )
    directions = F.normalize(directions, dim=0)
    projected = (normalized @ directions).sort(dim=1).values
    target = _normal_quantiles(normalized.shape[1], normalized).view(1, -1, 1)
    shape_loss = (projected - target).square().mean()

    total = (
        center_weight * center_loss
        + scale_weight * scale_loss
        + shape_weight * shape_loss
    )
    return VISRegComponents(total, center_loss, scale_loss, shape_loss)


def visreg_objective(
    predicted: torch.Tensor,
    target: torch.Tensor,
    regularization_weight: float = 0.05,
    slices: int = 64,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not 0.0 <= regularization_weight <= 1.0:
        raise ValueError("regularization_weight must be in [0, 1]")
    prediction_loss = F.mse_loss(predicted, target)
    combined = torch.cat((predicted, target), dim=0).reshape(
        -1, predicted.shape[-1]
    )
    regularization = visreg_regularizer(combined, slices=slices)
    total = (
        (1.0 - regularization_weight) * prediction_loss
        + regularization_weight * regularization.total
    )
    metrics = {
        "loss": total,
        "prediction": prediction_loss,
        "visreg": regularization.total,
        "center": regularization.center,
        "scale": regularization.scale,
        "shape": regularization.shape,
    }
    return total, metrics

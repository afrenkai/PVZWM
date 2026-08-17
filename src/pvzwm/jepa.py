from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pvzwm.models.regularizers import sigreg_objective, vicreg_objective
from pvzwm.models.visreg import visreg_objective


class VectorEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class DivisibleSpatialPool(nn.Module):
    """MPS-safe spatial average pool with an exact requested output grid."""

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.output_size = (height, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output_height, output_width = self.output_size
        height, width = value.shape[-2:]
        target_height = ((height + output_height - 1) // output_height) * output_height
        target_width = ((width + output_width - 1) // output_width) * output_width
        if target_height != height or target_width != width:
            value = F.pad(
                value,
                (0, target_width - width, 0, target_height - height),
                mode="replicate",
            )
        kernel = (target_height // output_height, target_width // output_width)
        return F.avg_pool2d(value, kernel_size=kernel, stride=kernel)


class ConvEncoder(nn.Module):
    """Compact from-scratch pixel encoder suitable for MPS experiments."""

    def __init__(
        self,
        channels: int = 3,
        latent_dim: int = 128,
        pool_height: int = 1,
        pool_width: int = 1,
    ) -> None:
        super().__init__()
        if pool_height < 1 or pool_width < 1:
            raise ValueError("Spatial pooling dimensions must be positive")
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            DivisibleSpatialPool(pool_height, pool_width),
        )
        self.project = nn.Linear(128 * pool_height * pool_width, latent_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.project(self.features(observation).flatten(1))


@dataclass(slots=True)
class JEPAPrediction:
    embedding: torch.Tensor
    reward: torch.Tensor
    done_logit: torch.Tensor


class ActionConditionedJEPA(nn.Module):
    """One-step action-conditioned JEPA with selectable regularization."""

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        latent_dim: int = 128,
        action_embed_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.reward_head = nn.Linear(latent_dim, 1)
        self.done_head = nn.Linear(latent_dim, 1)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def predict(self, embedding: torch.Tensor, action: torch.Tensor) -> JEPAPrediction:
        predicted = self.predictor(
            torch.cat((embedding, self.action_embedding(action)), dim=-1)
        )
        return JEPAPrediction(
            predicted,
            self.reward_head(predicted).squeeze(-1),
            self.done_head(predicted).squeeze(-1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> JEPAPrediction:
        return self.predict(self.encode(observation), action)

    def loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
        regularizer: str = "visreg",
        regularization_weight: float = 0.05,
        slices: int = 64,
        reward_weight: float = 1.0,
        done_weight: float = 0.25,
    ) -> dict[str, torch.Tensor]:
        source = self.encode(observation)
        target = self.encode(next_observation)
        prediction = self.predict(source, action)
        if regularizer == "visreg":
            latent_loss, metrics = visreg_objective(
                prediction.embedding,
                target,
                regularization_weight=regularization_weight,
                slices=slices,
            )
        elif regularizer == "sigreg":
            latent_loss, metrics = sigreg_objective(
                prediction.embedding,
                target,
                regularization_weight=regularization_weight,
                slices=slices,
            )
        elif regularizer == "vicreg":
            latent_loss, metrics = vicreg_objective(
                prediction.embedding,
                target,
                regularization_weight=regularization_weight,
            )
        elif regularizer == "none":
            latent_loss = F.mse_loss(prediction.embedding, target)
            metrics = {"prediction": latent_loss}
        else:
            raise ValueError(f"Unsupported JEPA regularizer: {regularizer}")
        reward_loss = F.smooth_l1_loss(prediction.reward, reward)
        done_loss = F.binary_cross_entropy_with_logits(prediction.done_logit, done)
        total = latent_loss + reward_weight * reward_loss + done_weight * done_loss
        return {**metrics, "loss": total, "reward": reward_loss, "done": done_loss}

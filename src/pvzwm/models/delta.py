from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class PixelDeltaStep:
    observation: torch.Tensor
    reward: torch.Tensor
    done_logit: torch.Tensor


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class PixelDeltaWorldModel(nn.Module):
    """Action-conditioned pixel residual initialized at exact persistence.

    PvZ frames are mostly static between decisions. Predicting a complete frame
    therefore gives a model an unnecessary opportunity to lose to the copy-last
    baseline. This model predicts only the bounded change around the current
    observation; its final convolution is zero initialized, so the untrained
    dynamics are exactly persistence rather than a random reconstruction.
    """

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_dim: int,
        hidden_channels: int = 64,
        residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        if len(observation_shape) != 3 or min(observation_shape) < 1:
            raise ValueError("Pixel delta dynamics require a positive CHW shape")
        if action_dim < 1 or hidden_channels < 8 or hidden_channels % 8:
            raise ValueError(
                "Action dimension must be positive and hidden channels divisible by eight"
            )
        if residual_blocks < 1:
            raise ValueError("Pixel delta dynamics require at least one residual block")
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.hidden_channels = hidden_channels
        self.residual_blocks = residual_blocks
        channels = observation_shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
        )
        self.action_embedding = nn.Embedding(action_dim, hidden_channels)
        self.dynamics = nn.Sequential(
            *[_ResidualBlock(hidden_channels) for _ in range(residual_blocks)]
        )
        self.delta_head = nn.Sequential(
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
        )
        final = self.delta_head[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.outcome_trunk = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels), nn.SiLU()
        )
        self.reward_head = nn.Linear(hidden_channels, 1)
        self.done_head = nn.Linear(hidden_channels, 1)
        self.legal_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, action_dim),
        )

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def _condition(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.dynamics(
            latent + self.action_embedding(action)[:, :, None, None]
        )

    def legal_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.legal_head(latent.mean(dim=(-2, -1)))

    def legal_logits(self, observation: torch.Tensor) -> torch.Tensor:
        return self.legal_logits_from_latent(self.encode(observation))

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> PixelDeltaStep:
        latent = self.encode(observation)
        return self._step_from_latent(observation, action, latent)

    def _step_from_latent(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        latent: torch.Tensor,
    ) -> PixelDeltaStep:
        conditioned = self._condition(latent, action)
        delta = self.delta_head(conditioned)
        delta = F.interpolate(
            delta,
            size=self.observation_shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        predicted = (observation + torch.tanh(delta)).clamp(0.0, 1.0)
        pooled = torch.cat(
            (latent.mean(dim=(-2, -1)), self.action_embedding(action)), dim=-1
        )
        outcome = self.outcome_trunk(pooled)
        return PixelDeltaStep(
            predicted,
            self.reward_head(outcome).squeeze(-1),
            self.done_head(outcome).squeeze(-1),
        )

    def loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        pixel_weight: float = 1.0,
        changed_pixel_weight: float = 1.0,
        change_threshold: float = 1.0 / 255.0,
        reward_weight: float = 1.0,
        done_weight: float = 0.25,
        legal_weight: float = 0.25,
    ) -> dict[str, torch.Tensor]:
        if pixel_weight < 0.0 or changed_pixel_weight < 0.0:
            raise ValueError("Pixel loss weights must be nonnegative")
        if change_threshold < 0.0:
            raise ValueError("Pixel change threshold must be nonnegative")
        latent = self.encode(observation)
        step = self._step_from_latent(observation, action, latent)
        squared_error = (step.observation - next_observation).square()
        pixel = squared_error.mean()
        changed = (next_observation - observation).abs() > change_threshold
        changed_count = changed.sum().clamp_min(1)
        changed_pixel = (squared_error * changed).sum() / changed_count
        reward_loss = F.smooth_l1_loss(step.reward, reward)
        done_loss = F.binary_cross_entropy_with_logits(step.done_logit, done)
        if legal_mask is None:
            legal_loss = pixel.new_zeros(())
        else:
            targets = legal_mask.to(pixel.dtype)
            logits = self.legal_logits_from_latent(latent)
            positives = targets.sum().clamp_min(1.0)
            negatives = (1.0 - targets).sum().clamp_min(1.0)
            legal_loss = 0.5 * (
                (F.softplus(-logits) * targets).sum() / positives
                + (F.softplus(logits) * (1.0 - targets)).sum() / negatives
            )
        total = (
            pixel_weight * pixel
            + changed_pixel_weight * changed_pixel
            + reward_weight * reward_loss
            + done_weight * done_loss
            + legal_weight * legal_loss
        )
        return {
            "loss": total,
            "pixel": pixel,
            "changed_pixel": changed_pixel,
            "reward": reward_loss,
            "done": done_loss,
            "legal": legal_loss,
        }

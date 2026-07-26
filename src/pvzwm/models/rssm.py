from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class RSSMState:
    deterministic: torch.Tensor
    stochastic: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class PixelRSSMEncoder(nn.Module):
    """MPS-safe convolutional encoder for a fixed channel-first frame shape."""

    def __init__(self, observation_shape: tuple[int, int, int], hidden_dim: int) -> None:
        super().__init__()
        channels, height, width = observation_shape
        encoded_height = (height + 7) // 8
        encoded_width = (width + 7) // 8
        if encoded_height % 3 or encoded_width % 5:
            raise ValueError(
                "Pixel RSSM frame dimensions must downsample to a grid divisible by 3x5"
            )
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(encoded_height // 3, encoded_width // 5),
                stride=(encoded_height // 3, encoded_width // 5),
            ),
            nn.Flatten(),
            nn.Linear(128 * 3 * 5, hidden_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.features(observation)


class PixelRSSMDecoder(nn.Module):
    """Decode RSSM features through progressive fixed-size upsampling."""

    def __init__(
        self, feature_dim: int, observation_shape: tuple[int, int, int]
    ) -> None:
        super().__init__()
        channels, height, width = observation_shape
        self.observation_shape = observation_shape
        self.project = nn.Sequential(nn.Linear(feature_dim, 128 * 3 * 5), nn.ELU())
        self.decode = nn.Sequential(
            nn.Upsample(size=(15, 20), mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Upsample(size=(60, 80), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Upsample(size=(height, width), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.project(features).reshape(-1, 128, 3, 5)
        return self.decode(projected)


class RecurrentStateSpaceModel(nn.Module):
    """Compact Gaussian RSSM baseline inspired by PlaNet/Dreamer."""

    def __init__(
        self,
        observation_dim: int | tuple[int, int, int],
        action_dim: int,
        deterministic_dim: int = 128,
        stochastic_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.observation_shape = (
            (observation_dim,) if isinstance(observation_dim, int) else observation_dim
        )
        self.pixel_observations = not isinstance(observation_dim, int)
        if self.pixel_observations and len(self.observation_shape) != 3:
            raise ValueError("Pixel RSSM observations must be channel-first [C, H, W]")
        self.action_embedding = nn.Embedding(action_dim, 32)
        self.observation_encoder = (
            PixelRSSMEncoder(self.observation_shape, hidden_dim)
            if self.pixel_observations
            else nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )
        self.recurrent = nn.GRUCell(stochastic_dim + 32, deterministic_dim)
        self.prior = nn.Linear(deterministic_dim, 2 * stochastic_dim)
        self.posterior = nn.Linear(deterministic_dim + hidden_dim, 2 * stochastic_dim)
        feature_dim = deterministic_dim + stochastic_dim
        self.observation_head = (
            PixelRSSMDecoder(feature_dim, self.observation_shape)
            if self.pixel_observations
            else nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, observation_dim),
            )
        )
        self.reward_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        self.done_head = nn.Linear(feature_dim, 1)

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        deterministic = torch.zeros(batch_size, self.deterministic_dim, device=device)
        stochastic = torch.zeros(batch_size, self.stochastic_dim, device=device)
        return RSSMState(deterministic, stochastic, stochastic, torch.ones_like(stochastic))

    @staticmethod
    def _stats(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = raw.chunk(2, dim=-1)
        return mean, F.softplus(raw_std) + 0.1

    @staticmethod
    def _sample(mean: torch.Tensor, std: torch.Tensor, sample: bool) -> torch.Tensor:
        return mean + std * torch.randn_like(std) if sample else mean

    def imagine(self, previous: RSSMState, action: torch.Tensor, sample: bool = True) -> RSSMState:
        deterministic = self.recurrent(
            torch.cat((previous.stochastic, self.action_embedding(action)), dim=-1),
            previous.deterministic,
        )
        mean, std = self._stats(self.prior(deterministic))
        return RSSMState(deterministic, self._sample(mean, std, sample), mean, std)

    def observe(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        observation: torch.Tensor,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        prior = self.imagine(previous, action, sample=sample)
        encoded = self.observation_encoder(observation)
        mean, std = self._stats(self.posterior(torch.cat((prior.deterministic, encoded), dim=-1)))
        posterior = RSSMState(
            prior.deterministic, self._sample(mean, std, sample), mean, std
        )
        return posterior, prior

    def decode(self, state: RSSMState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.cat((state.deterministic, state.stochastic), dim=-1)
        return (
            self.observation_head(features),
            self.reward_head(features).squeeze(-1),
            self.done_head(features).squeeze(-1),
        )

    @staticmethod
    def _kl_divergence(posterior: RSSMState, prior: RSSMState) -> torch.Tensor:
        variance_ratio = posterior.std.square() / prior.std.square()
        mean_difference = (posterior.mean - prior.mean).square() / prior.std.square()
        return 0.5 * (
            variance_ratio + mean_difference - 1.0 + 2.0 * (prior.std.log() - posterior.std.log())
        ).sum(dim=-1)

    def loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
        kl_weight: float = 0.1,
        free_nats: float = 1.0,
        reward_weight: float = 1.0,
        done_weight: float = 0.25,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Sequence ELBO-style loss over tensors shaped [batch, time, ...]."""
        expected_dimensions = 5 if self.pixel_observations else 3
        if observation.ndim != expected_dimensions:
            expected = "[batch, time, channels, height, width]" if self.pixel_observations else "[batch, time, features]"
            raise ValueError(f"RSSM loss expects {expected} observations")
        batch_size, sequence_length = observation.shape[:2]
        state = self.initial(batch_size, observation.device)
        bootstrap_action = torch.zeros(batch_size, dtype=torch.long, device=observation.device)
        state, _ = self.observe(state, bootstrap_action, observation[:, 0])

        observation_loss = observation.new_zeros(())
        reward_loss = observation.new_zeros(())
        done_loss = observation.new_zeros(())
        kl_loss = observation.new_zeros(())
        for index in range(sequence_length):
            step_action = action[:, index].contiguous()
            step_observation = next_observation[:, index].contiguous()
            step_reward = reward[:, index].contiguous()
            step_done = done[:, index].contiguous()
            state, prior = self.observe(state, step_action, step_observation)
            predicted_observation, predicted_reward, predicted_done = self.decode(state)
            observation_loss = observation_loss + F.smooth_l1_loss(
                predicted_observation, step_observation
            )
            reward_loss = reward_loss + F.smooth_l1_loss(predicted_reward, step_reward)
            done_loss = done_loss + F.binary_cross_entropy_with_logits(
                predicted_done, step_done
            )
            step_kl = self._kl_divergence(state, prior).mean()
            kl_loss = kl_loss + torch.clamp_min(step_kl, free_nats)

        observation_loss = observation_loss / sequence_length
        reward_loss = reward_loss / sequence_length
        done_loss = done_loss / sequence_length
        kl_loss = kl_loss / sequence_length
        total = (
            observation_loss
            + reward_weight * reward_loss
            + done_weight * done_loss
            + kl_weight * kl_loss
        )
        return {
            "loss": total,
            "observation": observation_loss,
            "reward": reward_loss,
            "done": done_loss,
            "kl": kl_loss,
        }

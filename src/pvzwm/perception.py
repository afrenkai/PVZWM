from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pvzwm.device import DeviceInfo
from pvzwm.domain import PLANT_COSTS, ActionCodec, ActionKind, GameConfig
from pvzwm.env import PVZEnv
from pvzwm.models.jepa import ConvEncoder
from pvzwm.rendering import SymbolicRenderer


@dataclass(frozen=True, slots=True)
class ObservationVectorLayout:
    config: GameConfig

    @property
    def cells(self) -> int:
        return self.config.rows * self.config.cols

    @property
    def plant_types(self) -> slice:
        return slice(0, self.cells)

    @property
    def globals(self) -> slice:
        return slice(4 * self.cells, 4 * self.cells + 4)

    @property
    def mowers(self) -> slice:
        start = 4 * self.cells + 4
        return slice(start, start + self.config.rows)

    @property
    def cooldowns(self) -> slice:
        start = 4 * self.cells + 4 + self.config.rows
        return slice(start, start + 3)

    @property
    def size(self) -> int:
        return 4 * self.cells + 4 + self.config.rows + 3


class BoardPerceptionModel(nn.Module):
    """Compact screen-to-structured-state probe trained without external weights."""

    def __init__(self, state_dim: int, channels: int = 3, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = ConvEncoder(channels, latent_dim)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, state_dim),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.encoder(frame)))

    def loss(self, frame: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(self(frame), state)


def legal_mask_from_vector(vector: np.ndarray, config: GameConfig) -> np.ndarray:
    if vector.ndim != 1:
        raise ValueError(f"Expected a one-dimensional vector, got {vector.shape}")
    return legal_masks_from_vectors(vector[None, :], config)[0]


def legal_masks_from_vectors(vectors: np.ndarray, config: GameConfig) -> np.ndarray:
    """Reconstruct action legality for a batch of normalized state vectors."""
    layout = ObservationVectorLayout(config)
    if vectors.ndim != 2 or vectors.shape[1] != layout.size:
        raise ValueError(
            f"Expected vector batch shape (batch, {layout.size}), got {vectors.shape}"
        )
    codec = ActionCodec(config)
    plant_values = vectors[:, layout.plant_types]
    occupied = np.rint(np.clip(plant_values, 0.0, 1.0) * 3.0).astype(np.int8) != 0
    sun = np.clip(vectors[:, layout.globals.start], 0.0, 1.0) * 1000.0
    cooldowns = np.clip(vectors[:, layout.cooldowns], 0.0, 1.0)
    cooldown_index = {
        ActionKind.PLANT_SUNFLOWER: 0,
        ActionKind.PLANT_PEASHOOTER: 1,
        ActionKind.PLANT_WALLNUT: 2,
    }
    mask = np.zeros((len(vectors), codec.size), dtype=np.bool_)
    mask[:, 0] = True
    for kind_index, kind in enumerate(codec.place_kinds):
        action_slice = slice(
            1 + kind_index * codec.cells,
            1 + (kind_index + 1) * codec.cells,
        )
        if kind == ActionKind.SHOVEL:
            mask[:, action_slice] = occupied
        else:
            available = (
                (sun >= PLANT_COSTS[kind])
                & (cooldowns[:, cooldown_index[kind]] <= 0.05)
            )
            mask[:, action_slice] = (~occupied) & available[:, None]
    return mask


class SyntheticPerceptionGenerator:
    def __init__(self, config: GameConfig, seed: int = 0, cell_size: int = 12) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.env = PVZEnv(config, seed=seed)
        self.observation, _ = self.env.reset(seed=seed)
        self.renderer = SymbolicRenderer(config, cell_size)
        self.episode_index = 0
        self.seed = seed

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        frame = self.renderer.render(self.observation)
        vector = self.observation.vector(self.config)
        legal = np.flatnonzero(self.env.legal_action_mask())
        action = int(self.rng.choice(legal))
        next_observation, _, terminated, truncated, _ = self.env.step(action)
        self.observation = next_observation
        if terminated or truncated:
            self.episode_index += 1
            self.observation, _ = self.env.reset(seed=self.seed + self.episode_index)
        return frame, vector

    def batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        items = [self.sample() for _ in range(size)]
        return (
            torch.from_numpy(np.stack([item[0] for item in items])),
            torch.from_numpy(np.stack([item[1] for item in items])),
        )


@dataclass(frozen=True, slots=True)
class PerceptionTrainConfig:
    steps: int = 2_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    latent_dim: int = 256
    cell_size: int = 12


def train_perception(
    game_config: GameConfig,
    device_info: DeviceInfo,
    config: PerceptionTrainConfig | None = None,
    seed: int = 0,
) -> tuple[BoardPerceptionModel, list[float]]:
    config = config or PerceptionTrainConfig()
    torch.manual_seed(seed)
    generator = SyntheticPerceptionGenerator(game_config, seed, config.cell_size)
    example_frame, example_state = generator.batch(1)
    model = BoardPerceptionModel(
        example_state.shape[-1], example_frame.shape[1], config.latent_dim
    ).to(device_info.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    for _ in range(config.steps):
        frame, state = generator.batch(config.batch_size)
        frame = frame.to(device_info.device)
        state = state.to(device_info.device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(frame, state)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, losses

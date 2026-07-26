from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class ActionKind(IntEnum):
    WAIT = 0
    PLANT_SUNFLOWER = 1
    PLANT_PEASHOOTER = 2
    PLANT_WALLNUT = 3
    SHOVEL = 4


class PlantKind(IntEnum):
    EMPTY = 0
    SUNFLOWER = 1
    PEASHOOTER = 2
    WALLNUT = 3


PLANT_COSTS = {
    ActionKind.PLANT_SUNFLOWER: 50.0,
    ActionKind.PLANT_PEASHOOTER: 100.0,
    ActionKind.PLANT_WALLNUT: 50.0,
}

PLANT_RECHARGE_SECONDS = {
    ActionKind.PLANT_SUNFLOWER: 7.5,
    ActionKind.PLANT_PEASHOOTER: 7.5,
    ActionKind.PLANT_WALLNUT: 30.0,
}


@dataclass(frozen=True, slots=True)
class GameConfig:
    rows: int = 5
    cols: int = 9
    tick_seconds: float = 0.5
    max_time: float = 180.0
    starting_sun: float = 50.0
    passive_sun_interval: float = 10.0
    spawn_interval: float = 7.0
    final_wave_time: float = 145.0
    max_zombies: int = 64


@dataclass(frozen=True, slots=True)
class PVZAction:
    kind: ActionKind
    row: int = 0
    col: int = 0


@dataclass(slots=True)
class Observation:
    plants: np.ndarray
    plant_hp: np.ndarray
    zombies: np.ndarray
    zombie_hp: np.ndarray
    sun: float
    time: float
    mowers: np.ndarray
    seed_cooldowns: np.ndarray
    won: bool = False
    lost: bool = False
    metadata: dict[str, float] = field(default_factory=dict)

    def vector(self, config: GameConfig) -> np.ndarray:
        """Return a stable normalized state vector for learning."""
        parts = [
            (self.plants.astype(np.float32) / float(max(PlantKind))).ravel(),
            np.clip(self.plant_hp.astype(np.float32) / 4000.0, 0.0, 1.0).ravel(),
            np.clip(self.zombies.astype(np.float32) / 4.0, 0.0, 1.0).ravel(),
            np.clip(self.zombie_hp.astype(np.float32) / 500.0, 0.0, 1.0).ravel(),
            np.asarray(
                [
                    min(self.sun / 1000.0, 1.0),
                    min(self.time / config.max_time, 1.0),
                    float(self.won),
                    float(self.lost),
                ],
                dtype=np.float32,
            ),
            self.mowers.astype(np.float32),
            self.seed_cooldowns.astype(np.float32),
        ]
        return np.concatenate(parts).astype(np.float32, copy=False)


class ActionCodec:
    """Maps the structured action space to one categorical action."""

    place_kinds = (
        ActionKind.PLANT_SUNFLOWER,
        ActionKind.PLANT_PEASHOOTER,
        ActionKind.PLANT_WALLNUT,
        ActionKind.SHOVEL,
    )

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.cells = config.rows * config.cols

    @property
    def size(self) -> int:
        return 1 + len(self.place_kinds) * self.cells

    def encode(self, action: PVZAction) -> int:
        if action.kind == ActionKind.WAIT:
            return 0
        if action.kind not in self.place_kinds:
            raise ValueError(f"Unsupported action: {action.kind}")
        if not (0 <= action.row < self.config.rows and 0 <= action.col < self.config.cols):
            raise ValueError("Action coordinates are outside the board")
        kind_index = self.place_kinds.index(action.kind)
        return 1 + kind_index * self.cells + action.row * self.config.cols + action.col

    def decode(self, action_id: int) -> PVZAction:
        if action_id == 0:
            return PVZAction(ActionKind.WAIT)
        if not 0 < action_id < self.size:
            raise ValueError(f"Action id {action_id} outside [0, {self.size})")
        offset = action_id - 1
        kind_index, cell = divmod(offset, self.cells)
        row, col = divmod(cell, self.config.cols)
        return PVZAction(self.place_kinds[kind_index], row, col)

    def encode_action(self, kind: ActionKind, row: int = 0, col: int = 0) -> int:
        return self.encode(PVZAction(kind, row, col))

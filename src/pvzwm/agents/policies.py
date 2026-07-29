from __future__ import annotations

from typing import Protocol

import numpy as np

from pvzwm.domain import ActionKind, Observation, PlantKind
from pvzwm.env import PVZEnv


class Policy(Protocol):
    def act(self, observation: Observation, env: PVZEnv) -> int: ...


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Observation, env: PVZEnv) -> int:
        legal = np.flatnonzero(env.legal_action_mask())
        return int(self.rng.choice(legal))


class HeuristicPolicy:
    """Economy-first scripted baseline with lane threat response."""

    def act(self, observation: Observation, env: PVZEnv) -> int:
        codec = env.codec
        sunflower_count = int(np.count_nonzero(observation.plants == PlantKind.SUNFLOWER))
        constraints = getattr(env, "_campaign_constraints", {})
        economy_target = int(constraints.get("economy_target", env.config.rows))
        if sunflower_count < economy_target and observation.sun >= 50:
            sunflower_counts = np.count_nonzero(
                observation.plants == PlantKind.SUNFLOWER, axis=1
            )
            for row in np.argsort(sunflower_counts):
                for col in range(2):
                    action = codec.encode_action(
                        ActionKind.PLANT_SUNFLOWER, int(row), col
                    )
                    if env.legal_action_mask()[action]:
                        return action

        lane_threat = observation.zombies.sum(axis=1)
        for row in np.argsort(-lane_threat):
            if lane_threat[row] <= 0:
                break
            shooters = np.count_nonzero(observation.plants[row] == PlantKind.PEASHOOTER)
            if shooters < 2 and observation.sun >= 100:
                for col in range(2, 6):
                    action = codec.encode_action(ActionKind.PLANT_PEASHOOTER, int(row), col)
                    if env.legal_action_mask()[action]:
                        return action
            if shooters < 2:
                continue
            if observation.sun >= 50:
                for col in range(6, env.config.cols):
                    action = codec.encode_action(ActionKind.PLANT_WALLNUT, int(row), col)
                    if env.legal_action_mask()[action]:
                        return action
        if observation.sun >= 100:
            shooter_counts = np.count_nonzero(
                observation.plants == PlantKind.PEASHOOTER, axis=1
            )
            for row in np.argsort(shooter_counts):
                for col in range(2, min(6, env.config.cols)):
                    action = codec.encode_action(
                        ActionKind.PLANT_PEASHOOTER, int(row), col
                    )
                    if env.legal_action_mask()[action]:
                        return action
        return 0

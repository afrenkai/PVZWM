from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvzwm.domain import (
    PLANT_COSTS,
    PLANT_RECHARGE_SECONDS,
    ActionCodec,
    ActionKind,
    GameConfig,
    Observation,
    PlantKind,
    PVZAction,
)


@dataclass(slots=True)
class Zombie:
    row: int
    x: float
    hp: float
    speed: float = 0.18
    damage_per_second: float = 30.0


class PVZEnv:
    """Small deterministic, seedable PvZ-like environment for research plumbing.

    It is intentionally not a frame-perfect clone. Its contract mirrors the real
    game adapter: observation, legal-action mask, scalar reward, and termination.
    """

    plant_max_hp = {
        PlantKind.SUNFLOWER: 300.0,
        PlantKind.PEASHOOTER: 300.0,
        PlantKind.WALLNUT: 4000.0,
    }

    def __init__(self, config: GameConfig | None = None, seed: int = 0) -> None:
        self.config = config or GameConfig()
        self.codec = ActionCodec(self.config)
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, object]]:
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)
        shape = (self.config.rows, self.config.cols)
        self.plants = np.zeros(shape, dtype=np.int8)
        self.plant_hp = np.zeros(shape, dtype=np.float32)
        self.sunflower_ready_at = np.full(shape, np.inf, dtype=np.float32)
        self.sun = self.config.starting_sun
        self.time = 0.0
        self.next_passive_sun = self.config.passive_sun_interval
        self.next_spawn = self.config.spawn_interval
        self.seed_ready_at = {
            action_kind: 0.0 for action_kind in PLANT_RECHARGE_SECONDS
        }
        self.zombies: list[Zombie] = []
        self.mowers = np.ones(self.config.rows, dtype=np.bool_)
        self.kills = 0
        self.won = False
        self.lost = False
        self.truncated = False
        return self._observation(), {"seed": self._seed}

    def close(self) -> None:
        """Match the external environment lifecycle; no resources are held here."""

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.codec.size, dtype=np.bool_)
        mask[0] = True
        for action_id in range(1, self.codec.size):
            action = self.codec.decode(action_id)
            occupied = self.plants[action.row, action.col] != PlantKind.EMPTY
            if action.kind == ActionKind.SHOVEL:
                mask[action_id] = occupied
            else:
                mask[action_id] = (
                    (not occupied)
                    and self.sun >= PLANT_COSTS[action.kind]
                    and self.time >= self.seed_ready_at[action.kind]
                )
        return mask

    def action_support_mask(self) -> np.ndarray:
        """Actions implemented by this environment, independent of current state."""
        return np.ones(self.codec.size, dtype=np.bool_)

    def step(
        self, action: int | PVZAction
    ) -> tuple[Observation, float, bool, bool, dict[str, object]]:
        if self.won or self.lost or self.truncated:
            raise RuntimeError("step() called on a terminated episode")
        structured = self.codec.decode(action) if isinstance(action, int) else action
        action_id = self.codec.encode(structured)
        if not self.legal_action_mask()[action_id]:
            raise ValueError(f"Illegal action: {structured}")

        reward = self._apply_action(structured)
        reward += self._advance_economy()
        reward += self._advance_combat()
        self.time += self.config.tick_seconds

        if any(zombie.x < -0.4 for zombie in self.zombies):
            self.lost = True
            reward -= 20.0
        if (
            self.time >= self.config.final_wave_time
            and not self.zombies
            and self.next_spawn == float("inf")
        ):
            self.won = True
            reward += 20.0
        if self.time >= self.config.max_time and not (self.won or self.lost):
            self.truncated = True

        reward -= 0.002
        info: dict[str, object] = {
            "kills": self.kills,
            "legal_action_mask": self.legal_action_mask(),
        }
        return self._observation(), float(reward), self.won or self.lost, self.truncated, info

    def _apply_action(self, action: PVZAction) -> float:
        if action.kind == ActionKind.WAIT:
            return 0.0
        if action.kind == ActionKind.SHOVEL:
            self.plants[action.row, action.col] = PlantKind.EMPTY
            self.plant_hp[action.row, action.col] = 0.0
            self.sunflower_ready_at[action.row, action.col] = np.inf
            return -0.02
        plant = {
            ActionKind.PLANT_SUNFLOWER: PlantKind.SUNFLOWER,
            ActionKind.PLANT_PEASHOOTER: PlantKind.PEASHOOTER,
            ActionKind.PLANT_WALLNUT: PlantKind.WALLNUT,
        }[action.kind]
        self.sun -= PLANT_COSTS[action.kind]
        self.plants[action.row, action.col] = plant
        self.plant_hp[action.row, action.col] = self.plant_max_hp[plant]
        self.seed_ready_at[action.kind] = self.time + PLANT_RECHARGE_SECONDS[action.kind]
        if plant == PlantKind.SUNFLOWER:
            self.sunflower_ready_at[action.row, action.col] = self.time + 7.0
        return 0.01

    def _advance_economy(self) -> float:
        dt = self.config.tick_seconds
        reward = 0.0
        if self.time + dt >= self.next_passive_sun:
            self.sun += 25.0
            self.next_passive_sun += self.config.passive_sun_interval
        producing = np.argwhere(self.time + dt >= self.sunflower_ready_at)
        for row, col in producing:
            if self.plants[row, col] == PlantKind.SUNFLOWER:
                self.sun += 25.0
                self.sunflower_ready_at[row, col] += 24.0
            else:
                self.sunflower_ready_at[row, col] = np.inf

        if self.time + dt >= self.next_spawn and self.time < self.config.final_wave_time:
            row = int(self.rng.integers(0, self.config.rows))
            difficulty = 1.0 + 0.8 * self.time / self.config.max_time
            self.zombies.append(Zombie(row, self.config.cols + 0.2, 120.0 * difficulty))
            jitter = float(self.rng.uniform(0.82, 1.18))
            self.next_spawn += self.config.spawn_interval * jitter
        elif self.time >= self.config.final_wave_time and not self.zombies:
            self.next_spawn = float("inf")
        return reward

    def _advance_combat(self) -> float:
        dt = self.config.tick_seconds
        reward = 0.0
        for row in range(self.config.rows):
            shooters = np.flatnonzero(self.plants[row] == PlantKind.PEASHOOTER)
            targets = [z for z in self.zombies if z.row == row and z.x >= 0]
            if shooters.size and targets:
                target = min(targets, key=lambda zombie: zombie.x)
                target.hp -= (20.0 / 1.425) * shooters.size * dt

        for zombie in self.zombies:
            col = int(np.clip(np.floor(zombie.x), 0, self.config.cols - 1))
            if 0 <= zombie.x < self.config.cols and self.plants[zombie.row, col] != PlantKind.EMPTY:
                self.plant_hp[zombie.row, col] -= zombie.damage_per_second * dt
                if self.plant_hp[zombie.row, col] <= 0:
                    self.plants[zombie.row, col] = PlantKind.EMPTY
                    self.plant_hp[zombie.row, col] = 0.0
                    self.sunflower_ready_at[zombie.row, col] = np.inf
                    reward -= 0.2
            else:
                zombie.x -= zombie.speed * dt

            if zombie.x < 0 and self.mowers[zombie.row]:
                self.mowers[zombie.row] = False
                for other in self.zombies:
                    if other.row == zombie.row:
                        other.hp = 0.0
                reward -= 0.5

        survivors: list[Zombie] = []
        for zombie in self.zombies:
            if zombie.hp <= 0:
                self.kills += 1
                reward += 1.0
            else:
                survivors.append(zombie)
        self.zombies = survivors
        return reward

    def _observation(self) -> Observation:
        zombie_count = np.zeros_like(self.plants, dtype=np.float32)
        zombie_hp = np.zeros_like(self.plant_hp)
        for zombie in self.zombies:
            col = int(np.clip(np.floor(zombie.x), 0, self.config.cols - 1))
            zombie_count[zombie.row, col] += 1.0
            zombie_hp[zombie.row, col] += max(zombie.hp, 0.0)
        return Observation(
            self.plants.copy(),
            self.plant_hp.copy(),
            zombie_count,
            zombie_hp,
            self.sun,
            self.time,
            self.mowers.copy(),
            np.asarray(
                [
                    max(self.seed_ready_at[kind] - self.time, 0.0)
                    / PLANT_RECHARGE_SECONDS[kind]
                    for kind in (
                        ActionKind.PLANT_SUNFLOWER,
                        ActionKind.PLANT_PEASHOOTER,
                        ActionKind.PLANT_WALLNUT,
                    )
                ],
                dtype=np.float32,
            ),
            self.won,
            self.lost,
            {"kills": float(self.kills)},
        )

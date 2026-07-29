from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pvzwm.domain import (
    PLANT_RECHARGE_SECONDS,
    ActionCodec,
    ActionKind,
    GameConfig,
    Observation,
    PVZAction,
    PlantKind,
)
from pvzwm.campaign import supported_campaign_stage_ids
from pvzwm.campaign_proxy import build_campaign_proxy


UPSTREAM_URL = "https://github.com/marblexu/PythonPlantsVsZombies"
PACKAGE_NAME = "plants-vs-zombies"
PACKAGE_VERSION = "0.0.5"


@dataclass(frozen=True, slots=True)
class PythonPortConfig:
    """Runtime settings for the marblexu Pygame port.

    The port advances sprite logic once per rendered frame, so each agent decision
    is divided into short frame updates instead of jumping the game clock forward.
    """

    level: int = 1
    decision_ms: int = 250
    frame_ms: int = 25
    max_steps: int = 1_200
    output_height: int = 120
    output_width: int = 160
    auto_collect_sun: bool = True
    headless: bool = True
    campaign_stage: str | None = None

    def __post_init__(self) -> None:
        if self.campaign_stage is not None and self.campaign_stage not in supported_campaign_stage_ids():
            raise ValueError(
                f"Campaign stage {self.campaign_stage} is not implemented by the port adapter"
            )
        if not 0 <= self.level <= 3:
            raise ValueError(
                "The shared-action RL adapter supports port demos 0 through 3; "
                "upstream demos 4 and 5 require conveyor/bowling actions"
            )
        if self.decision_ms < 1 or self.frame_ms < 1:
            raise ValueError("decision_ms and frame_ms must be positive")
        if self.frame_ms > self.decision_ms:
            raise ValueError("frame_ms cannot exceed decision_ms")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.output_height < 8 or self.output_width < 8:
            raise ValueError("Rendered observations are too small")

    @property
    def scenario_id(self) -> str:
        return self.campaign_stage or f"port-demo-{self.level}"


class PythonPortEnv:
    """Gym-like adapter around ``marblexu/PythonPlantsVsZombies``.

    This is a Python clone, not the original PopCap executable. It deliberately
    exposes the same 181-action/192-value contract as the clean-room environment
    for transfer experiments. The upstream clone has no shovel mechanic, so all
    shovel actions are masked out.
    """

    _card_index = {
        ActionKind.PLANT_SUNFLOWER: 0,
        ActionKind.PLANT_PEASHOOTER: 1,
        ActionKind.PLANT_WALLNUT: 3,
    }
    _plant_kind = {
        "SunFlower": PlantKind.SUNFLOWER,
        "Peashooter": PlantKind.PEASHOOTER,
        "WallNut": PlantKind.WALLNUT,
    }
    _plant_max_units = {
        "SunFlower": 5.0,
        "Peashooter": 5.0,
        "WallNut": 30.0,
    }
    _plant_canonical_hp = {
        "SunFlower": 300.0,
        "Peashooter": 300.0,
        "WallNut": 4_000.0,
    }

    def __init__(self, settings: PythonPortConfig | None = None, seed: int = 0) -> None:
        self.settings = settings or PythonPortConfig()
        self.config = GameConfig(
            tick_seconds=self.settings.decision_ms / 1_000.0,
            max_time=self.settings.max_steps * self.settings.decision_ms / 1_000.0,
        )
        self.codec = ActionCodec(self.config)
        self._load_backend()
        self.reset(seed=seed)

    def _load_backend(self) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        if self.settings.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            import pygame
            from plants_vs_zombies import constants
            from plants_vs_zombies import tool
            from plants_vs_zombies.state import level
        except ImportError as error:
            raise RuntimeError(
                "The Python port is unavailable; run `uv sync` to install "
                f"{PACKAGE_NAME}=={PACKAGE_VERSION}"
            ) from error
        self.pg = pygame
        self.constants = constants
        self.tool = tool
        self.level_module = level

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, object]]:
        self.seed = 0 if seed is None else seed
        random.seed(self.seed)
        self.time_ms = 0
        self.steps = 0
        self.kills = 0
        self._dead_zombie_ids: set[int] = set()
        self.pg.event.clear()
        self.surface = self.tool.SCREEN
        self.level = self.level_module.Level()
        campaign_definition = self._campaign_definition()
        self._campaign_constraints = (
            {} if campaign_definition is None else campaign_definition["known_constraints"]
        )
        if campaign_definition is not None:
            self.level.loadMap = lambda: setattr(
                self.level, "map_data", campaign_definition["engine"]
            )
        game_info = {
            self.constants.CURRENT_TIME: 0,
            self.constants.LEVEL_NUM: self.settings.level,
        }
        self.level.startup(0, game_info)
        available_names = set(self._campaign_constraints.get("available_plants", []))
        available_actions = {
            "Sunflower": ActionKind.PLANT_SUNFLOWER,
            "Peashooter": ActionKind.PLANT_PEASHOOTER,
            "Wall-nut": ActionKind.PLANT_WALLNUT,
        }
        card_indices = (
            [
                self._card_index[action_kind]
                for name, action_kind in available_actions.items()
                if name in available_names
            ]
            if campaign_definition is not None
            else list(self._card_index.values())
        )
        self.level.initPlay(card_indices)
        self.level.menubar.update(0)
        self.level.draw(self.surface)
        return self._observation(), self._info()

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.codec.size, dtype=np.bool_)
        mask[0] = True
        if self.level.done:
            return mask
        for action_id in range(1, self.codec.size):
            action = self.codec.decode(action_id)
            if action.kind == ActionKind.SHOVEL:
                continue
            card = self._card(action.kind)
            active_rows = self._campaign_constraints.get("active_rows")
            active_row = active_rows is None or action.row in active_rows
            empty = self.level.map.isMovable(action.col, action.row)
            mask[action_id] = card is not None and active_row and empty and card.canClick(
                self.level.menubar.sun_value, self.time_ms
            )
        return mask

    def action_support_mask(self) -> np.ndarray:
        """Permanent adapter support, separated from sun/cooldown/occupancy."""
        mask = np.zeros(self.codec.size, dtype=np.bool_)
        mask[0] = True
        active_rows = self._campaign_constraints.get("active_rows")
        for action_id in range(1, self.codec.size):
            action = self.codec.decode(action_id)
            if action.kind == ActionKind.SHOVEL:
                continue
            mask[action_id] = (
                self._card(action.kind) is not None
                and (active_rows is None or action.row in active_rows)
            )
        return mask

    def step(
        self, action: int | PVZAction
    ) -> tuple[Observation, float, bool, bool, dict[str, object]]:
        if self.level.done or self.steps >= self.settings.max_steps:
            raise RuntimeError("step() called on a terminated episode")
        structured = self.codec.decode(action) if isinstance(action, int) else action
        action_id = self.codec.encode(structured)
        if not self.legal_action_mask()[action_id]:
            raise ValueError(f"Illegal action for Python port: {structured}")

        previous_plant_ids = self._plant_ids()
        if structured.kind != ActionKind.WAIT:
            self._place(structured)

        target_time = self.time_ms + self.settings.decision_ms
        while self.time_ms < target_time and not self.level.done:
            self.time_ms = min(self.time_ms + self.settings.frame_ms, target_time)
            required_placements = int(
                self._campaign_constraints.get("spawn_requires_placements", 0)
            )
            if required_placements and len(self._plant_ids()) < required_placements:
                self.level.zombie_start_time = self.time_ms
            self.level.update(self.surface, self.time_ms, None, [False, False])
            if self.settings.auto_collect_sun:
                self._collect_sun()

        self.steps += 1
        current_plant_ids = self._plant_ids()
        lost_plants = len(previous_plant_ids - current_plant_ids)
        new_kills = self._update_kills()
        won = self.level.done and self.level.next == self.constants.GAME_VICTORY
        lost = self.level.done and self.level.next == self.constants.GAME_LOSE
        truncated = self.steps >= self.settings.max_steps and not (won or lost)
        reward = new_kills - 0.2 * lost_plants - 0.002
        if structured.kind != ActionKind.WAIT:
            reward += 0.01
        if won:
            reward += 20.0
        elif lost:
            reward -= 20.0
        observation = self._observation(won=won, lost=lost)
        info = self._info()
        info.update(
            {
                "legal_action_mask": self.legal_action_mask(),
                "new_kills": new_kills,
                "lost_plants": lost_plants,
            }
        )
        return observation, float(reward), won or lost, truncated, info

    def render(self) -> np.ndarray:
        """Return a compact channel-first uint8 frame from the upstream sprites."""
        resized = self.pg.transform.smoothscale(
            self.surface, (self.settings.output_width, self.settings.output_height)
        )
        width_first = self.pg.surfarray.array3d(resized)
        return np.transpose(width_first, (2, 1, 0)).copy()

    def close(self) -> None:
        self.pg.event.clear()

    def _card(self, kind: ActionKind) -> Any | None:
        index = self._card_index[kind]
        return next(
            (card for card in self.level.menubar.card_list if card.name_index == index), None
        )

    def _place(self, action: PVZAction) -> None:
        card = self._card(action.kind)
        if card is None:
            raise ValueError(f"Plant is unavailable in {self.settings.scenario_id}")
        plant_name = self.constants.SUNFLOWER
        if action.kind == ActionKind.PLANT_PEASHOOTER:
            plant_name = self.constants.PEASHOOTER
        elif action.kind == ActionKind.PLANT_WALLNUT:
            plant_name = self.constants.WALLNUT
        self.level.setupMouseImage(plant_name, card)
        position = self.level.map.getMapGridPos(action.col, action.row)
        original_can_seed = self.level.canSeedPlant
        self.level.canSeedPlant = lambda: position
        try:
            self.level.addPlant()
        finally:
            self.level.canSeedPlant = original_can_seed

    def _collect_sun(self) -> None:
        for sun in list(self.level.sun_group):
            self.level.menubar.increaseSunValue(sun.sun_value)
            sun.kill()

    def _plant_ids(self) -> set[int]:
        return {
            id(sprite)
            for group in self.level.plant_groups
            for sprite in group
            if sprite.name in self._plant_kind
        }

    def _update_kills(self) -> int:
        newly_dead = 0
        for group in self.level.zombie_groups:
            for zombie in group:
                identity = id(zombie)
                if zombie.health <= 0 and identity not in self._dead_zombie_ids:
                    self._dead_zombie_ids.add(identity)
                    newly_dead += 1
        self.kills += newly_dead
        return newly_dead

    def _observation(self, *, won: bool = False, lost: bool = False) -> Observation:
        shape = (self.config.rows, self.config.cols)
        plants = np.zeros(shape, dtype=np.int8)
        plant_hp = np.zeros(shape, dtype=np.float32)
        for row, group in enumerate(self.level.plant_groups):
            for sprite in group:
                kind = self._plant_kind.get(sprite.name)
                if kind is None or sprite.health <= 0:
                    continue
                map_x, map_y = self.level.map.getMapIndex(*sprite.getPosition())
                if 0 <= map_y < self.config.rows and 0 <= map_x < self.config.cols:
                    plants[map_y, map_x] = kind
                    fraction = max(sprite.health, 0.0) / self._plant_max_units[sprite.name]
                    plant_hp[map_y, map_x] = fraction * self._plant_canonical_hp[sprite.name]

        zombie_count = np.zeros(shape, dtype=np.float32)
        zombie_hp = np.zeros(shape, dtype=np.float32)
        for row, group in enumerate(self.level.zombie_groups):
            for zombie in group:
                if zombie.health <= 0:
                    continue
                col = int(
                    np.clip(
                        (zombie.rect.centerx - self.constants.MAP_OFFSET_X)
                        // self.constants.GRID_X_SIZE,
                        0,
                        self.config.cols - 1,
                    )
                )
                zombie_count[row, col] += 1.0
                zombie_hp[row, col] += float(zombie.health * 20)

        mower_rows = {car.map_y for car in self.level.cars if not car.dead}
        mowers = np.asarray(
            [row in mower_rows for row in range(self.config.rows)], dtype=np.bool_
        )
        cooldowns = []
        for kind in (
            ActionKind.PLANT_SUNFLOWER,
            ActionKind.PLANT_PEASHOOTER,
            ActionKind.PLANT_WALLNUT,
        ):
            card = self._card(kind)
            if card is None:
                cooldowns.append(1.0)
                continue
            remaining = max(card.frozen_time - (self.time_ms - card.frozen_timer), 0)
            cooldowns.append(remaining / max(card.frozen_time, 1))
        return Observation(
            plants,
            plant_hp,
            zombie_count,
            zombie_hp,
            float(self.level.menubar.sun_value),
            self.time_ms / 1_000.0,
            mowers,
            np.asarray(cooldowns, dtype=np.float32),
            won,
            lost,
            {"kills": float(self.kills), "level": float(self.settings.level)},
        )

    def _info(self) -> dict[str, object]:
        return {
            "source": "python_port",
            "upstream": UPSTREAM_URL,
            "package": PACKAGE_NAME,
            "package_version": PACKAGE_VERSION,
            "level": self.settings.level,
            "scenario_id": self.settings.scenario_id,
            "upstream_level_file": (
                None
                if self.settings.campaign_stage
                else f"level_{self.settings.level}.json"
            ),
            "campaign_stage": self.settings.campaign_stage,
            "step": self.steps,
            "time_ms": self.time_ms,
            "kills": self.kills,
            "sun": self.level.menubar.sun_value,
            "auto_collect_sun": self.settings.auto_collect_sun,
        }

    def _campaign_definition(self) -> dict[str, Any] | None:
        if self.settings.campaign_stage is None:
            return None
        path = Path(__file__).with_name("campaign_data") / f"{self.settings.campaign_stage}.json"
        if path.is_file():
            return json.loads(path.read_text())
        return build_campaign_proxy(self.settings.campaign_stage, self.seed)

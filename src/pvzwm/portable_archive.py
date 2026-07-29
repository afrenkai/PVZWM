from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Callable

import numpy as np

from pvzwm.agents.go_explore import (
    ArchiveCell,
    RandomProjectionCellEncoder,
    TrajectoryArchive,
)
from pvzwm.portable import (
    PortableActionCodec,
    PortableEnv,
    PortableObservation,
    PortableProcessConfig,
    collect_sun_first_mask,
)


def load_portable_archive(
    path: Path,
) -> tuple[TrajectoryArchive, RandomProjectionCellEncoder]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Archive checkpoint must contain an object")
    raw_encoder = payload.get("encoder")
    if not isinstance(raw_encoder, dict):
        raise ValueError("Archive checkpoint has no encoder metadata")
    if raw_encoder.get("kind") != "random_projection_quantized":
        raise ValueError("Archive checkpoint uses an unsupported cell encoder")
    encoder = RandomProjectionCellEncoder(
        int(raw_encoder["state_dim"]),
        int(raw_encoder["projections"]),
        int(raw_encoder["seed"]),
        float(raw_encoder["cell_width"]),
    )
    return TrajectoryArchive.from_payload(payload, encoder), encoder


def resolve_archive_action(
    replay_action: int,
    raw_legal_mask: np.ndarray,
    codec: PortableActionCodec,
    tolerate_collect_timing: bool,
) -> tuple[int | None, bool]:
    """Replay controls, allowing only deterministic collection timing drift."""

    prioritized = collect_sun_first_mask(codec, raw_legal_mask)
    if prioritized[replay_action]:
        return replay_action, False
    if (
        tolerate_collect_timing
        and replay_action == codec.collect_action
        and raw_legal_mask[0]
    ):
        return 0, True
    if (
        tolerate_collect_timing
        and replay_action == 0
        and raw_legal_mask[codec.collect_action]
    ):
        return codec.collect_action, True
    return None, False


def _semantic_branch(action: int, codec: PortableActionCodec) -> tuple[str, int]:
    decoded = codec.decode(action)
    if decoded.kind.name == "PLANT":
        return ("plant", decoded.slot)
    return (decoded.kind.name.lower(), 0)


def _branch_ids(
    cells: list[ArchiveCell], codec: PortableActionCodec, trace_window: int
) -> list[tuple[tuple[str, int], tuple[tuple[str, int], ...]]]:
    branch_ids: list[tuple[tuple[str, int], tuple[tuple[str, int], ...]]] = []
    for cell in cells:
        if not cell.actions:
            initial = ("initial", 0)
            branch_ids.append((initial, (initial,)))
            continue
        latest = _semantic_branch(cell.actions[-1], codec)
        recent = tuple(
            sorted(
                {
                    _semantic_branch(action, codec)
                    for action in cell.actions[-trace_window:]
                }
            )
        )
        branch_ids.append((latest, recent))
    return branch_ids


class PortableArchiveCurriculum:
    """Train from restorable Go-Explore cells with unmodified environment reward.

    Archive prefixes are reset helpers, not demonstrations: replayed actions do
    not enter the policy update. The learned policy controls every action after
    the restored state and receives only the configured authentic bridge reward.
    """

    def __init__(
        self,
        config: PortableProcessConfig,
        archive_path: Path,
        *,
        start_probability: float = 0.8,
        frontier_fraction: float = 0.5,
        minimum_depth: int = 1,
        maximum_depth: int | None = None,
        branch_trace_window: int = 64,
        restore_attempts: int = 8,
        tolerate_collect_timing: bool = True,
        environment_factory: Callable[[PortableProcessConfig], PortableEnv] = PortableEnv,
    ) -> None:
        if not 0.0 <= start_probability <= 1.0:
            raise ValueError("Archive start probability must be in [0, 1]")
        if not 0.0 < frontier_fraction <= 1.0:
            raise ValueError("Archive frontier fraction must be in (0, 1]")
        if minimum_depth < 0:
            raise ValueError("Archive minimum depth cannot be negative")
        if maximum_depth is not None and maximum_depth < minimum_depth:
            raise ValueError("Archive maximum depth cannot precede minimum depth")
        if branch_trace_window < 1 or restore_attempts < 1:
            raise ValueError("Archive trace window and restore attempts must be positive")
        archive, encoder = load_portable_archive(archive_path)
        candidates = [
            cell
            for cell in archive.cells.values()
            if len(cell.actions) >= minimum_depth
            and (maximum_depth is None or len(cell.actions) <= maximum_depth)
        ]
        if not candidates:
            raise ValueError("Archive depth filter selected no cells")
        if encoder.state_dim != 585:
            raise ValueError(
                "Archive curriculum requires the current 585-value portable vector"
            )
        self.config = config
        self.codec = PortableActionCodec()
        self.archive_path = archive_path
        self.start_probability = start_probability
        self.frontier_fraction = frontier_fraction
        self.branch_trace_window = branch_trace_window
        self.restore_attempts = restore_attempts
        self.tolerate_collect_timing = tolerate_collect_timing
        self.encoder = encoder
        self.archive = TrajectoryArchive()
        self.archive.cells = {cell.key: cell for cell in candidates}
        self._environment = environment_factory(config)
        self._archive_started = False
        self._archive_depth = 0
        self._archive_key = ""

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[PortableObservation, dict[str, object]]:
        self._archive_started = False
        self._archive_depth = 0
        self._archive_key = ""
        generator = np.random.default_rng(0 if seed is None else seed)
        if generator.random() >= self.start_probability:
            observation, info = self._environment.reset(seed=seed)
            return observation, self._archive_info(info, attempts=0, fallback=False)

        cells = list(self.archive.cells.values())
        branch_ids = _branch_ids(cells, self.codec, self.branch_trace_window)
        for attempt in range(1, self.restore_attempts + 1):
            start = self.archive.select(
                generator,
                strategy="frontier-branch",
                frontier_fraction=self.frontier_fraction,
                branch_ids=branch_ids,
            )
            observation, _ = self._environment.reset(seed=start.seed)
            approximate = False
            restored = True
            for replay_action in start.actions:
                raw_legal_mask = self._environment.legal_action_mask()
                executed_action, substituted = resolve_archive_action(
                    replay_action,
                    raw_legal_mask,
                    self.codec,
                    self.tolerate_collect_timing,
                )
                if executed_action is None:
                    restored = False
                    break
                observation, _, terminated, truncated, _ = self._environment.step(
                    executed_action
                )
                approximate = approximate or substituted
                if terminated or truncated:
                    restored = False
                    break
            if restored:
                key = self.encoder.encode(observation.vector(self.config))
                if not approximate and key != start.key:
                    restored = False
            if restored:
                self._archive_started = True
                self._archive_depth = len(start.actions)
                self._archive_key = start.key.hex()
                info = {
                    "source": "pvz_portable",
                    "level": observation.level_id,
                    "legal_action_mask": self._environment.legal_action_mask(),
                    "mowers_ready": observation.mowers_ready,
                    "mowers_triggered": observation.mowers_triggered,
                    "mowers_disabled": self.config.disable_mowers,
                }
                return observation, self._archive_info(
                    info, attempts=attempt, fallback=False
                )

        observation, info = self._environment.reset(seed=seed)
        return observation, self._archive_info(
            info, attempts=self.restore_attempts, fallback=True
        )

    def step(
        self, action: int
    ) -> tuple[PortableObservation, float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self._environment.step(action)
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._archive_info(info, attempts=0, fallback=False),
        )

    def legal_action_mask(self) -> np.ndarray:
        return self._environment.legal_action_mask()

    def action_support_mask(self) -> np.ndarray:
        return self._environment.action_support_mask()

    def render(self) -> np.ndarray:
        return self._environment.render()

    def close(self) -> None:
        self._environment.close()

    def _archive_info(
        self, info: dict[str, object], *, attempts: int, fallback: bool
    ) -> dict[str, object]:
        if fallback or not self._archive_started:
            self._archive_started = False
            self._archive_depth = 0
            self._archive_key = ""
        return {
            **info,
            "archive_started": self._archive_started,
            "archive_start_depth": self._archive_depth,
            "archive_start_key": self._archive_key,
            "archive_restore_attempts": attempts,
            "archive_restore_fallback": fallback,
            "archive_start_probability": self.start_probability,
        }

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from pvzwm.agents.go_explore import RandomProjectionCellEncoder, TrajectoryArchive
from pvzwm.portable import (
    PortableActionCodec,
    PortableObservation,
    PortableProcessConfig,
)
from pvzwm.portable_archive import PortableArchiveCurriculum


class _FakePortableEnv:
    def __init__(self, config: PortableProcessConfig) -> None:
        self.config = config
        self.codec = PortableActionCodec()
        self.actions: list[int] = []
        self.reset_seeds: list[int | None] = []
        self.closed = False
        self.observation = PortableObservation(
            {
                "area": 3,
                "stage": 1,
                "board_result": 0,
                "mowers_ready": 6,
                "legal_actions": [0],
            }
        )

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[PortableObservation, dict[str, object]]:
        self.actions.clear()
        self.reset_seeds.append(seed)
        return self.observation, {"source": "fake"}

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.codec.size, dtype=np.bool_)
        mask[0] = True
        return mask

    def action_support_mask(self) -> np.ndarray:
        return np.ones(self.codec.size, dtype=np.bool_)

    def step(
        self, action: int
    ) -> tuple[PortableObservation, float, bool, bool, dict[str, object]]:
        self.actions.append(action)
        return self.observation, 0.0, False, False, {"source": "fake"}

    def render(self) -> np.ndarray:
        return np.zeros((3, 1, 1), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def _write_archive(path: Path, actions: list[int]) -> None:
    environment = _FakePortableEnv(
        PortableProcessConfig(Path("bridge"), Path("resources"))
    )
    encoder = RandomProjectionCellEncoder(585, 20, 7, 0.05)
    key = encoder.encode(environment.observation.vector())
    archive = TrajectoryArchive()
    archive.add(key, 123, actions, 0.0)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(archive.payload(encoder), handle)


def test_archive_curriculum_restores_prefix_before_policy_control(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.json.gz"
    _write_archive(archive_path, [0, 0])
    fake: _FakePortableEnv | None = None

    def factory(config: PortableProcessConfig) -> _FakePortableEnv:
        nonlocal fake
        fake = _FakePortableEnv(config)
        return fake

    environment = PortableArchiveCurriculum(
        PortableProcessConfig(Path("bridge"), Path("resources")),
        archive_path,
        start_probability=1.0,
        environment_factory=factory,
    )
    observation, info = environment.reset(seed=10)
    assert fake is not None
    assert observation.level_id == "3-1"
    assert fake.reset_seeds == [123]
    assert fake.actions == [0, 0]
    assert info["archive_started"] is True
    assert info["archive_start_depth"] == 2

    _, reward, terminated, truncated, step_info = environment.step(0)
    assert reward == 0.0
    assert not terminated and not truncated
    assert fake.actions == [0, 0, 0]
    assert step_info["archive_started"] is True


def test_archive_curriculum_keeps_clean_resets_in_training_mix(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive.json.gz"
    _write_archive(archive_path, [0])
    fake: _FakePortableEnv | None = None

    def factory(config: PortableProcessConfig) -> _FakePortableEnv:
        nonlocal fake
        fake = _FakePortableEnv(config)
        return fake

    environment = PortableArchiveCurriculum(
        PortableProcessConfig(Path("bridge"), Path("resources")),
        archive_path,
        start_probability=0.0,
        environment_factory=factory,
    )
    _, info = environment.reset(seed=55)
    assert fake is not None
    assert fake.reset_seeds == [55]
    assert fake.actions == []
    assert info["archive_started"] is False
    assert info["archive_start_depth"] == 0

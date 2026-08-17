import importlib.util
import sys
from pathlib import Path

import numpy as np

from pvzwm.agents.go_explore import ArchiveCell
from pvzwm.portable import PortableActionCodec, PortableProcessConfig

_SPEC = importlib.util.spec_from_file_location(
    "collect_verified_archive_replay",
    Path("scripts/collect_verified_archive_replay.py"),
)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


class _Observation:
    def __init__(self, step: int) -> None:
        self.step = step

    def vector(self, _config: PortableProcessConfig) -> np.ndarray:
        return np.full(585, self.step, dtype=np.float32)


class _Environment:
    last_instance: "_Environment | None" = None

    def __init__(self, _config: PortableProcessConfig) -> None:
        self.codec = PortableActionCodec()
        self.step_count = 0
        self.closed = False
        _Environment.last_instance = self

    def reset(self, *, seed: int) -> tuple[_Observation, dict[str, object]]:
        assert seed == 7
        return _Observation(0), {}

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.codec.size, dtype=np.bool_)
        mask[[0, 1]] = True
        return mask

    def step(
        self, action: int
    ) -> tuple[_Observation, float, bool, bool, dict[str, object]]:
        self.step_count += 1
        return _Observation(self.step_count), float(action), False, False, {}

    def close(self) -> None:
        self.closed = True


def test_replay_prefix_records_vector_sequence_and_marks_prefix_boundary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_SCRIPT, "PortableEnv", _Environment)
    config = PortableProcessConfig(
        binary=Path("bridge"),
        resources=Path("resources"),
        frame_width=0,
        frame_height=0,
    )
    cell = ArchiveCell(b"cell", 7, (0, 1), 0.0)

    result = _SCRIPT.replay_prefix(_SCRIPT.PrefixReplayTask(config, cell))

    assert result.valid
    assert result.depth == 2
    assert len(result.transitions) == 2
    np.testing.assert_array_equal(
        result.transitions[0].next_observation,
        result.transitions[1].observation,
    )
    assert not result.transitions[0].truncated
    assert result.transitions[1].truncated
    assert _Environment.last_instance is not None
    assert _Environment.last_instance.closed

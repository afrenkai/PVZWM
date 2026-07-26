import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

from pvzwm.agents.goal_iql import GoalEpisode
from pvzwm.device import DeviceInfo
from pvzwm.portable import PortableActionCodec, PortableProcessConfig


_SPEC = importlib.util.spec_from_file_location(
    "eval_goal_iql_reachability",
    Path("scripts/eval_goal_iql_reachability.py"),
)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def test_unseen_calibration_source_requires_explicit_opt_in() -> None:
    with np.testing.assert_raises_regex(ValueError, "not included"):
        _SCRIPT.resolve_source_split_seed(
            ["verified_archive_prefix"],
            "clean_failure",
            17,
            allow_unseen_source=False,
        )
    assert _SCRIPT.resolve_source_split_seed(
        ["verified_archive_prefix"],
        "clean_failure",
        17,
        allow_unseen_source=True,
    ) == (17, False)


def test_trained_calibration_source_preserves_split_offset() -> None:
    assert _SCRIPT.resolve_source_split_seed(
        ["first", "second"],
        "second",
        17,
        allow_unseen_source=False,
    ) == (18, True)


def test_parse_seed_types_preserves_order() -> None:
    assert _SCRIPT.parse_seed_types("4,7,12,13,3,0") == (4, 7, 12, 13, 3, 0)


class _Observation:
    def __init__(self, step: int) -> None:
        self.step = step
        self.mowers_triggered = 0

    def vector(self, _config: PortableProcessConfig) -> np.ndarray:
        return np.full(585, self.step, dtype=np.float32)


class _Environment:
    def __init__(self, _config: PortableProcessConfig) -> None:
        self.codec = PortableActionCodec()
        self.step_count = 0

    def reset(self, *, seed: int):
        assert seed == 11
        self.step_count = 0
        return _Observation(0), {}

    def legal_action_mask(self) -> np.ndarray:
        legal = np.zeros(self.codec.size, dtype=np.bool_)
        legal[0] = True
        return legal

    def step(self, action: int):
        assert action == 0
        self.step_count += 1
        return _Observation(self.step_count), 0.0, False, False, {}

    def close(self) -> None:
        pass


class _Encoder:
    def encode(self, state: np.ndarray) -> bytes:
        return int(state[0]).to_bytes(2, "little")


class _Actor(torch.nn.Module):
    def forward(self, state_goal: torch.Tensor):
        logits = torch.full((len(state_goal), 650), -20.0)
        logits[:, 0] = 0.0
        return logits, torch.zeros(len(state_goal))


class _Value(torch.nn.Module):
    def forward(self, state: torch.Tensor, goal: torch.Tensor):
        return -(goal[:, 0] - state[:, 0]).abs()


def test_policy_reachability_runs_in_authentic_step_loop(monkeypatch) -> None:
    monkeypatch.setattr(_SCRIPT, "PortableEnv", _Environment)
    states = np.stack(
        [np.full(585, step, dtype=np.float32) for step in range(3)]
    )
    episode = GoalEpisode(
        states,
        np.asarray([0, 0], dtype=np.int64),
        np.ones((2, 650), dtype=np.bool_),
    )
    task = _SCRIPT.ReachabilityTask(
        "episode",
        11,
        episode,
        0,
        2,
        2,
        _Encoder().encode(states[2]),
    )
    config = PortableProcessConfig(
        binary=Path("bridge"),
        resources=Path("resources"),
        frame_width=0,
        frame_height=0,
    )

    result = _SCRIPT._policy_reaches(
        task,
        config,
        _Encoder(),
        _Actor(),
        _Value(),
        DeviceInfo(torch.device("cpu"), "cpu", False),
        restore_attempts=1,
        horizon_multiplier=2.0,
        horizon_slack=0,
        stochastic=False,
        generator=torch.Generator().manual_seed(1),
    )

    assert result["restored"]
    assert result["success"]
    assert result["steps"] == 2
    assert result["predicted_value"] == -2.0

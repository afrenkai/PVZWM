from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from pvzwm.domain import ActionCodec, ActionKind, GameConfig, PVZAction
from pvzwm.env import PVZEnv
from pvzwm.native import (
    ACTION_COUNT,
    OBSERVATION_SIZE,
    NativeBatchEnv,
    build_native_library,
)


@pytest.fixture(scope="module")
def native_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("cc") is None:
        pytest.skip("A C compiler is required for the native simulator test")
    return build_native_library(tmp_path_factory.mktemp("pvzwm-native"), optimize=False)


def test_native_contract_and_legality(native_library: Path) -> None:
    codec = ActionCodec(GameConfig())
    sunflower = codec.encode(PVZAction(ActionKind.PLANT_SUNFLOWER, 0, 0))
    peashooter = codec.encode(PVZAction(ActionKind.PLANT_PEASHOOTER, 0, 1))
    with NativeBatchEnv(2, library=native_library) as env:
        observations, masks = env.reset(np.asarray([3, 7], dtype=np.uint64))
        assert observations.shape == (2, OBSERVATION_SIZE)
        assert masks.shape == (2, ACTION_COUNT)
        assert masks.dtype == np.bool_
        assert masks[:, 0].all()
        assert masks[:, sunflower].all()
        assert not masks[:, peashooter].any()

        transition = env.step(np.asarray([sunflower, 0], dtype=np.int32))
        assert transition.observations[0, 0] == pytest.approx(1.0 / 3.0)
        assert transition.observations[0, 45] == pytest.approx(300.0 / 4000.0)
        assert transition.rewards[0] == pytest.approx(0.008)
        assert not transition.legal_action_masks[0, sunflower]


def test_native_matches_python_before_random_spawns(native_library: Path) -> None:
    config = GameConfig(
        max_time=20.0,
        passive_sun_interval=100.0,
        spawn_interval=100.0,
        final_wave_time=100.0,
    )
    codec = ActionCodec(config)
    actions = [
        codec.encode(PVZAction(ActionKind.PLANT_SUNFLOWER, 0, 0)),
        0,
        0,
        0,
    ]
    python_env = PVZEnv(config, seed=11)
    python_observation, _ = python_env.reset(seed=11)
    with NativeBatchEnv(1, config, library=native_library) as native_env:
        native_observations, native_masks = native_env.reset(11)
        np.testing.assert_allclose(
            native_observations[0], python_observation.vector(config), atol=1e-7
        )
        np.testing.assert_array_equal(native_masks[0], python_env.legal_action_mask())
        for action in actions:
            python_observation, reward, terminal, truncated, info = python_env.step(action)
            native_step = native_env.step(np.asarray([action], dtype=np.int32))
            np.testing.assert_allclose(
                native_step.observations[0],
                python_observation.vector(config),
                rtol=1e-6,
                atol=1e-7,
            )
            assert native_step.rewards[0] == pytest.approx(reward, abs=1e-6)
            assert bool(native_step.terminals[0]) is terminal
            assert bool(native_step.truncations[0]) is truncated
            np.testing.assert_array_equal(
                native_step.legal_action_masks[0], info["legal_action_mask"]
            )


def test_native_rejects_illegal_batch_without_stepping(native_library: Path) -> None:
    codec = ActionCodec(GameConfig())
    peashooter = codec.encode(PVZAction(ActionKind.PLANT_PEASHOOTER, 0, 0))
    with NativeBatchEnv(2, library=native_library) as env:
        before, _ = env.reset(4)
        before = before.copy()
        with pytest.raises(ValueError, match="environment 1"):
            env.step(np.asarray([0, peashooter], dtype=np.int32))
        np.testing.assert_array_equal(env.observations, before)


def test_native_rng_is_reproducible_per_environment(native_library: Path) -> None:
    with NativeBatchEnv(2, library=native_library) as env:
        env.reset(np.asarray([19, 19], dtype=np.uint64))
        for _ in range(100):
            transition = env.step(np.zeros(2, dtype=np.int32))
            np.testing.assert_array_equal(
                transition.observations[0], transition.observations[1]
            )
            assert transition.rewards[0] == transition.rewards[1]
            np.testing.assert_array_equal(
                transition.legal_action_masks[0],
                transition.legal_action_masks[1],
            )

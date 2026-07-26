import json
from pathlib import Path

import numpy as np

from pvzwm.domain import (
    PLANT_COSTS,
    PLANT_RECHARGE_SECONDS,
    ActionCodec,
    ActionKind,
    GameConfig,
    PVZAction,
)
from pvzwm.env import PVZEnv


def test_action_codec_round_trip() -> None:
    codec = ActionCodec(GameConfig())
    for action_id in range(codec.size):
        assert codec.encode(codec.decode(action_id)) == action_id


def test_environment_observation_and_legal_mask() -> None:
    config = GameConfig(max_time=2.0)
    env = PVZEnv(config, seed=4)
    observation, _ = env.reset()
    assert observation.vector(config).dtype == np.float32
    assert env.legal_action_mask().shape == (env.codec.size,)

    action = PVZAction(ActionKind.PLANT_SUNFLOWER, 0, 0)
    action_id = env.codec.encode(action)
    assert env.legal_action_mask()[action_id]
    _, _, _, _, _ = env.step(action_id)
    assert not env.legal_action_mask()[action_id]
    second_sunflower = env.codec.encode(PVZAction(ActionKind.PLANT_SUNFLOWER, 1, 0))
    assert not env.legal_action_mask()[second_sunflower]


def test_time_limit_truncates_without_false_win() -> None:
    config = GameConfig(max_time=1.0, spawn_interval=100.0)
    env = PVZEnv(config, seed=0)
    env.reset()
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
    assert truncated
    assert not env.won


def test_core_simulator_constants_match_sourced_reference() -> None:
    reference_path = Path(__file__).parents[1] / "data" / "reference" / "pvz1_core.json"
    reference = json.loads(reference_path.read_text())
    mapping = {
        "sunflower": ActionKind.PLANT_SUNFLOWER,
        "peashooter": ActionKind.PLANT_PEASHOOTER,
        "wallnut": ActionKind.PLANT_WALLNUT,
    }
    for name, action_kind in mapping.items():
        assert PLANT_COSTS[action_kind] == reference["plants"][name]["sun_cost"]
        assert (
            PLANT_RECHARGE_SECONDS[action_kind]
            == reference["plants"][name]["recharge_seconds"]
        )

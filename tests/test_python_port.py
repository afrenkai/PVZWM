import numpy as np

from pvzwm.domain import ActionKind, PlantKind
from pvzwm.python_port import PythonPortConfig, PythonPortEnv


def test_python_port_exposes_shared_contract() -> None:
    settings = PythonPortConfig(
        decision_ms=100,
        frame_ms=25,
        max_steps=4,
        output_height=60,
        output_width=80,
    )
    env = PythonPortEnv(settings, seed=7)
    try:
        observation, info = env.reset(seed=7)
        assert observation.vector(env.config).shape == (192,)
        assert info["source"] == "python_port"
        assert env.render().shape == (3, 60, 80)
        assert env.render().dtype == np.uint8

        observation, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert not truncated
        action = env.codec.encode_action(ActionKind.PLANT_SUNFLOWER, 0, 0)
        assert env.legal_action_mask()[action]
        observation, _, _, _, _ = env.step(action)
        assert observation.plants[0, 0] == PlantKind.SUNFLOWER
        assert observation.sun == 0
        assert not env.legal_action_mask()[action]
        shovel = env.codec.encode_action(ActionKind.SHOVEL, 0, 0)
        assert not env.legal_action_mask()[shovel]
        assert not env.action_support_mask()[shovel]
        assert env.action_support_mask()[
            env.codec.encode_action(ActionKind.PLANT_SUNFLOWER, 0, 0)
        ]
    finally:
        env.close()


def test_partial_campaign_1_1_masks_to_peashooter_and_middle_lane() -> None:
    settings = PythonPortConfig(
        campaign_stage="1-1",
        decision_ms=100,
        frame_ms=25,
        max_steps=8,
        output_height=60,
        output_width=80,
    )
    env = PythonPortEnv(settings, seed=3)
    try:
        observation, info = env.reset(seed=3)
        assert info["campaign_stage"] == "1-1"
        assert observation.sun == 150
        assert len(env.level.zombie_list) == 5
        env.step(0)
        mask = env.legal_action_mask()
        assert mask[env.codec.encode_action(ActionKind.PLANT_PEASHOOTER, 2, 0)]
        assert not mask[env.codec.encode_action(ActionKind.PLANT_PEASHOOTER, 1, 0)]
        assert not mask[env.codec.encode_action(ActionKind.PLANT_SUNFLOWER, 2, 0)]
    finally:
        env.close()


def test_partial_campaign_1_2_uses_wiki_progression_constraints() -> None:
    settings = PythonPortConfig(
        campaign_stage="1-2",
        decision_ms=100,
        frame_ms=25,
        max_steps=8,
        output_height=60,
        output_width=80,
    )
    env = PythonPortEnv(settings, seed=4)
    try:
        observation, info = env.reset(seed=4)
        assert info["campaign_stage"] == "1-2"
        assert observation.sun == 50
        assert len(env.level.zombie_list) == 12
        assert sum(zombie[1] == "FlagZombie" for zombie in env.level.zombie_list) == 1
        env.step(0)
        mask = env.legal_action_mask()
        assert mask[env.codec.encode_action(ActionKind.PLANT_SUNFLOWER, 1, 0)]
        assert env._card(ActionKind.PLANT_PEASHOOTER) is not None
        assert not mask[env.codec.encode_action(ActionKind.PLANT_SUNFLOWER, 0, 0)]
        assert not mask[env.codec.encode_action(ActionKind.PLANT_WALLNUT, 2, 0)]
    finally:
        env.close()


def test_campaign_1_3_proxy_runs_with_pinned_wave_facts() -> None:
    settings = PythonPortConfig(
        campaign_stage="1-3",
        decision_ms=100,
        frame_ms=25,
        max_steps=8,
        output_height=60,
        output_width=80,
    )
    env = PythonPortEnv(settings, seed=5)
    try:
        observation, info = env.reset(seed=5)
        assert info["campaign_stage"] == "1-3"
        assert observation.sun == 50
        assert len(env.level.zombie_list) >= 8
        assert env._campaign_constraints["adventure1_wave_count"] == 8
        assert env._campaign_constraints["reward_unlock"] == ["Wall-nut"]
        env.step(0)
        mask = env.legal_action_mask()
        assert mask[env.codec.encode_action(ActionKind.PLANT_SUNFLOWER, 0, 0)]
        assert env._card(ActionKind.PLANT_PEASHOOTER) is not None
        assert env._card(ActionKind.PLANT_WALLNUT) is None
    finally:
        env.close()

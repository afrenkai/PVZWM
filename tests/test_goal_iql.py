import numpy as np
import pytest
import torch

import pvzwm.agents.goal_iql as goal_iql
from pvzwm.agents.goal_iql import (
    ClusterHindsightGoalSampler,
    GoalConditionedAutoregressiveActor,
    GoalConditionedSpatialQ,
    GoalEpisode,
    GoalIQLConfig,
    GoalRelabelConfig,
    HindsightGoalSampler,
    split_goal_episode_ids,
)
from pvzwm.device import DeviceInfo


def _episode(offset: float = 0.0) -> GoalEpisode:
    states = np.stack(
        [np.full(585, offset + step, dtype=np.float32) for step in range(4)]
    )
    actions = np.asarray([0, 1, 2], dtype=np.int64)
    legal = np.ones((3, 650), dtype=np.bool_)
    return GoalEpisode(states, actions, legal)


def test_next_state_hindsight_goals_are_immediate_successes() -> None:
    sampler = HindsightGoalSampler(
        [_episode()],
        GoalRelabelConfig(next_fraction=1.0, future_fraction=0.0, random_fraction=0.0),
        seed=3,
    )

    batch = sampler.sample(16)

    assert set(batch.modes) == {"next"}
    torch.testing.assert_close(batch.goal, batch.next_state)
    assert not bool(batch.reward.any())
    assert bool(batch.done.all())
    assert batch.legal_mask.shape == (16, 650)


def test_hindsight_sampler_exposes_future_and_cross_trajectory_modes() -> None:
    future = HindsightGoalSampler(
        [_episode()],
        GoalRelabelConfig(next_fraction=0.0, future_fraction=1.0, random_fraction=0.0),
        seed=4,
    ).sample(32)
    random = HindsightGoalSampler(
        [_episode(), _episode(100.0)],
        GoalRelabelConfig(next_fraction=0.0, future_fraction=0.0, random_fraction=1.0),
        seed=5,
    ).sample(32)

    assert set(future.modes) == {"future"}
    assert set(random.modes) == {"random"}
    assert bool((future.goal[:, 0] >= future.next_state[:, 0]).all())


def test_cluster_hindsight_uses_prototypes_and_equivalent_achievement() -> None:
    episode = _episode()
    prototypes = np.stack(
        [np.full(585, 50.0, dtype=np.float32), np.full(585, 75.0, dtype=np.float32)]
    )
    sampler = ClusterHindsightGoalSampler(
        [episode],
        [np.asarray([0, 1, 1, 0], dtype=np.int64)],
        prototypes,
        GoalRelabelConfig(next_fraction=1.0, future_fraction=0.0, random_fraction=0.0),
        seed=3,
    )

    batch = sampler.sample(32)

    assert set(batch.modes) == {"next"}
    assert not bool(batch.reward.any())
    assert bool(batch.done.all())
    assert set(batch.goal[:, 0].tolist()).issubset({50.0, 75.0})
    assert not torch.equal(batch.goal, batch.next_state)


def test_cluster_hindsight_validates_alignment() -> None:
    with pytest.raises(ValueError, match="align"):
        ClusterHindsightGoalSampler(
            [_episode()],
            [np.asarray([0, 1], dtype=np.int64)],
            np.zeros((2, 585), dtype=np.float32),
        )


def test_goal_relabel_fractions_are_validated() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        GoalRelabelConfig(0.2, 0.2, 0.2).validate()
    with pytest.raises(ValueError, match="negative"):
        GoalRelabelConfig(-0.1, 0.9, 0.2).validate()


def test_goal_episode_split_is_stratified_across_multiple_sources() -> None:
    class _Store:
        def split_episode_ids(self, source, validation_fraction, seed):
            assert validation_fraction == 0.2
            return [f"{source}-train-{seed}"], [f"{source}-validation-{seed}"]

    training, validation = split_goal_episode_ids(
        _Store(), ["archive", "fresh"], validation_fraction=0.2, seed=17
    )
    assert training == ["archive-train-17", "fresh-train-18"]
    assert validation == ["archive-validation-17", "fresh-validation-18"]


def test_goal_conditioned_spatial_models_accept_state_goal_pairs() -> None:
    state_goal = torch.randn(3, 1170)
    actor = GoalConditionedAutoregressiveActor(585, 650, hidden_dim=16)
    critic = GoalConditionedSpatialQ(585, 650, hidden_dim=16)

    actor_logits, actor_value = actor(state_goal)
    q_values, q_value = critic(state_goal)
    loss = actor_logits.mean() + actor_value.mean() + q_values.mean() + q_value.mean()
    loss.backward()

    assert actor_logits.shape == (3, 650)
    assert actor_value.shape == (3,)
    assert q_values.shape == (3, 650)
    assert q_value.shape == (3,)
    assert any(parameter.grad is not None for parameter in actor.parameters())
    assert any(parameter.grad is not None for parameter in critic.parameters())


def test_goal_iql_executes_a_small_training_update(monkeypatch) -> None:
    episodes = [_episode(), _episode(10.0)]

    class _ReplayStore:
        def __init__(self, _database, read_only=False) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def split_episode_ids(self, source, validation_fraction, seed):
            return ["train"], ["validation"]

    monkeypatch.setattr(goal_iql, "ReplayStore", _ReplayStore)
    monkeypatch.setattr(
        goal_iql,
        "load_goal_episodes",
        lambda _database, source=None, episode_ids=None: (episodes, 650),
    )
    config = GoalIQLConfig(
        epochs=1,
        batch_size=2,
        samples_per_epoch=2,
        hidden_dim=16,
    )

    actor, result, auxiliary = goal_iql.train_goal_iql(
        "unused.duckdb",
        DeviceInfo(torch.device("cpu"), "cpu", False),
        config,
        seed=9,
    )

    assert actor.training is False
    assert result.training_episodes == 2
    assert result.validation_episodes == 2
    assert result.training_transitions == 6
    assert result.validation_transitions == 6
    assert np.isfinite(result.q_loss)
    assert np.isfinite(result.actor_loss)
    assert "value_state_dict" in auxiliary

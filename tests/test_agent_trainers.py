import numpy as np
import pytest
import torch

from pvzwm.agents.imagination import FlowPPOTrainConfig, train_flow_ppo
from pvzwm.agents.offline import (
    BehaviorCloneConfig,
    DiscreteIQLConfig,
    branch_balanced_sampling_weights,
    train_behavior_clone,
    train_discrete_iql,
)
from pvzwm.agents.ppo import (
    ActorCritic,
    AutoregressiveSpatialConvActorCritic,
    GroupPolicyBatch,
    SpatialActorCritic,
    SpatialConvActorCritic,
    VDNAutoregressiveSpatialConvActorCritic,
    VDNPPOBatch,
    group_policy_loss,
    vdn_ppo_loss,
)
from pvzwm.agents.trainers import (
    AgentTrainResult,
    DQNTrainConfig,
    DynaDQNTrainConfig,
    GroupPolicyTrainConfig,
    PPOTrainConfig,
    _AdaptiveIntrinsicWeight,
    group_relative_advantages,
    train_dqn,
    train_dyna_dqn,
    train_group_policy,
    train_ppo,
)
from pvzwm.data.store import ReplayStore, Transition
from pvzwm.device import DeviceInfo
from pvzwm.domain import GameConfig
from pvzwm.env import PVZEnv
from pvzwm.models.dynamics import DynamicsEnsemble
from pvzwm.models.flow import RepresentationFlowWorldModel
from pvzwm.portable import PortableActionCodec
from pvzwm.python_port import PythonPortConfig, PythonPortEnv


def test_agent_summary_reports_sequential_curriculum_progress() -> None:
    result = AgentTrainResult(
        episode_returns=[100.0, -100.0, 100.0],
        episode_wins=[True, False, True],
        losses=[0.5],
        episode_levels=[1, 2, 2],
    )
    summary = result.summary()
    assert summary["highest_level_reached"] == 2
    assert summary["last_episode_level"] == 2
    assert summary["distinct_levels_visited"] == 2


def test_dyna_rejects_invalid_informative_replay_fraction(tmp_path) -> None:
    database = tmp_path / "dyna_invalid_fraction.duckdb"
    game = GameConfig(max_time=1.0, final_wave_time=0.5, spawn_interval=100.0)
    env = PVZEnv(game, seed=0)
    observation, _ = env.reset(seed=0)
    with ReplayStore(database) as store:
        with store.episode("symbolic", 0) as writer:
            action = int(np.flatnonzero(env.legal_action_mask())[0])
            next_observation, reward, terminated, truncated, _ = env.step(action)
            writer.append(
                Transition(
                    observation.vector(game),
                    action,
                    reward,
                    next_observation.vector(game),
                    terminated,
                    truncated,
                    env.legal_action_mask(),
                )
            )
    env.close()
    model = DynamicsEnsemble(2, observation.vector(game).size, 181, hidden_dim=8)
    with pytest.raises(ValueError, match="informative_replay_fraction"):
        train_dyna_dqn(
            game,
            DeviceInfo(torch.device("cpu"), "cpu", False),
            str(database),
            model,
            DynaDQNTrainConfig(
                gradient_steps=1,
                batch_size=1,
                hidden_dim=8,
                informative_replay_fraction=1.1,
            ),
        )


def test_dqn_and_ppo_training_loops() -> None:
    device = DeviceInfo(torch.device("cpu"), "cpu", False)
    game = GameConfig(max_time=2.0, final_wave_time=1.0, spawn_interval=100.0)
    dqn, dqn_result = train_dqn(
        game,
        device,
        DQNTrainConfig(
            steps=16,
            batch_size=4,
            replay_capacity=32,
            warmup_steps=4,
            target_interval=4,
            hidden_dim=16,
        ),
    )
    assert dqn_result.episode_returns
    assert dqn_result.losses
    assert next(dqn.online.parameters()).grad is not None

    ppo, ppo_result = train_ppo(
        game,
        device,
        PPOTrainConfig(
            steps=16,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=4,
            hidden_dim=16,
        ),
    )
    assert ppo_result.episode_returns
    assert ppo_result.losses
    assert next(ppo.parameters()).grad is not None


def test_rnd_ppo_records_intrinsic_state_without_changing_episode_returns() -> None:
    device = DeviceInfo(torch.device("cpu"), "cpu", False)
    game = GameConfig(max_time=2.0, final_wave_time=1.0, spawn_interval=100.0)
    _, result = train_ppo(
        game,
        device,
        PPOTrainConfig(
            steps=16,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=4,
            hidden_dim=16,
            intrinsic_reward_weight=0.1,
            rnd_hidden_dim=8,
            rnd_feature_dim=4,
        ),
        seed=7,
    )
    assert result.episode_returns
    assert result.intrinsic_rewards
    assert result.summary()["mean_intrinsic_reward"] > 0.0
    assert result.auxiliary_checkpoint is not None
    assert result.auxiliary_checkpoint["algorithm"] == "rnd"
    assert "normalizer" in result.auxiliary_checkpoint


def test_ride_ppo_records_controllable_episodic_change() -> None:
    device = DeviceInfo(torch.device("cpu"), "cpu", False)
    game = GameConfig(max_time=2.0, final_wave_time=1.0, spawn_interval=100.0)
    _, result = train_ppo(
        game,
        device,
        PPOTrainConfig(
            steps=16,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=4,
            hidden_dim=16,
            intrinsic_method="ride",
            intrinsic_reward_weight=0.1,
            adaptive_intrinsic_weight=True,
            rnd_hidden_dim=8,
            rnd_feature_dim=4,
        ),
        seed=8,
    )
    assert result.episode_returns
    assert result.intrinsic_rewards
    assert result.intrinsic_weights
    assert result.summary()["mean_intrinsic_reward"] >= 0.0
    assert result.auxiliary_checkpoint is not None
    assert result.auxiliary_checkpoint["algorithm"] == "ride"
    assert result.auxiliary_checkpoint["episodic_count"] is True
    assert result.auxiliary_checkpoint["adaptive_weight"] is not None


def test_adaptive_intrinsic_weight_tracks_extrinsic_constraint() -> None:
    controller = _AdaptiveIntrinsicWeight(0.1, lagrange_rate=2.0, smoothing=1.0)
    assert controller.update([-50.0]) == pytest.approx(0.1)
    distracted = controller.update([-100.0])
    assert 0.0 < distracted < 0.1
    assert controller.update([0.0]) > distracted

    restored = _AdaptiveIntrinsicWeight.from_checkpoint(
        controller.checkpoint_dict(),
        base_weight=0.1,
        lagrange_rate=2.0,
        smoothing=1.0,
    )
    assert restored.weight == pytest.approx(controller.weight)
    assert restored.reference == pytest.approx(controller.reference)


def test_group_relative_advantages_are_leave_one_out_and_centered() -> None:
    advantages = group_relative_advantages([1.0, 2.0, 3.0])
    assert advantages.sum() == pytest.approx(0.0)
    assert advantages[0] < 0.0
    assert advantages[2] > 0.0


@pytest.mark.parametrize("algorithm", ["reinforce", "grpo"])
def test_critic_free_group_policy_algorithms_train(algorithm: str) -> None:
    device = DeviceInfo(torch.device("cpu"), "cpu", False)
    game = GameConfig(max_time=0.4, final_wave_time=0.2, spawn_interval=100.0)
    model, result = train_group_policy(
        game,
        device,
        GroupPolicyTrainConfig(
            algorithm=algorithm,
            steps=4,
            group_size=2,
            update_epochs=1,
            minibatch_size=8,
            hidden_dim=8,
        ),
        seed=11,
    )
    assert len(result.episode_returns) >= 2
    assert len(result.episode_returns) % 2 == 0
    assert result.losses
    assert model.policy.weight.grad is not None
    assert model.value.weight.grad is None


def test_parameter_noise_grpo_trains_clean_policy() -> None:
    device = DeviceInfo(torch.device("cpu"), "cpu", False)
    game = GameConfig(max_time=0.4, final_wave_time=0.2, spawn_interval=100.0)
    model, result = train_group_policy(
        game,
        device,
        GroupPolicyTrainConfig(
            algorithm="grpo",
            steps=4,
            group_size=2,
            update_epochs=1,
            minibatch_size=8,
            hidden_dim=8,
            parameter_noise_std=0.01,
            importance_ratio_cap=10.0,
        ),
        seed=13,
    )
    assert result.losses
    assert model.policy.weight.grad is not None


def test_parameter_noise_importance_ratio_is_truncated() -> None:
    model = ActorCritic(2, 2, hidden_dim=4)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
    clean_log_probability = torch.full((2,), -np.log(2.0))
    batch = GroupPolicyBatch(
        state=torch.zeros(2, 2),
        action=torch.zeros(2, dtype=torch.long),
        old_log_probability=clean_log_probability,
        behavior_log_probability=clean_log_probability - np.log(100.0),
        reference_log_probability=clean_log_probability,
        advantage=torch.ones(2),
        trajectory_weight=torch.ones(2),
        legal_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    losses = group_policy_loss(
        model,
        batch,
        algorithm="grpo",
        entropy_weight=0.0,
        reference_kl_weight=0.0,
        importance_ratio_cap=10.0,
    )
    assert losses["policy"].item() == pytest.approx(-10.0)


def test_ppo_rejects_invalid_long_horizon_settings() -> None:
    device = DeviceInfo(torch.device("cpu"), "cpu", False)
    game = GameConfig(max_time=1.0, final_wave_time=0.5, spawn_interval=100.0)
    with pytest.raises(ValueError, match="discount"):
        train_ppo(game, device, PPOTrainConfig(steps=1, discount=1.01))
    with pytest.raises(ValueError, match="gae_lambda"):
        train_ppo(game, device, PPOTrainConfig(steps=1, gae_lambda=-0.1))


def test_behavior_clone_trains_masked_policy_and_initializes_ppo(tmp_path) -> None:
    database = tmp_path / "demonstrations.duckdb"
    state_dim = 6
    action_dim = 5
    with ReplayStore(database) as store:
        with store.episode("planner", 1) as writer:
            for step in range(8):
                state = np.zeros(state_dim, dtype=np.float32)
                state[step % state_dim] = 1.0
                writer.append(
                    Transition(
                        state,
                        0 if step % 2 == 0 else 3,
                        1.0 if step == 7 else 0.0,
                        state,
                        step == 7,
                        legal_mask=np.ones(action_dim, dtype=np.bool_),
                    )
                )
        episode_ids = store.successful_episode_ids("planner")
    clone, clone_result = train_behavior_clone(
        database,
        DeviceInfo(torch.device("cpu"), "cpu", False),
        BehaviorCloneConfig(
            epochs=4,
            batch_size=4,
            learning_rate=1e-2,
            hidden_dim=16,
        ),
        seed=3,
        episode_ids=episode_ids,
    )
    assert clone_result.transitions == 8
    assert clone_result.non_wait_accuracy >= 0.0
    assert clone.policy.out_features == action_dim

    game = GameConfig(max_time=1.0, final_wave_time=0.5, spawn_interval=100.0)
    env = PVZEnv(game, seed=3)
    observation, _ = env.reset(seed=3)
    compatible = ActorCritic(
        observation.vector(game).size, env.codec.size, hidden_dim=16
    )
    env.close()
    ppo, _ = train_ppo(
        game,
        DeviceInfo(torch.device("cpu"), "cpu", False),
        PPOTrainConfig(
            steps=4,
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=2,
            hidden_dim=16,
        ),
        seed=3,
        initial_state_dict=compatible.state_dict(),
    )
    assert next(ppo.parameters()).grad is not None


def test_spatial_actor_critic_matches_portable_action_layout() -> None:
    model = SpatialActorCritic(585, 650, hidden_dim=16)
    logits, values = model(torch.randn(3, 585))
    assert logits.shape == (3, 650)
    assert values.shape == (3,)
    logits.mean().backward()
    assert model.plant_cell.weight.grad is not None


def test_spatial_conv_actor_critic_matches_portable_action_layout() -> None:
    model = SpatialConvActorCritic(585, 650, hidden_dim=16)
    logits, values = model(torch.randn(3, 585))
    assert logits.shape == (3, 650)
    assert values.shape == (3,)
    (logits.mean() + values.mean()).backward()
    assert model.cell_mixer[0].weight.grad is not None


def test_autoregressive_actor_has_normalized_portable_joint_distribution() -> None:
    model = AutoregressiveSpatialConvActorCritic(585, 650, hidden_dim=16)
    joint_log_probabilities, values = model(torch.randn(3, 585))
    assert joint_log_probabilities.shape == (3, 650)
    assert values.shape == (3,)
    torch.testing.assert_close(
        joint_log_probabilities.exp().sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5
    )
    (joint_log_probabilities.mean() + values.mean()).backward()
    assert model.kind_head.weight.grad is not None
    assert model.plant_slot_query.weight.grad is not None
    assert model.cell_mixer[0].weight.grad is not None


def test_vdn_actor_centers_and_masks_conditional_portable_factors() -> None:
    model = VDNAutoregressiveSpatialConvActorCritic(585, 650, hidden_dim=16)
    states = torch.randn(3, 585)
    logits, _ = model(states)
    legal = torch.ones(3, 650, dtype=torch.bool)
    probabilities = model.factor_distribution(logits, legal)
    offset = 0
    for size in model.factor_sizes:
        torch.testing.assert_close(
            probabilities[:, offset : offset + size].sum(dim=-1),
            torch.ones(3),
        )
        offset += size

    centered = model.centered_factor_advantages(states, probabilities)
    offset = 0
    for advantage, size in zip(centered, model.factor_sizes, strict=True):
        expected = (
            probabilities[:, offset : offset + size] * advantage
        ).sum(dim=-1)
        torch.testing.assert_close(expected, torch.zeros(3), atol=1e-5, rtol=1e-5)
        offset += size

    codec = PortableActionCodec()
    actions = torch.tensor([0, codec.plant(2, 3, 4), codec.shovel(1, 5)])
    weights = model.factor_importance_weights(
        states, probabilities, actions, alpha=1.0
    )
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))
    assert weights[0, 1:].count_nonzero() == 0
    assert weights[1].count_nonzero() == 4
    assert weights[2, 1] == 0


def test_vdn_ppo_loss_updates_actor_and_branching_critic() -> None:
    model = VDNAutoregressiveSpatialConvActorCritic(585, 650, hidden_dim=16)
    states = torch.randn(4, 585)
    legal = torch.ones(4, 650, dtype=torch.bool)
    actions = torch.tensor([0, 1, 541, 595])
    with torch.no_grad():
        logits, _ = model(states)
        behavior = model.factor_distribution(logits, legal)
        old_log_probability = model.selected_factor_log_probabilities(
            behavior, actions
        )
    batch = VDNPPOBatch(
        state=states,
        action=actions,
        old_factor_log_probability=old_log_probability,
        factor_advantage=torch.randn(4, 4),
        behavior_factor_probability=behavior,
        value_target=torch.randn(4),
        legal_mask=legal,
    )
    losses = vdn_ppo_loss(model, batch)
    losses["loss"].backward()
    assert model.kind_head.weight.grad is not None
    assert model.kind_advantage.weight.grad is not None
    assert torch.isfinite(losses["loss"])


def test_vdn_ppo_accepts_autoregressive_actor_initialization() -> None:
    class Observation:
        def __init__(self, value: float = 0.0) -> None:
            self.state = np.full(585, value, dtype=np.float32)
            self.won = False

        def vector(self, _config: GameConfig) -> np.ndarray:
            return self.state

    class Environment:
        def __init__(self) -> None:
            self.codec = PortableActionCodec()
            self.steps = 0

        def reset(self, seed: int | None = None):
            self.steps = 0
            return Observation(float(seed or 0) / 100.0), {}

        def legal_action_mask(self) -> np.ndarray:
            legal = np.zeros(self.codec.size, dtype=np.bool_)
            legal[[0, self.codec.plant(0, 0, 0), self.codec.shovel(0, 0)]] = True
            return legal

        def step(self, _action: int):
            self.steps += 1
            terminal = self.steps == 2
            return Observation(self.steps / 10.0), 0.1, terminal, False, {}

        def close(self) -> None:
            pass

    initializer = AutoregressiveSpatialConvActorCritic(585, 650, hidden_dim=8)
    model, result = train_ppo(
        GameConfig(),
        DeviceInfo(torch.device("cpu"), "cpu", False),
        PPOTrainConfig(
            steps=4,
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=2,
            hidden_dim=8,
            policy_architecture="portable-vdn-autoregressive",
        ),
        env_factory=lambda _seed: Environment(),
        initial_state_dict=initializer.state_dict(),
    )
    assert isinstance(model, VDNAutoregressiveSpatialConvActorCritic)
    assert result.losses
    assert model.kind_advantage.weight.grad is not None


def test_autoregressive_behavior_clone_trains_on_portable_vectors(tmp_path) -> None:
    database = tmp_path / "autoregressive_demonstrations.duckdb"
    codec = PortableActionCodec()
    with ReplayStore(database) as store:
        with store.episode("pvz_portable_vector", 9) as writer:
            for step in range(6):
                state = np.zeros(585, dtype=np.float32)
                state[step] = 1.0
                legal = np.zeros(codec.size, dtype=np.bool_)
                action = codec.plant(step % codec.seed_slots, step % 6, step % 9)
                legal[[0, action]] = True
                writer.append(
                    Transition(
                        state,
                        action,
                        0.0,
                        state,
                        step == 5,
                        legal_mask=legal,
                    )
                )
        episodes = store.episode_ids("pvz_portable_vector")
    model, result = train_behavior_clone(
        database,
        DeviceInfo(torch.device("cpu"), "cpu", False),
        BehaviorCloneConfig(
            epochs=1,
            batch_size=3,
            hidden_dim=16,
            sampling="branch-balanced",
            policy_architecture="portable-autoregressive",
        ),
        episode_ids=episodes,
    )
    assert isinstance(model, AutoregressiveSpatialConvActorCritic)
    assert result.transitions == 6


def test_spatial_behavior_clone_trains_on_portable_vectors(tmp_path) -> None:
    database = tmp_path / "spatial_demonstrations.duckdb"
    with ReplayStore(database) as store:
        with store.episode("pvz_portable_vector", 2) as writer:
            for step in range(6):
                state = np.zeros(585, dtype=np.float32)
                state[step] = 1.0
                legal = np.zeros(650, dtype=np.bool_)
                legal[[0, 1 + step]] = True
                writer.append(
                    Transition(
                        state,
                        1 + step,
                        1.0 if step == 5 else 0.0,
                        state,
                        step == 5,
                        legal_mask=legal,
                    )
                )
        episodes = store.successful_episode_ids("pvz_portable_vector")
    model, result = train_behavior_clone(
        database,
        DeviceInfo(torch.device("cpu"), "cpu", False),
        BehaviorCloneConfig(
            epochs=2,
            batch_size=3,
            hidden_dim=16,
            policy_architecture="spatial",
        ),
        episode_ids=episodes,
    )
    assert isinstance(model, SpatialActorCritic)
    assert result.transitions == 6


def test_branch_balanced_bc_gives_equal_mass_to_semantic_branches() -> None:
    codec = PortableActionCodec()
    actions = torch.tensor(
        [
            0,
            codec.plant(0, 0, 0),
            codec.plant(0, 0, 1),
            codec.plant(2, 1, 0),
            codec.collect_action,
        ]
    )

    weights = branch_balanced_sampling_weights(actions, codec.size)

    assert torch.isclose(weights[1] + weights[2], weights[0])
    assert torch.isclose(weights[3], weights[0])
    assert torch.isclose(weights[4], weights[0])
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.double))


def test_discrete_iql_trains_actor_and_checkpoints_critics(tmp_path) -> None:
    database = tmp_path / "iql.duckdb"
    with ReplayStore(database) as store:
        for episode, terminal_reward in enumerate((1.0, -1.0)):
            with store.episode("offline", episode) as writer:
                for step in range(4):
                    state = np.zeros(6, dtype=np.float32)
                    state[step] = 1.0
                    next_state = state.copy()
                    next_state[(step + 1) % 6] = 1.0
                    legal = np.array([True, True, True, False])
                    action = 1 if episode == 0 else 2
                    writer.append(
                        Transition(
                            state,
                            action,
                            terminal_reward if step == 3 else 0.0,
                            next_state,
                            step == 3,
                            legal_mask=legal,
                        )
                    )
    actor, result = train_discrete_iql(
        database,
        DeviceInfo(torch.device("cpu"), "cpu", False),
        DiscreteIQLConfig(
            epochs=2,
            batch_size=4,
            hidden_dim=16,
            success_fraction=0.5,
        ),
        seed=3,
    )
    assert result.episodes == 2
    assert result.successful_episodes == 1
    assert result.transitions == 8
    assert result.auxiliary_checkpoint["algorithm"] == "discrete_iql"
    assert result.summary()["q_loss"] >= 0.0
    assert next(actor.parameters()).grad is not None


def test_discrete_iql_rejects_invalid_expectile() -> None:
    with pytest.raises(ValueError, match="expectile"):
        train_discrete_iql(
            "missing.duckdb",
            DeviceInfo(torch.device("cpu"), "cpu", False),
            DiscreteIQLConfig(expectile=1.0),
        )


def test_agents_can_train_against_python_port() -> None:
    device = DeviceInfo(torch.device("cpu"), "cpu", False)
    settings = PythonPortConfig(
        decision_ms=100,
        frame_ms=25,
        max_steps=4,
        output_height=24,
        output_width=32,
    )
    game = GameConfig(tick_seconds=0.1, max_time=0.4)
    factory = lambda seed: PythonPortEnv(settings, seed=seed)
    dqn, dqn_result = train_dqn(
        game,
        device,
        DQNTrainConfig(
            steps=8,
            batch_size=2,
            replay_capacity=16,
            warmup_steps=2,
            target_interval=2,
            hidden_dim=8,
        ),
        env_factory=factory,
    )
    assert dqn_result.episode_returns
    assert dqn_result.losses
    assert next(dqn.online.parameters()).grad is not None

    ppo, ppo_result = train_ppo(
        game,
        device,
        PPOTrainConfig(
            steps=8,
            rollout_steps=4,
            update_epochs=1,
            minibatch_size=2,
            hidden_dim=8,
        ),
        env_factory=factory,
    )
    assert ppo_result.episode_returns
    assert ppo_result.losses
    assert next(ppo.parameters()).grad is not None


def test_dyna_dqn_learns_from_real_and_world_model_transitions(tmp_path) -> None:
    device = DeviceInfo(torch.device("cpu"), "cpu", False)
    game = GameConfig(max_time=2.0, final_wave_time=1.0, spawn_interval=100.0)
    env = PVZEnv(game, seed=3)
    observation, _ = env.reset(seed=3)
    database = tmp_path / "vector.duckdb"
    with ReplayStore(database) as store:
        with store.episode("test", 3) as writer:
            for _ in range(8):
                state = observation.vector(game)
                legal_mask = env.legal_action_mask()
                action = int(np.flatnonzero(legal_mask)[0])
                next_observation, reward, terminated, truncated, _ = env.step(action)
                writer.append(
                    Transition(
                        state,
                        action,
                        reward,
                        next_observation.vector(game),
                        terminated,
                        truncated,
                        legal_mask,
                    )
                )
                observation = next_observation
                if terminated or truncated:
                    observation, _ = env.reset(seed=4)
    env.close()

    state_dim = observation.vector(game).size
    action_dim = PVZEnv(game).codec.size
    world_model = DynamicsEnsemble(2, state_dim, action_dim, hidden_dim=8)
    agent, result = train_dyna_dqn(
        game,
        device,
        str(database),
        world_model,
        DynaDQNTrainConfig(
            gradient_steps=4,
            batch_size=4,
            hidden_dim=8,
            model_ratio=0.5,
            rollout_horizon=1,
            target_interval=2,
        ),
        seed=5,
    )
    assert len(result.losses) == 4
    assert result.imagined_transitions == 8
    assert next(agent.online.parameters()).grad is not None
    assert all(parameter.grad is None for parameter in world_model.parameters())


def test_dyna_dqn_uses_learned_legality_for_non_clean_room_replay(tmp_path) -> None:
    database = tmp_path / "portable_vectors.duckdb"
    action_dim = 5
    state_dim = 6
    with ReplayStore(database) as store:
        with store.episode("pvz_portable_vector", 1) as writer:
            for step in range(6):
                state = np.full(state_dim, step / 10, dtype=np.float32)
                mask = np.asarray([True, step % 2 == 0, True, False, False])
                writer.append(
                    Transition(
                        state,
                        int(np.flatnonzero(mask)[-1]),
                        0.1,
                        state + 0.1,
                        step == 5,
                        legal_mask=mask,
                    )
                )
    world_model = DynamicsEnsemble(
        2,
        state_dim,
        action_dim,
        hidden_dim=8,
        predict_legality=True,
    )
    agent, result = train_dyna_dqn(
        GameConfig(),
        DeviceInfo(torch.device("cpu"), "cpu", False),
        str(database),
        world_model,
        DynaDQNTrainConfig(
            gradient_steps=3,
            batch_size=4,
            hidden_dim=8,
            model_ratio=0.5,
            rollout_horizon=1,
        ),
        seed=2,
    )
    assert result.imagined_transitions == 6
    assert agent.online.network[-1].out_features == action_dim


def test_flow_ppo_learns_only_from_imagined_pixel_transitions(tmp_path) -> None:
    database = tmp_path / "pixels.duckdb"
    shape = (3, 16, 24)
    with ReplayStore(database) as store:
        with store.episode("pixels", 1) as writer:
            for step in range(4):
                writer.append(
                    Transition(
                        np.full(shape, step * 10, dtype=np.uint8),
                        step % 5,
                        0.1,
                        np.full(shape, (step + 1) * 10, dtype=np.uint8),
                        step == 3,
                        legal_mask=np.ones(5, dtype=np.bool_),
                    )
                )
    world_model = RepresentationFlowWorldModel(
        shape,
        action_dim=5,
        code_dim=4,
        latent_height=2,
        latent_width=2,
        autoencoder_hidden_dim=16,
        model_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    agent, result = train_flow_ppo(
        DeviceInfo(torch.device("cpu"), "cpu", False),
        database,
        world_model,
        FlowPPOTrainConfig(
            updates=2,
            parallel_envs=2,
            imagination_horizon=2,
            update_epochs=1,
            minibatch_size=4,
            hidden_dim=8,
            flow_steps=1,
        ),
        seed=4,
    )
    assert result.imagined_transitions == 8
    assert len(result.losses) == 2
    assert next(agent.parameters()).grad is not None
    assert all(parameter.grad is None for parameter in world_model.parameters())

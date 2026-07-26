from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable, Generic, Protocol, TypeVar

import numpy as np
import torch

from pvzwm.agents.dqn import DQNBatch, MaskedDoubleDQN
from pvzwm.agents.ppo import (
    GroupPolicyBatch,
    PPOBatch,
    PolicyModel,
    RandomNetworkDistillation,
    RideNetwork,
    VDNPPOBatch,
    VDNAutoregressiveSpatialConvActorCritic,
    build_actor_critic,
    POLICY_ARCHITECTURES,
    group_policy_loss,
    ppo_loss,
    vdn_ppo_loss,
)
from pvzwm.device import DeviceInfo
from pvzwm.data.store import ReplayStore
from pvzwm.domain import ActionCodec, GameConfig, Observation
from pvzwm.env import PVZEnv
from pvzwm.models.dynamics import DeterministicDynamics, DynamicsEnsemble
from pvzwm.perception import legal_mask_from_vector, legal_masks_from_vectors
from pvzwm.portable import collect_sun_first_mask


@dataclass(frozen=True, slots=True)
class DQNTrainConfig:
    steps: int = 50_000
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_steps: int = 1_000
    learning_rate: float = 3e-4
    discount: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30_000
    target_interval: int = 500
    hidden_dim: int = 256


@dataclass(frozen=True, slots=True)
class DynaDQNTrainConfig:
    """Offline Dyna-Q with short, uncertainty-aware learned-model rollouts."""

    gradient_steps: int = 50_000
    batch_size: int = 128
    learning_rate: float = 3e-4
    discount: float = 0.99
    target_interval: int = 500
    hidden_dim: int = 256
    model_ratio: float = 0.5
    rollout_horizon: int = 1
    model_epsilon: float = 0.1
    uncertainty_penalty: float = 1.0
    max_uncertainty: float = float("inf")
    done_threshold: float = 0.5
    max_replay_transitions: int | None = None
    informative_replay_fraction: float = 0.0


@dataclass(frozen=True, slots=True)
class PPOTrainConfig:
    steps: int = 50_000
    rollout_steps: int = 512
    update_epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    hidden_dim: int = 256
    intrinsic_method: str = "rnd"
    intrinsic_reward_weight: float = 0.0
    intrinsic_reward_clip: float = 5.0
    adaptive_intrinsic_weight: bool = False
    intrinsic_lagrange_rate: float = 1.0
    intrinsic_performance_smoothing: float = 0.1
    rnd_hidden_dim: int = 128
    rnd_feature_dim: int = 64
    rnd_learning_rate: float = 3e-4
    rnd_update_epochs: int = 1
    policy_architecture: str = "flat"
    vdn_importance_anneal_fraction: float = 1.0


@dataclass(frozen=True, slots=True)
class GroupPolicyTrainConfig:
    algorithm: str = "grpo"
    steps: int = 50_000
    group_size: int = 8
    update_epochs: int = 2
    minibatch_size: int = 512
    learning_rate: float = 1e-4
    discount: float = 1.0
    clip_ratio: float = 0.2
    entropy_weight: float = 0.01
    reference_kl_weight: float = 0.01
    parameter_noise_std: float = 0.0
    importance_ratio_cap: float = 10.0
    win_bonus: float = 0.0
    hidden_dim: int = 256
    policy_architecture: str = "flat"


@dataclass(slots=True)
class AgentTrainResult:
    episode_returns: list[float]
    episode_wins: list[bool]
    losses: list[float]
    episode_levels: list[int] = field(default_factory=list)
    imagined_transitions: int = 0
    rejected_imagined_transitions: int = 0
    intrinsic_rewards: list[float] = field(default_factory=list)
    intrinsic_weights: list[float] = field(default_factory=list)
    auxiliary_checkpoint: dict[str, object] | None = None

    def summary(self) -> dict[str, float]:
        recent = self.episode_returns[-20:]
        recent_wins = self.episode_wins[-20:]
        summary = {
            "episodes": float(len(self.episode_returns)),
            "mean_return_20": float(np.mean(recent)) if recent else 0.0,
            "win_rate_20": float(np.mean(recent_wins)) if recent_wins else 0.0,
            "last_loss": self.losses[-1] if self.losses else 0.0,
        }
        if self.imagined_transitions or self.rejected_imagined_transitions:
            summary["imagined_transitions"] = float(self.imagined_transitions)
            summary["rejected_imagined_transitions"] = float(
                self.rejected_imagined_transitions
            )
        if self.episode_levels:
            summary["highest_level_reached"] = float(max(self.episode_levels))
            summary["last_episode_level"] = float(self.episode_levels[-1])
            summary["distinct_levels_visited"] = float(len(set(self.episode_levels)))
        if self.intrinsic_rewards:
            summary["mean_intrinsic_reward"] = float(np.mean(self.intrinsic_rewards))
            summary["max_intrinsic_reward"] = float(np.max(self.intrinsic_rewards))
        if self.intrinsic_weights:
            summary["mean_intrinsic_weight"] = float(np.mean(self.intrinsic_weights))
            summary["final_intrinsic_weight"] = float(self.intrinsic_weights[-1])
            summary["minimum_intrinsic_weight"] = float(
                np.min(self.intrinsic_weights)
            )
        return summary


@dataclass(slots=True)
class _AdaptiveIntrinsicWeight:
    """CIM-style dual control against a learned extrinsic-return frontier."""

    base_weight: float
    lagrange_rate: float
    smoothing: float
    lagrange: float = 0.0
    performance: float | None = None
    reference: float | None = None

    @property
    def weight(self) -> float:
        return self.base_weight / (1.0 + self.lagrange)

    def update(self, episode_returns: list[float]) -> float:
        if not episode_returns:
            return self.weight
        current = float(np.mean(episode_returns))
        if self.performance is None:
            self.performance = current
        else:
            self.performance += self.smoothing * (current - self.performance)
        if self.reference is None:
            self.reference = self.performance
            return self.weight
        scale = max(1.0, abs(self.reference))
        constraint_gap = (self.reference - self.performance) / scale
        self.lagrange = max(
            0.0, self.lagrange + self.lagrange_rate * constraint_gap
        )
        self.reference = max(self.reference, self.performance)
        return self.weight

    def checkpoint_dict(self) -> dict[str, float | None]:
        return {
            "base_weight": self.base_weight,
            "lagrange_rate": self.lagrange_rate,
            "smoothing": self.smoothing,
            "lagrange": self.lagrange,
            "performance": self.performance,
            "reference": self.reference,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: object,
        *,
        base_weight: float,
        lagrange_rate: float,
        smoothing: float,
    ) -> _AdaptiveIntrinsicWeight:
        controller = cls(base_weight, lagrange_rate, smoothing)
        if not isinstance(checkpoint, dict):
            return controller
        controller.lagrange = max(0.0, float(checkpoint.get("lagrange", 0.0)))
        for name in ("performance", "reference"):
            value = checkpoint.get(name)
            setattr(controller, name, None if value is None else float(value))
        return controller


@dataclass(slots=True)
class _RunningVariance:
    count: float = 1e-4
    mean: float = 0.0
    squared_deviation: float = 1e-4

    @property
    def variance(self) -> float:
        return self.squared_deviation / self.count

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.size == 0:
            return
        batch_count = float(batch.size)
        batch_mean = float(batch.mean())
        batch_deviation = float(((batch - batch_mean) ** 2).sum())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.squared_deviation += (
            batch_deviation + delta * delta * self.count * batch_count / total
        )
        self.mean += delta * batch_count / total
        self.count = total

    def checkpoint_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "squared_deviation": self.squared_deviation,
        }

    @classmethod
    def from_checkpoint(cls, state: object) -> _RunningVariance:
        if not isinstance(state, dict):
            return cls()
        return cls(
            count=float(state.get("count", 1e-4)),
            mean=float(state.get("mean", 0.0)),
            squared_deviation=float(state.get("squared_deviation", 1e-4)),
        )


class _EpisodicStateCounter:
    """Generic random-projection visitation count for episodic RIDE decay."""

    def __init__(
        self,
        state_dim: int,
        seed: int,
        projections: int = 16,
        cell_width: float = 0.05,
    ) -> None:
        generator = np.random.default_rng(seed)
        self.projection = generator.normal(
            0.0, state_dim**-0.5, size=(projections, state_dim)
        ).astype(np.float32)
        self.cell_width = cell_width
        self.counts: dict[bytes, int] = {}

    def weight(self, state: np.ndarray) -> float:
        projected = self.projection @ np.asarray(state, dtype=np.float32)
        key = np.rint(projected / self.cell_width).astype(np.int16).tobytes()
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        return count**-0.5

    def reset(self) -> None:
        self.counts.clear()


def group_relative_advantages(scores: list[float]) -> np.ndarray:
    """Leave-one-out standardized advantages for rollouts sharing one seed."""

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Group-relative advantages require at least two scores")
    scale = float(values.std())
    if scale < 1e-8:
        return np.zeros_like(values)
    other_means = (values.sum() - values) / (values.size - 1)
    return (values - other_means) / (scale + 1e-8)


def discounted_returns(rewards: list[float], discount: float) -> np.ndarray:
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must be in (0, 1]")
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + discount * running
        returns[index] = running
    return returns

T = TypeVar("T")


class _Replay(Generic[T]):
    def __init__(self, capacity: int, seed: int) -> None:
        self.items: deque[T] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def append(self, item: T) -> None:
        self.items.append(item)

    def sample(self, count: int) -> list[T]:
        indices = self.rng.choice(len(self.items), size=count, replace=False)
        return [self.items[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.items)


ReplayItem = tuple[np.ndarray, int, float, np.ndarray, float, np.ndarray]


class VectorEnvironment(Protocol):
    config: GameConfig
    codec: ActionCodec

    def reset(self, seed: int | None = None) -> tuple[Observation, dict[str, object]]: ...

    def legal_action_mask(self) -> np.ndarray: ...

    def step(
        self, action: int
    ) -> tuple[Observation, float, bool, bool, dict[str, object]]: ...

    def close(self) -> None: ...


EnvironmentFactory = Callable[[int], VectorEnvironment]


def _create_environment(
    game_config: GameConfig, seed: int, factory: EnvironmentFactory | None
) -> VectorEnvironment:
    return PVZEnv(game_config, seed=seed) if factory is None else factory(seed)


def train_dqn(
    game_config: GameConfig,
    device_info: DeviceInfo,
    config: DQNTrainConfig | None = None,
    seed: int = 0,
    env_factory: EnvironmentFactory | None = None,
) -> tuple[MaskedDoubleDQN, AgentTrainResult]:
    config = config or DQNTrainConfig()
    torch.manual_seed(seed)
    env = _create_environment(game_config, seed, env_factory)
    observation, _ = env.reset(seed=seed)
    state_dim = observation.vector(game_config).size
    agent = MaskedDoubleDQN(
        state_dim,
        env.codec.size,
        config.hidden_dim,
        config.discount,
        device_info.device,
    )
    optimizer = torch.optim.AdamW(agent.online.parameters(), lr=config.learning_rate)
    replay: _Replay[ReplayItem] = _Replay(config.replay_capacity, seed)
    result = AgentTrainResult([], [], [])
    episode_return = 0.0
    episode_index = 0

    for step in range(config.steps):
        fraction = min(step / max(config.epsilon_decay_steps, 1), 1.0)
        epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)
        state = observation.vector(game_config)
        legal_mask = collect_sun_first_mask(env.codec, env.legal_action_mask())
        action = agent.act(
            torch.from_numpy(state), torch.from_numpy(legal_mask), epsilon=epsilon
        )
        next_observation, reward, terminated, truncated, step_info = env.step(action)
        episode_boundary = terminated or truncated
        next_state = next_observation.vector(game_config)
        next_mask = collect_sun_first_mask(env.codec, env.legal_action_mask())
        replay.append((state, action, reward, next_state, float(terminated), next_mask))
        episode_return += reward
        observation = next_observation

        if len(replay) >= max(config.warmup_steps, config.batch_size):
            items = replay.sample(config.batch_size)
            batch = DQNBatch(
                torch.from_numpy(np.stack([item[0] for item in items])),
                torch.tensor([item[1] for item in items], dtype=torch.long),
                torch.tensor([item[2] for item in items], dtype=torch.float32),
                torch.from_numpy(np.stack([item[3] for item in items])),
                torch.tensor([item[4] for item in items], dtype=torch.float32),
                torch.from_numpy(np.stack([item[5] for item in items])),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = agent.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(), 10.0)
            optimizer.step()
            result.losses.append(float(loss.detach().cpu()))

        if (step + 1) % config.target_interval == 0:
            agent.update_target()
        if episode_boundary:
            result.episode_returns.append(episode_return)
            result.episode_wins.append(next_observation.won)
            curriculum_level = step_info.get("curriculum_episode_level_number")
            if curriculum_level is not None:
                result.episode_levels.append(int(curriculum_level))
            episode_index += 1
            observation, _ = env.reset(seed=seed + episode_index)
            episode_return = 0.0
    env.close()
    return agent, result


def _sample_replay_items(
    items: list[ReplayItem], count: int, rng: np.random.Generator
) -> list[ReplayItem]:
    indices = rng.integers(0, len(items), size=count)
    return [items[int(index)] for index in indices]


def _sample_replay_mix(
    replay: list[ReplayItem],
    informative: list[ReplayItem],
    count: int,
    informative_fraction: float,
    rng: np.random.Generator,
) -> list[ReplayItem]:
    informative_count = (
        min(count, int(round(count * informative_fraction))) if informative else 0
    )
    return _sample_replay_items(informative, informative_count, rng) + _sample_replay_items(
        replay, count - informative_count, rng
    )


def _batch_from_items(items: list[ReplayItem]) -> DQNBatch:
    return DQNBatch(
        torch.from_numpy(np.stack([item[0] for item in items])),
        torch.tensor([item[1] for item in items], dtype=torch.long),
        torch.tensor([item[2] for item in items], dtype=torch.float32),
        torch.from_numpy(np.stack([item[3] for item in items])),
        torch.tensor([item[4] for item in items], dtype=torch.float32),
        torch.from_numpy(np.stack([item[5] for item in items])),
    )


@torch.inference_mode()
def _imagine_dqn_transitions(
    agent: MaskedDoubleDQN,
    world_model: DeterministicDynamics | DynamicsEnsemble,
    replay: list[ReplayItem],
    count: int,
    game_config: GameConfig,
    config: DynaDQNTrainConfig,
    rng: np.random.Generator,
) -> tuple[list[ReplayItem], int]:
    """Branch short policy rollouts from real states and reject uncertain branches."""
    if count == 0:
        return [], 0
    branches = max(1, int(np.ceil(count / config.rollout_horizon)))
    seeds = _sample_replay_items(replay, branches, rng)
    states = np.stack([item[0] for item in seeds]).astype(np.float32, copy=False)
    alive = np.ones(branches, dtype=np.bool_)
    imagined: list[ReplayItem] = []
    rejected = 0

    for _ in range(config.rollout_horizon):
        active_indices = np.flatnonzero(alive)
        if active_indices.size == 0 or len(imagined) >= count:
            break
        active_states = states[active_indices]
        state_tensor = torch.from_numpy(active_states).to(agent.device)
        if world_model.predict_legality:
            legal_masks = world_model.legal_mask(state_tensor).cpu().numpy()
            legal_masks[:, 0] = True
        else:
            legal_masks = legal_masks_from_vectors(active_states, game_config)
        mask_tensor = torch.from_numpy(legal_masks).to(agent.device)
        values = agent.online(state_tensor).masked_fill(~mask_tensor, -torch.inf)
        actions = values.argmax(dim=1).cpu().numpy().astype(np.int64)
        explore = rng.random(len(actions)) < config.model_epsilon
        for index in np.flatnonzero(explore):
            actions[index] = int(rng.choice(np.flatnonzero(legal_masks[index])))

        prediction = world_model(
            state_tensor, torch.from_numpy(actions).to(agent.device)
        )
        next_states = prediction.next_state.clamp(0.0, 1.0).cpu().numpy()
        rewards = prediction.reward.cpu().numpy()
        done_probabilities = prediction.done_logit.sigmoid().cpu().numpy()
        if prediction.variance is None:
            uncertainty = np.zeros(len(actions), dtype=np.float32)
        else:
            uncertainty = (
                prediction.variance.flatten(1).mean(dim=1).clamp_min(0.0).sqrt().cpu().numpy()
            )
        rewards = rewards - config.uncertainty_penalty * uncertainty
        if world_model.predict_legality:
            next_masks = world_model.legal_mask(
                torch.from_numpy(next_states).to(agent.device)
            ).cpu().numpy()
            next_masks[:, 0] = True
        else:
            next_masks = legal_masks_from_vectors(next_states, game_config)

        for local_index, branch_index in enumerate(active_indices):
            if uncertainty[local_index] > config.max_uncertainty:
                alive[branch_index] = False
                rejected += 1
                continue
            done = bool(done_probabilities[local_index] >= config.done_threshold)
            next_mask = next_masks[local_index]
            imagined.append(
                (
                    active_states[local_index],
                    int(actions[local_index]),
                    float(rewards[local_index]),
                    next_states[local_index],
                    float(done),
                    next_mask,
                )
            )
            states[branch_index] = next_states[local_index]
            alive[branch_index] = not done
            if len(imagined) >= count:
                break
    return imagined, rejected


def train_dyna_dqn(
    game_config: GameConfig,
    device_info: DeviceInfo,
    database: str,
    world_model: DeterministicDynamics | DynamicsEnsemble,
    config: DynaDQNTrainConfig | None = None,
    seed: int = 0,
    episode_ids: list[str] | None = None,
) -> tuple[MaskedDoubleDQN, AgentTrainResult]:
    """Learn a masked Q policy from real replay and learned-model experience."""
    config = config or DynaDQNTrainConfig()
    if config.gradient_steps < 1 or config.batch_size < 1:
        raise ValueError("gradient_steps and batch_size must be positive")
    if not 0.0 <= config.model_ratio <= 1.0:
        raise ValueError("model_ratio must be in [0, 1]")
    if config.rollout_horizon < 1:
        raise ValueError("rollout_horizon must be positive")
    if not 0.0 <= config.informative_replay_fraction <= 1.0:
        raise ValueError("informative_replay_fraction must be in [0, 1]")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    replay: list[ReplayItem] = []
    with ReplayStore(database, read_only=True) as store:
        action_dim = store.infer_action_dim(episode_ids)
        for transition in store.transitions(
            limit=config.max_replay_transitions, episode_ids=episode_ids
        ):
            if transition.observation.ndim != 1:
                raise ValueError("Dyna DQN requires vector replay observations")
            terminal = transition.terminated
            episode_boundary = transition.terminated or transition.truncated
            if transition.next_legal_mask is not None:
                next_mask = transition.next_legal_mask
            elif terminal:
                next_mask = np.ones(action_dim, dtype=np.bool_)
            elif not episode_boundary and action_dim == ActionCodec(game_config).size:
                next_mask = legal_mask_from_vector(
                    transition.next_observation, game_config
                )
            else:
                # A collector/process failure can leave one incomplete tail row.
                # It has no valid Bellman target and must not be synthesized.
                continue
            if next_mask.shape != (action_dim,):
                raise ValueError("Replay legal-action masks have inconsistent sizes")
            replay.append(
                (
                    transition.observation.astype(np.float32, copy=False),
                    transition.action,
                    transition.reward,
                    transition.next_observation.astype(np.float32, copy=False),
                    float(terminal),
                    next_mask,
                )
            )
    if not replay:
        raise ValueError("Dyna DQN replay selection is empty")
    informative_replay = [
        item for item in replay if item[1] != 0 or abs(item[2]) > 1e-12
    ]
    state_dim = int(replay[0][0].size)
    if any(item[0].shape != (state_dim,) for item in replay):
        raise ValueError("Dyna DQN replay contains inconsistent state shapes")
    model_state_dim = (
        world_model.members[0].state_dim
        if isinstance(world_model, DynamicsEnsemble)
        else world_model.state_dim
    )
    if model_state_dim != state_dim:
        raise ValueError("World model and replay state dimensions differ")
    model_action_dim = (
        world_model.members[0].action_dim
        if isinstance(world_model, DynamicsEnsemble)
        else world_model.action_dim
    )
    if model_action_dim != action_dim:
        raise ValueError("World model and replay action dimensions differ")
    if action_dim != ActionCodec(game_config).size and not world_model.predict_legality:
        raise ValueError(
            "Non-clean-room Dyna requires a world model trained with legality prediction"
        )

    agent = MaskedDoubleDQN(
        state_dim, action_dim, config.hidden_dim, config.discount, device_info.device
    )
    optimizer = torch.optim.AdamW(agent.online.parameters(), lr=config.learning_rate)
    world_model = world_model.to(device_info.device).eval()
    world_model.requires_grad_(False)
    result = AgentTrainResult([], [], [])

    for step in range(config.gradient_steps):
        requested_imagined = int(round(config.batch_size * config.model_ratio))
        imagined, rejected = _imagine_dqn_transitions(
            agent,
            world_model,
            replay,
            requested_imagined,
            game_config,
            config,
            rng,
        )
        real = _sample_replay_mix(
            replay,
            informative_replay,
            config.batch_size - len(imagined),
            config.informative_replay_fraction,
            rng,
        )
        items = real + imagined
        rng.shuffle(items)
        batch = _batch_from_items(items)
        optimizer.zero_grad(set_to_none=True)
        loss = agent.loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.online.parameters(), 10.0)
        optimizer.step()
        result.losses.append(float(loss.detach().cpu()))
        result.imagined_transitions += len(imagined)
        result.rejected_imagined_transitions += rejected
        if (step + 1) % config.target_interval == 0:
            agent.update_target()
    return agent, result


def train_ppo(
    game_config: GameConfig,
    device_info: DeviceInfo,
    config: PPOTrainConfig | None = None,
    seed: int = 0,
    env_factory: EnvironmentFactory | None = None,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    initial_rnd_checkpoint: dict[str, object] | None = None,
) -> tuple[PolicyModel, AgentTrainResult]:
    config = config or PPOTrainConfig()
    if config.intrinsic_reward_weight < 0.0:
        raise ValueError("intrinsic_reward_weight must be nonnegative")
    if config.intrinsic_method not in {"rnd", "ride"}:
        raise ValueError("intrinsic_method must be rnd or ride")
    if not 0.0 < config.discount <= 1.0:
        raise ValueError("discount must be in (0, 1]")
    if not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if not 0.0 < config.clip_ratio < 1.0:
        raise ValueError("clip_ratio must be in (0, 1)")
    if config.value_weight < 0.0 or config.entropy_weight < 0.0:
        raise ValueError("PPO loss weights must be nonnegative")
    if config.policy_architecture not in POLICY_ARCHITECTURES:
        raise ValueError(f"policy_architecture must be one of {POLICY_ARCHITECTURES}")
    if not 0.0 < config.vdn_importance_anneal_fraction <= 1.0:
        raise ValueError("vdn_importance_anneal_fraction must be in (0, 1]")
    if config.intrinsic_reward_clip <= 0.0:
        raise ValueError("intrinsic_reward_clip must be positive")
    if config.adaptive_intrinsic_weight and config.intrinsic_reward_weight <= 0.0:
        raise ValueError("adaptive intrinsic weighting requires a positive base weight")
    if config.intrinsic_lagrange_rate <= 0.0:
        raise ValueError("intrinsic_lagrange_rate must be positive")
    if not 0.0 < config.intrinsic_performance_smoothing <= 1.0:
        raise ValueError("intrinsic_performance_smoothing must be in (0, 1]")
    if config.rnd_hidden_dim < 1 or config.rnd_feature_dim < 1:
        raise ValueError("RND dimensions must be positive")
    if config.rnd_learning_rate <= 0.0 or config.rnd_update_epochs < 1:
        raise ValueError("RND optimization settings must be positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = _create_environment(game_config, seed, env_factory)
    observation, _ = env.reset(seed=seed)
    state_dim = observation.vector(game_config).size
    model = build_actor_critic(
        state_dim,
        env.codec.size,
        config.hidden_dim,
        config.policy_architecture,
    ).to(device_info.device)
    if initial_state_dict is not None:
        if isinstance(model, VDNAutoregressiveSpatialConvActorCritic):
            incompatible = model.load_state_dict(initial_state_dict, strict=False)
            allowed_missing = {
                name
                for name, _ in model.named_parameters()
                if name.startswith(
                    (
                        "kind_advantage.",
                        "slot_advantage.",
                        "row_advantage.",
                        "column_advantage.",
                    )
                )
            }
            allowed_missing.update({"action_factors", "active_factors"})
            if incompatible.unexpected_keys or not set(
                incompatible.missing_keys
            ).issubset(allowed_missing):
                raise ValueError("VDN initializer is not actor-compatible")
        else:
            model.load_state_dict(initial_state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    rnd: RandomNetworkDistillation | None = None
    ride: RideNetwork | None = None
    intrinsic_optimizer: torch.optim.Optimizer | None = None
    intrinsic_normalizer = _RunningVariance()
    episodic_counter: _EpisodicStateCounter | None = None
    adaptive_intrinsic: _AdaptiveIntrinsicWeight | None = None
    if config.intrinsic_reward_weight > 0.0:
        if config.intrinsic_method == "rnd":
            rnd = RandomNetworkDistillation(
                state_dim,
                config.rnd_hidden_dim,
                config.rnd_feature_dim,
            ).to(device_info.device)
            intrinsic_parameters = rnd.predictor.parameters()
        else:
            ride = RideNetwork(
                state_dim,
                env.codec.size,
                config.rnd_hidden_dim,
                config.rnd_feature_dim,
            ).to(device_info.device)
            intrinsic_parameters = ride.parameters()
            episodic_counter = _EpisodicStateCounter(state_dim, seed)
        if (
            initial_rnd_checkpoint is not None
            and initial_rnd_checkpoint.get("algorithm") == config.intrinsic_method
        ):
            raw_state = initial_rnd_checkpoint.get("state_dict")
            intrinsic_model = rnd if rnd is not None else ride
            if isinstance(raw_state, dict) and intrinsic_model is not None:
                intrinsic_model.load_state_dict(raw_state)
            intrinsic_normalizer = _RunningVariance.from_checkpoint(
                initial_rnd_checkpoint.get("normalizer")
            )
        intrinsic_optimizer = torch.optim.AdamW(
            intrinsic_parameters, lr=config.rnd_learning_rate
        )
        if config.adaptive_intrinsic_weight:
            adaptive_intrinsic = _AdaptiveIntrinsicWeight.from_checkpoint(
                None
                if initial_rnd_checkpoint is None
                else initial_rnd_checkpoint.get("adaptive_weight"),
                base_weight=config.intrinsic_reward_weight,
                lagrange_rate=config.intrinsic_lagrange_rate,
                smoothing=config.intrinsic_performance_smoothing,
            )
    result = AgentTrainResult([], [], [])
    episode_return = 0.0
    episode_index = 0
    total_steps = 0

    while total_steps < config.steps:
        states: list[np.ndarray] = []
        actions: list[int] = []
        log_probabilities: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        policy_rewards: list[float] = []
        terminals: list[float] = []
        episode_boundaries: list[float] = []
        next_values: list[float] = []
        masks: list[np.ndarray] = []
        next_states: list[np.ndarray] = []
        visitation_weights: list[float] = []
        behavior_factor_probabilities: list[np.ndarray] = []
        old_factor_log_probabilities: list[np.ndarray] = []
        factor_importance_weights: list[np.ndarray] = []
        completed_episode_returns: list[float] = []
        rollout_size = min(config.rollout_steps, config.steps - total_steps)

        for _ in range(rollout_size):
            state = observation.vector(game_config)
            legal_mask = collect_sun_first_mask(env.codec, env.legal_action_mask())
            state_tensor = torch.from_numpy(state).to(device_info.device).unsqueeze(0)
            mask_tensor = torch.from_numpy(legal_mask).to(device_info.device).unsqueeze(0)
            with torch.inference_mode():
                logits, value = model(state_tensor)
                logits = logits.masked_fill(~mask_tensor, -torch.inf)
                distribution = torch.distributions.Categorical(logits=logits)
                action_tensor = distribution.sample()
                log_probability = distribution.log_prob(action_tensor)
                if isinstance(model, VDNAutoregressiveSpatialConvActorCritic):
                    factor_probability = model.factor_distribution(logits, mask_tensor)
                    factor_log_probability = model.selected_factor_log_probabilities(
                        factor_probability, action_tensor
                    )
                    anneal_steps = max(
                        1.0, config.steps * config.vdn_importance_anneal_fraction
                    )
                    importance_alpha = min(total_steps / anneal_steps, 1.0)
                    factor_weight = model.factor_importance_weights(
                        state_tensor,
                        factor_probability,
                        action_tensor,
                        importance_alpha,
                    )
            action = int(action_tensor.item())
            next_observation, reward, terminated, truncated, step_info = env.step(action)
            episode_boundary = terminated or truncated
            if terminated:
                next_value = 0.0
            else:
                with torch.inference_mode():
                    next_state = torch.from_numpy(next_observation.vector(game_config)).to(
                        device_info.device
                    )
                    _, next_value_tensor = model(next_state.unsqueeze(0))
                    next_value = float(next_value_tensor.item())

            states.append(state)
            actions.append(action)
            log_probabilities.append(float(log_probability.item()))
            values.append(float(value.item()))
            rewards.append(reward)
            policy_rewards.append(reward)
            terminals.append(float(terminated))
            episode_boundaries.append(float(episode_boundary))
            next_values.append(next_value)
            masks.append(legal_mask)
            next_state_array = next_observation.vector(game_config)
            next_states.append(next_state_array)
            visitation_weights.append(
                1.0
                if episodic_counter is None
                else episodic_counter.weight(next_state_array)
            )
            if isinstance(model, VDNAutoregressiveSpatialConvActorCritic):
                behavior_factor_probabilities.append(
                    factor_probability.squeeze(0).cpu().numpy()
                )
                old_factor_log_probabilities.append(
                    factor_log_probability.squeeze(0).cpu().numpy()
                )
                factor_importance_weights.append(
                    factor_weight.squeeze(0).cpu().numpy()
                )
            episode_return += reward
            observation = next_observation
            total_steps += 1
            if episode_boundary:
                result.episode_returns.append(episode_return)
                completed_episode_returns.append(episode_return)
                result.episode_wins.append(next_observation.won)
                curriculum_level = step_info.get("curriculum_episode_level_number")
                if curriculum_level is not None:
                    result.episode_levels.append(int(curriculum_level))
                episode_index += 1
                observation, _ = env.reset(seed=seed + episode_index)
                episode_return = 0.0
                if episodic_counter is not None:
                    episodic_counter.reset()

        if intrinsic_optimizer is not None and (rnd is not None or ride is not None):
            state_tensor = torch.from_numpy(np.stack(states)).to(device_info.device)
            next_state_tensor = torch.from_numpy(np.stack(next_states)).to(
                device_info.device
            )
            with torch.no_grad():
                if rnd is not None:
                    raw_intrinsic = rnd.novelty(next_state_tensor)
                else:
                    assert ride is not None
                    raw_intrinsic = ride.impact(state_tensor, next_state_tensor)
                    raw_intrinsic = raw_intrinsic * torch.tensor(
                        visitation_weights,
                        dtype=raw_intrinsic.dtype,
                        device=device_info.device,
                    )
                raw_intrinsic = raw_intrinsic.cpu().numpy()
            intrinsic_normalizer.update(raw_intrinsic)
            intrinsic_scale = np.sqrt(
                intrinsic_normalizer.variance + intrinsic_normalizer.mean**2 + 1e-8
            )
            normalized_intrinsic = np.clip(
                raw_intrinsic / intrinsic_scale,
                0.0,
                config.intrinsic_reward_clip,
            ).astype(np.float32)
            effective_intrinsic_weight = config.intrinsic_reward_weight
            if adaptive_intrinsic is not None:
                effective_intrinsic_weight = adaptive_intrinsic.update(
                    completed_episode_returns
                )
                result.intrinsic_weights.append(effective_intrinsic_weight)
            policy_rewards = (
                np.asarray(policy_rewards, dtype=np.float32)
                + effective_intrinsic_weight * normalized_intrinsic
            ).tolist()
            result.intrinsic_rewards.extend(normalized_intrinsic.tolist())
            for _ in range(config.rnd_update_epochs):
                permutation = rng.permutation(rollout_size)
                for start in range(0, rollout_size, config.minibatch_size):
                    indices = torch.from_numpy(
                        permutation[start : start + config.minibatch_size]
                    ).to(device_info.device)
                    intrinsic_optimizer.zero_grad(set_to_none=True)
                    if rnd is not None:
                        intrinsic_loss = rnd.novelty(next_state_tensor[indices]).mean()
                        intrinsic_parameters_for_clip = rnd.predictor.parameters()
                    else:
                        assert ride is not None
                        action_tensor = torch.tensor(
                            actions, dtype=torch.long, device=device_info.device
                        )
                        intrinsic_loss = ride.inverse_loss(
                            state_tensor[indices],
                            next_state_tensor[indices],
                            action_tensor[indices],
                        )
                        intrinsic_parameters_for_clip = ride.parameters()
                    intrinsic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        intrinsic_parameters_for_clip, 10.0
                    )
                    intrinsic_optimizer.step()

        deltas = (
            np.asarray(policy_rewards, dtype=np.float32)
            + config.discount
            * (1.0 - np.asarray(terminals, dtype=np.float32))
            * np.asarray(next_values, dtype=np.float32)
            - np.asarray(values, dtype=np.float32)
        )
        value_targets = deltas + np.asarray(values, dtype=np.float32)
        if isinstance(model, VDNAutoregressiveSpatialConvActorCritic):
            weights = np.stack(factor_importance_weights).astype(np.float32)
            factor_advantages = np.zeros(
                (rollout_size, model.factor_count), dtype=np.float32
            )
            running = np.zeros(model.factor_count, dtype=np.float32)
            for index in range(rollout_size - 1, -1, -1):
                running = (
                    weights[index] * deltas[index]
                    + config.discount
                    * config.gae_lambda
                    * (1.0 - episode_boundaries[index])
                    * running
                )
                factor_advantages[index] = running
            joint_advantage = factor_advantages.sum(axis=-1)
            advantage_mean = float(joint_advantage.mean())
            advantage_scale = float(joint_advantage.std()) + 1e-8
            factor_advantages = (
                factor_advantages - weights * advantage_mean
            ) / advantage_scale
        else:
            advantages = np.zeros(rollout_size, dtype=np.float32)
            gae = 0.0
            for index in range(rollout_size - 1, -1, -1):
                gae = (
                    deltas[index]
                    + config.discount
                    * config.gae_lambda
                    * (1.0 - episode_boundaries[index])
                    * gae
                )
                advantages[index] = gae
            returns = advantages + np.asarray(values, dtype=np.float32)
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        tensors: dict[str, torch.Tensor] = {
            "state": torch.from_numpy(np.stack(states)).to(device_info.device),
            "action": torch.tensor(actions, dtype=torch.long, device=device_info.device),
            "legal_mask": torch.from_numpy(np.stack(masks)).to(device_info.device),
        }
        if isinstance(model, VDNAutoregressiveSpatialConvActorCritic):
            tensors.update(
                {
                    "old_factor_log_probability": torch.from_numpy(
                        np.stack(old_factor_log_probabilities)
                    ).to(device_info.device),
                    "factor_advantage": torch.from_numpy(factor_advantages).to(
                        device_info.device
                    ),
                    "behavior_factor_probability": torch.from_numpy(
                        np.stack(behavior_factor_probabilities)
                    ).to(device_info.device),
                    "value_target": torch.from_numpy(value_targets).to(
                        device_info.device
                    ),
                }
            )
        else:
            tensors.update(
                {
                    "old_log_probability": torch.tensor(
                        log_probabilities,
                        dtype=torch.float32,
                        device=device_info.device,
                    ),
                    "advantage": torch.from_numpy(advantages).to(device_info.device),
                    "returns": torch.from_numpy(returns).to(device_info.device),
                }
            )
        for _ in range(config.update_epochs):
            permutation = rng.permutation(rollout_size)
            for start in range(0, rollout_size, config.minibatch_size):
                indices = torch.from_numpy(permutation[start : start + config.minibatch_size]).to(
                    device_info.device
                )
                optimizer.zero_grad(set_to_none=True)
                if isinstance(model, VDNAutoregressiveSpatialConvActorCritic):
                    vdn_batch = VDNPPOBatch(
                        **{name: value[indices] for name, value in tensors.items()}
                    )
                    losses = vdn_ppo_loss(
                        model,
                        vdn_batch,
                        config.clip_ratio,
                        config.value_weight,
                        config.entropy_weight,
                    )
                else:
                    batch = PPOBatch(
                        **{name: value[indices] for name, value in tensors.items()}
                    )
                    losses = ppo_loss(
                        model,
                        batch,
                        config.clip_ratio,
                        config.value_weight,
                        config.entropy_weight,
                    )
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                result.losses.append(float(losses["loss"].detach().cpu()))
    if rnd is not None or ride is not None:
        intrinsic_model = rnd if rnd is not None else ride
        assert intrinsic_model is not None
        result.auxiliary_checkpoint = {
            "algorithm": config.intrinsic_method,
            "state_dict": intrinsic_model.state_dict(),
            "normalizer": intrinsic_normalizer.checkpoint_dict(),
            "episodic_count": episodic_counter is not None,
            "adaptive_weight": (
                None
                if adaptive_intrinsic is None
                else adaptive_intrinsic.checkpoint_dict()
            ),
        }
    env.close()
    return model, result


def train_group_policy(
    game_config: GameConfig,
    device_info: DeviceInfo,
    config: GroupPolicyTrainConfig | None = None,
    seed: int = 0,
    env_factory: EnvironmentFactory | None = None,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[PolicyModel, AgentTrainResult]:
    """Train critic-free REINFORCE or same-seed grouped GRPO."""

    config = config or GroupPolicyTrainConfig()
    if config.algorithm not in {"reinforce", "grpo"}:
        raise ValueError("Group policy algorithm must be reinforce or grpo")
    if config.steps < 1 or config.update_epochs < 1 or config.minibatch_size < 1:
        raise ValueError("Group policy training counts must be positive")
    if config.algorithm == "grpo" and config.group_size < 2:
        raise ValueError("GRPO requires at least two rollouts per group")
    if config.group_size < 1:
        raise ValueError("Group size must be positive")
    if not 0.0 < config.discount <= 1.0:
        raise ValueError("discount must be in (0, 1]")
    if not 0.0 < config.clip_ratio < 1.0:
        raise ValueError("clip_ratio must be in (0, 1)")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.entropy_weight < 0.0 or config.reference_kl_weight < 0.0:
        raise ValueError("Group policy loss weights must be nonnegative")
    if config.parameter_noise_std < 0.0:
        raise ValueError("parameter_noise_std must be nonnegative")
    if config.algorithm != "grpo" and config.parameter_noise_std > 0.0:
        raise ValueError("parameter-space noise currently requires GRPO")
    if config.importance_ratio_cap < 1.0:
        raise ValueError("importance_ratio_cap must be at least one")
    if config.policy_architecture not in POLICY_ARCHITECTURES:
        raise ValueError(f"policy_architecture must be one of {POLICY_ARCHITECTURES}")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = _create_environment(game_config, seed, env_factory)
    try:
        observation, reset_info = env.reset(seed=seed)
        if "curriculum_final_level" in reset_info:
            raise ValueError(
                "Grouped rollouts require a fixed level; advance the curriculum between runs"
            )
        state_dim = observation.vector(game_config).size
        model = build_actor_critic(
            state_dim,
            env.codec.size,
            config.hidden_dim,
            config.policy_architecture,
        ).to(device_info.device)
        if initial_state_dict is not None:
            model.load_state_dict(initial_state_dict)
        reference = build_actor_critic(
            state_dim,
            env.codec.size,
            config.hidden_dim,
            config.policy_architecture,
        ).to(device_info.device)
        reference.load_state_dict(model.state_dict())
        reference.eval().requires_grad_(False)
        behavior = build_actor_critic(
            state_dim,
            env.codec.size,
            config.hidden_dim,
            config.policy_architecture,
        ).to(device_info.device)
        behavior.eval().requires_grad_(False)
        actor_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("value.")
        ]
        optimizer = torch.optim.AdamW(actor_parameters, lr=config.learning_rate)
        result = AgentTrainResult([], [], [])
        total_steps = 0
        group_index = 0

        while total_steps < config.steps:
            group_seed = seed + group_index
            trajectories: list[dict[str, object]] = []
            scores: list[float] = []
            for _ in range(config.group_size):
                behavior.load_state_dict(model.state_dict())
                if config.parameter_noise_std > 0.0:
                    with torch.no_grad():
                        for name, parameter in behavior.named_parameters():
                            if not name.startswith("value."):
                                parameter.add_(
                                    torch.randn_like(parameter)
                                    * config.parameter_noise_std
                                )
                observation, _ = env.reset(seed=group_seed)
                states: list[np.ndarray] = []
                actions: list[int] = []
                old_log_probabilities: list[float] = []
                behavior_log_probabilities: list[float] = []
                reference_log_probabilities: list[float] = []
                rewards: list[float] = []
                masks: list[np.ndarray] = []
                terminated = False
                truncated = False
                step_info: dict[str, object] = {}
                while not (terminated or truncated):
                    state = observation.vector(game_config)
                    legal_mask = collect_sun_first_mask(
                        env.codec, env.legal_action_mask()
                    )
                    state_tensor = torch.from_numpy(state).to(device_info.device).unsqueeze(0)
                    mask_tensor = torch.from_numpy(legal_mask).to(
                        device_info.device
                    ).unsqueeze(0)
                    with torch.inference_mode():
                        behavior_logits, _ = behavior(state_tensor)
                        behavior_logits = behavior_logits.masked_fill(
                            ~mask_tensor, -torch.inf
                        )
                        distribution = torch.distributions.Categorical(
                            logits=behavior_logits
                        )
                        action_tensor = distribution.sample()
                        behavior_log_probability = distribution.log_prob(action_tensor)
                        clean_logits, _ = model(state_tensor)
                        clean_logits = clean_logits.masked_fill(
                            ~mask_tensor, -torch.inf
                        )
                        clean_distribution = torch.distributions.Categorical(
                            logits=clean_logits
                        )
                        old_log_probability = clean_distribution.log_prob(action_tensor)
                        reference_logits, _ = reference(state_tensor)
                        reference_logits = reference_logits.masked_fill(
                            ~mask_tensor, -torch.inf
                        )
                        reference_distribution = torch.distributions.Categorical(
                            logits=reference_logits
                        )
                        reference_log_probability = reference_distribution.log_prob(
                            action_tensor
                        )
                    action = int(action_tensor.item())
                    next_observation, reward, terminated, truncated, step_info = env.step(
                        action
                    )
                    states.append(state)
                    actions.append(action)
                    old_log_probabilities.append(float(old_log_probability.item()))
                    behavior_log_probabilities.append(
                        float(behavior_log_probability.item())
                    )
                    reference_log_probabilities.append(
                        float(reference_log_probability.item())
                    )
                    rewards.append(float(reward))
                    masks.append(legal_mask)
                    observation = next_observation
                    total_steps += 1
                episode_return = float(sum(rewards))
                won = bool(observation.won)
                score = episode_return + config.win_bonus * float(won)
                scores.append(score)
                trajectories.append(
                    {
                        "states": states,
                        "actions": actions,
                        "old_log_probabilities": old_log_probabilities,
                        "behavior_log_probabilities": behavior_log_probabilities,
                        "reference_log_probabilities": reference_log_probabilities,
                        "rewards": rewards,
                        "masks": masks,
                    }
                )
                result.episode_returns.append(episode_return)
                result.episode_wins.append(won)
                curriculum_level = step_info.get("curriculum_episode_level_number")
                if curriculum_level is not None:
                    result.episode_levels.append(int(curriculum_level))

            if config.algorithm == "grpo":
                episode_advantages = group_relative_advantages(scores)
                trajectory_advantages = [
                    np.full(len(trajectory["rewards"]), episode_advantages[index])
                    for index, trajectory in enumerate(trajectories)
                ]
            else:
                trajectory_advantages = [
                    discounted_returns(trajectory["rewards"], config.discount)
                    for trajectory in trajectories
                ]
                concatenated = np.concatenate(trajectory_advantages)
                mean = float(concatenated.mean())
                standard_deviation = float(concatenated.std())
                trajectory_advantages = [
                    (advantages - mean) / (standard_deviation + 1e-8)
                    for advantages in trajectory_advantages
                ]

            states = np.concatenate(
                [np.stack(trajectory["states"]) for trajectory in trajectories], axis=0
            )
            actions = np.concatenate(
                [np.asarray(trajectory["actions"], dtype=np.int64) for trajectory in trajectories]
            )
            old_log_probabilities = np.concatenate(
                [
                    np.asarray(trajectory["old_log_probabilities"], dtype=np.float32)
                    for trajectory in trajectories
                ]
            )
            reference_log_probabilities = np.concatenate(
                [
                    np.asarray(
                        trajectory["reference_log_probabilities"], dtype=np.float32
                    )
                    for trajectory in trajectories
                ]
            )
            behavior_log_probabilities = np.concatenate(
                [
                    np.asarray(
                        trajectory["behavior_log_probabilities"], dtype=np.float32
                    )
                    for trajectory in trajectories
                ]
            )
            advantages = np.concatenate(trajectory_advantages).astype(np.float32)
            masks = np.concatenate(
                [np.stack(trajectory["masks"]) for trajectory in trajectories], axis=0
            )
            trajectory_weights = np.concatenate(
                [
                    np.full(
                        len(trajectory["rewards"]),
                        1.0 / max(1, len(trajectory["rewards"])),
                        dtype=np.float32,
                    )
                    for trajectory in trajectories
                ]
            )
            tensors = {
                "state": torch.from_numpy(states).to(device_info.device),
                "action": torch.from_numpy(actions).to(device_info.device),
                "old_log_probability": torch.from_numpy(old_log_probabilities).to(
                    device_info.device
                ),
                "behavior_log_probability": torch.from_numpy(
                    behavior_log_probabilities
                ).to(device_info.device),
                "reference_log_probability": torch.from_numpy(
                    reference_log_probabilities
                ).to(device_info.device),
                "advantage": torch.from_numpy(advantages).to(device_info.device),
                "trajectory_weight": torch.from_numpy(trajectory_weights).to(
                    device_info.device
                ),
                "legal_mask": torch.from_numpy(masks).to(device_info.device),
            }
            sample_count = len(actions)
            for _ in range(config.update_epochs):
                permutation = rng.permutation(sample_count)
                for start in range(0, sample_count, config.minibatch_size):
                    indices = torch.from_numpy(
                        permutation[start : start + config.minibatch_size]
                    ).to(device_info.device)
                    batch = GroupPolicyBatch(
                        **{name: value[indices] for name, value in tensors.items()}
                    )
                    optimizer.zero_grad(set_to_none=True)
                    losses = group_policy_loss(
                        model,
                        batch,
                        algorithm=config.algorithm,
                        clip_ratio=config.clip_ratio,
                        entropy_weight=config.entropy_weight,
                        reference_kl_weight=config.reference_kl_weight,
                        importance_ratio_cap=config.importance_ratio_cap,
                    )
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(actor_parameters, 10.0)
                    optimizer.step()
                    result.losses.append(float(losses["loss"].detach().cpu()))
            group_index += 1
        return model, result
    finally:
        env.close()


def config_dict(
    config: DQNTrainConfig | DynaDQNTrainConfig | PPOTrainConfig | GroupPolicyTrainConfig,
) -> dict[str, object]:
    return asdict(config)

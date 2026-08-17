from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pvzwm.agents.latent_goals import encode_vector_states, nearest_cluster
from pvzwm.agents.ppo import (
    AutoregressiveSpatialConvActorCritic,
    SpatialConvActorCritic,
)
from pvzwm.data.store import ReplayStore, Transition
from pvzwm.device import DeviceInfo
from pvzwm.models.jepa import ActionConditionedJEPA


@dataclass(frozen=True, slots=True)
class GoalRelabelConfig:
    next_fraction: float = 0.2
    future_fraction: float = 0.6
    random_fraction: float = 0.2

    def validate(self) -> None:
        fractions = (self.next_fraction, self.future_fraction, self.random_fraction)
        if any(value < 0.0 for value in fractions):
            raise ValueError("Goal relabel fractions cannot be negative")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("Goal relabel fractions must sum to one")


@dataclass(frozen=True, slots=True)
class GoalIQLConfig:
    epochs: int = 50
    batch_size: int = 256
    samples_per_epoch: int | None = None
    learning_rate: float = 3e-4
    hidden_dim: int = 256
    discount: float = 0.99
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    max_advantage_weight: float = 100.0
    target_tau: float = 0.005
    validation_fraction: float = 0.2
    split_seed: int = 17
    relabel: GoalRelabelConfig = GoalRelabelConfig()

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Goal IQL epochs and batch size must be positive")
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError("Goal IQL samples_per_epoch must be positive")
        if self.learning_rate <= 0.0 or self.hidden_dim < 1:
            raise ValueError("Goal IQL learning rate and hidden dimension must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("Goal IQL discount must be in [0, 1]")
        if not 0.0 < self.expectile < 1.0:
            raise ValueError("Goal IQL expectile must be in (0, 1)")
        if self.advantage_temperature <= 0.0 or self.max_advantage_weight < 1.0:
            raise ValueError("Goal IQL advantage parameters are invalid")
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError("Goal IQL target_tau must be in (0, 1]")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("Goal IQL validation_fraction must be in [0, 1)")
        self.relabel.validate()


@dataclass(frozen=True, slots=True)
class GoalEpisode:
    states: np.ndarray
    actions: np.ndarray
    legal_masks: np.ndarray

    def __post_init__(self) -> None:
        if self.states.ndim != 2 or len(self.states) != len(self.actions) + 1:
            raise ValueError("A goal episode needs T+1 vector states for T actions")
        if self.actions.ndim != 1:
            raise ValueError("Goal episode actions must be a vector")
        if self.legal_masks.ndim != 2 or len(self.legal_masks) != len(self.actions):
            raise ValueError("Goal episode legal masks must align with actions")


@dataclass(frozen=True, slots=True)
class GoalBatch:
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    goal: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    legal_mask: torch.Tensor
    modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoalIQLResult:
    training_episodes: int
    validation_episodes: int
    training_transitions: int
    validation_transitions: int
    q_loss: float
    value_loss: float
    actor_loss: float
    mean_advantage_weight: float
    next_goal_value: float
    random_goal_value: float


def _episode_from_transitions(transitions: list[Transition]) -> GoalEpisode:
    if not transitions:
        raise ValueError("A goal episode cannot be empty")
    if any(item.observation.ndim != 1 for item in transitions):
        raise ValueError("Goal IQL requires vector observations")
    if any(item.legal_mask is None for item in transitions):
        raise ValueError("Goal IQL requires legal-action masks")
    states = np.concatenate(
        [
            np.stack([item.observation for item in transitions]),
            transitions[-1].next_observation[None],
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return GoalEpisode(
        states,
        np.asarray([item.action for item in transitions], dtype=np.int64),
        np.stack([item.legal_mask for item in transitions]).astype(np.bool_),
    )


def load_goal_episodes(
    database: str | Path,
    *,
    source: str | None = None,
    episode_ids: list[str] | None = None,
) -> tuple[list[GoalEpisode], int]:
    with ReplayStore(database, read_only=True) as store:
        selected_ids = store.episode_ids(source) if episode_ids is None else episode_ids
        action_dim = store.infer_action_dim(selected_ids)
        episodes = [
            _episode_from_transitions(
                list(store.transitions(episode_ids=[episode_id]))
            )
            for episode_id in selected_ids
        ]
    if not episodes:
        raise ValueError("Goal IQL replay selection is empty")
    state_dim = episodes[0].states.shape[1]
    if any(episode.states.shape[1] != state_dim for episode in episodes):
        raise ValueError("Goal IQL episodes have inconsistent state dimensions")
    return episodes, action_dim


def split_goal_episode_ids(
    store: ReplayStore,
    sources: str | Sequence[str] | None,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if sources is None or isinstance(sources, str):
        return store.split_episode_ids(
            sources, validation_fraction=validation_fraction, seed=seed
        )
    normalized = tuple(dict.fromkeys(str(source) for source in sources))
    if not normalized:
        raise ValueError("Goal IQL source selection cannot be empty")
    training: list[str] = []
    validation: list[str] = []
    for index, source in enumerate(normalized):
        source_training, source_validation = store.split_episode_ids(
            source,
            validation_fraction=validation_fraction,
            seed=seed + index,
        )
        training.extend(source_training)
        validation.extend(source_validation)
    return training, validation


class HindsightGoalSampler:
    def __init__(
        self,
        episodes: list[GoalEpisode],
        config: GoalRelabelConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if not episodes:
            raise ValueError("Hindsight goal sampling requires episodes")
        self.episodes = episodes
        self.config = config or GoalRelabelConfig()
        self.config.validate()
        self.generator = np.random.default_rng(seed)
        self.mode_probabilities = np.asarray(
            [
                self.config.next_fraction,
                self.config.future_fraction,
                self.config.random_fraction,
            ],
            dtype=np.float64,
        )

    def sample(self, batch_size: int) -> GoalBatch:
        if batch_size < 1:
            raise ValueError("Goal batch size must be positive")
        states: list[np.ndarray] = []
        actions: list[int] = []
        next_states: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[float] = []
        legal_masks: list[np.ndarray] = []
        modes: list[str] = []
        for _ in range(batch_size):
            episode = self.episodes[int(self.generator.integers(len(self.episodes)))]
            step = int(self.generator.integers(len(episode.actions)))
            mode_index = int(self.generator.choice(3, p=self.mode_probabilities))
            if mode_index == 0:
                goal = episode.states[step + 1]
                mode = "next"
            elif mode_index == 1:
                goal_index = int(
                    self.generator.integers(step + 1, len(episode.states))
                )
                goal = episode.states[goal_index]
                mode = "future"
            else:
                goal_episode = self.episodes[
                    int(self.generator.integers(len(self.episodes)))
                ]
                goal = goal_episode.states[
                    int(self.generator.integers(1, len(goal_episode.states)))
                ]
                mode = "random"
            next_state = episode.states[step + 1]
            achieved = bool(np.array_equal(next_state, goal))
            boundary = step + 1 == len(episode.actions)
            states.append(episode.states[step])
            actions.append(int(episode.actions[step]))
            next_states.append(next_state)
            goals.append(goal)
            rewards.append(0.0 if achieved else -1.0)
            dones.append(float(achieved or boundary))
            legal_masks.append(episode.legal_masks[step])
            modes.append(mode)
        return GoalBatch(
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.long),
            torch.from_numpy(np.stack(next_states)),
            torch.from_numpy(np.stack(goals)),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.from_numpy(np.stack(legal_masks)),
            tuple(modes),
        )


class ClusterHindsightGoalSampler:
    """Relabel replay with JEPA cluster-equivalent prototype goals."""

    def __init__(
        self,
        episodes: list[GoalEpisode],
        cluster_ids: list[np.ndarray],
        prototypes: np.ndarray,
        config: GoalRelabelConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if not episodes or len(cluster_ids) != len(episodes):
            raise ValueError("Cluster goal sampling requires aligned episodes")
        if prototypes.ndim != 2:
            raise ValueError("Cluster goal prototypes must be a state matrix")
        if any(ids.shape != (len(episode.states),) for ids, episode in zip(cluster_ids, episodes, strict=True)):
            raise ValueError("Cluster ids must align with every episode state")
        if any(ids.min() < 0 or ids.max() >= len(prototypes) for ids in cluster_ids):
            raise ValueError("Cluster ids exceed the prototype vocabulary")
        self.episodes = episodes
        self.cluster_ids = cluster_ids
        self.prototypes = prototypes.astype(np.float32, copy=False)
        self.config = config or GoalRelabelConfig()
        self.config.validate()
        self.generator = np.random.default_rng(seed)
        self.mode_probabilities = np.asarray(
            [
                self.config.next_fraction,
                self.config.future_fraction,
                self.config.random_fraction,
            ],
            dtype=np.float64,
        )

    def sample(self, batch_size: int) -> GoalBatch:
        if batch_size < 1:
            raise ValueError("Goal batch size must be positive")
        states: list[np.ndarray] = []
        actions: list[int] = []
        next_states: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[float] = []
        legal_masks: list[np.ndarray] = []
        modes: list[str] = []
        for _ in range(batch_size):
            episode_index = int(self.generator.integers(len(self.episodes)))
            episode = self.episodes[episode_index]
            ids = self.cluster_ids[episode_index]
            step = int(self.generator.integers(len(episode.actions)))
            mode_index = int(self.generator.choice(3, p=self.mode_probabilities))
            if mode_index == 0:
                goal_cluster = int(ids[step + 1])
                mode = "next"
            elif mode_index == 1:
                goal_index = int(self.generator.integers(step + 1, len(episode.states)))
                goal_cluster = int(ids[goal_index])
                mode = "future"
            else:
                goal_episode_index = int(self.generator.integers(len(self.episodes)))
                goal_episode = self.episodes[goal_episode_index]
                goal_index = int(self.generator.integers(1, len(goal_episode.states)))
                goal_cluster = int(self.cluster_ids[goal_episode_index][goal_index])
                mode = "random"
            next_state = episode.states[step + 1]
            achieved = int(ids[step + 1]) == goal_cluster
            boundary = step + 1 == len(episode.actions)
            states.append(episode.states[step])
            actions.append(int(episode.actions[step]))
            next_states.append(next_state)
            goals.append(self.prototypes[goal_cluster])
            rewards.append(0.0 if achieved else -1.0)
            dones.append(float(achieved or boundary))
            legal_masks.append(episode.legal_masks[step])
            modes.append(mode)
        return GoalBatch(
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.long),
            torch.from_numpy(np.stack(next_states)),
            torch.from_numpy(np.stack(goals)),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            torch.from_numpy(np.stack(legal_masks)),
            tuple(modes),
        )


def build_cluster_goal_sampler(
    episodes: list[GoalEpisode],
    model: ActionConditionedJEPA,
    centers: torch.Tensor,
    prototypes: torch.Tensor,
    device: torch.device,
    config: GoalRelabelConfig,
    *,
    seed: int,
) -> ClusterHindsightGoalSampler:
    lengths = [len(episode.states) for episode in episodes]
    states = np.concatenate([episode.states for episode in episodes])
    latents = encode_vector_states(model, states, device=device)
    assignments = nearest_cluster(latents, centers).numpy()
    cluster_ids: list[np.ndarray] = []
    offset = 0
    for length in lengths:
        cluster_ids.append(assignments[offset : offset + length])
        offset += length
    return ClusterHindsightGoalSampler(
        episodes,
        cluster_ids,
        prototypes.numpy(),
        config,
        seed=seed,
    )


class _GoalFusion:
    def _initialize_goal_fusion(self, hidden_dim: int) -> None:
        self.goal_cell_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
        )
        self.goal_slot_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
        )
        self.goal_global_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
        )

    def _encode_goal_conditioned(self, state_goal: torch.Tensor):
        if state_goal.shape[-1] != self.state_dim * 2:
            raise ValueError(
                f"Expected concatenated state/goal width {self.state_dim * 2}"
            )
        state, goal = state_goal.split(self.state_dim, dim=-1)
        batch_shape, state_cells, state_slots, state_global = super().encode_state(state)
        _, goal_cells, goal_slots, goal_global = super().encode_state(goal)
        cells = self.goal_cell_fusion(
            torch.cat([state_cells, goal_cells, goal_cells - state_cells], dim=-1)
        )
        slots = self.goal_slot_fusion(
            torch.cat([state_slots, goal_slots, goal_slots - state_slots], dim=-1)
        )
        global_features = self.goal_global_fusion(
            torch.cat([state_global, goal_global, goal_global - state_global], dim=-1)
        )
        return batch_shape, cells, slots, global_features


class GoalConditionedSpatialQ(_GoalFusion, SpatialConvActorCritic):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(state_dim, action_dim, hidden_dim)
        self._initialize_goal_fusion(hidden_dim)

    def encode_state(self, state_goal: torch.Tensor):
        return self._encode_goal_conditioned(state_goal)


class GoalConditionedAutoregressiveActor(
    _GoalFusion, AutoregressiveSpatialConvActorCritic
):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(state_dim, action_dim, hidden_dim)
        self._initialize_goal_fusion(hidden_dim)

    def encode_state(self, state_goal: torch.Tensor):
        return self._encode_goal_conditioned(state_goal)


class GoalValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, goal, goal - state], dim=-1)).squeeze(-1)


def train_goal_iql(
    database: str | Path,
    device_info: DeviceInfo,
    config: GoalIQLConfig | None = None,
    *,
    source: str | Sequence[str] | None = None,
    seed: int = 0,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    initial_auxiliary: dict[str, object] | None = None,
    latent_goal_model: ActionConditionedJEPA | None = None,
    latent_goal_centers: torch.Tensor | None = None,
    latent_goal_prototypes: torch.Tensor | None = None,
) -> tuple[GoalConditionedAutoregressiveActor, GoalIQLResult, dict[str, object]]:
    config = config or GoalIQLConfig()
    config.validate()
    torch.manual_seed(seed)
    with ReplayStore(database, read_only=True) as store:
        training_ids, validation_ids = split_goal_episode_ids(
            store,
            source,
            validation_fraction=config.validation_fraction,
            seed=config.split_seed,
        )
    training_episodes, action_dim = load_goal_episodes(
        database, episode_ids=training_ids
    )
    validation_episodes = (
        load_goal_episodes(database, episode_ids=validation_ids)[0]
        if validation_ids
        else []
    )
    state_dim = int(training_episodes[0].states.shape[1])
    if state_dim != 585 or action_dim != 650:
        raise ValueError("Goal IQL currently requires portable vector-v2 dimensions 585/650")
    latent_parts = (
        latent_goal_model,
        latent_goal_centers,
        latent_goal_prototypes,
    )
    if any(part is not None for part in latent_parts) and not all(
        part is not None for part in latent_parts
    ):
        raise ValueError("Latent goal model, centers, and prototypes are all required")

    def make_sampler(episodes: list[GoalEpisode], sampler_seed: int):
        if latent_goal_model is None:
            return HindsightGoalSampler(episodes, config.relabel, seed=sampler_seed)
        assert latent_goal_centers is not None and latent_goal_prototypes is not None
        return build_cluster_goal_sampler(
            episodes,
            latent_goal_model,
            latent_goal_centers,
            latent_goal_prototypes,
            device_info.device,
            config.relabel,
            seed=sampler_seed,
        )

    sampler = make_sampler(training_episodes, seed)
    actor = GoalConditionedAutoregressiveActor(
        state_dim, action_dim, config.hidden_dim
    ).to(device_info.device)
    if initial_state_dict is not None:
        actor.load_state_dict(initial_state_dict)
    q1 = GoalConditionedSpatialQ(state_dim, action_dim, config.hidden_dim).to(
        device_info.device
    )
    q2 = GoalConditionedSpatialQ(state_dim, action_dim, config.hidden_dim).to(
        device_info.device
    )
    target_q1 = GoalConditionedSpatialQ(state_dim, action_dim, config.hidden_dim).to(
        device_info.device
    )
    target_q2 = GoalConditionedSpatialQ(state_dim, action_dim, config.hidden_dim).to(
        device_info.device
    )
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    target_q1.requires_grad_(False)
    target_q2.requires_grad_(False)
    value = GoalValueNetwork(state_dim, config.hidden_dim).to(device_info.device)
    if initial_auxiliary is not None:
        modules = {
            "q1_state_dict": q1,
            "q2_state_dict": q2,
            "target_q1_state_dict": target_q1,
            "target_q2_state_dict": target_q2,
            "value_state_dict": value,
        }
        for key, module in modules.items():
            state_dict = initial_auxiliary.get(key)
            if not isinstance(state_dict, dict):
                raise ValueError(f"Goal-IQL initializer is missing {key}")
            module.load_state_dict(state_dict)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.AdamW(
        [*q1.parameters(), *q2.parameters()], lr=config.learning_rate
    )
    value_optimizer = torch.optim.AdamW(value.parameters(), lr=config.learning_rate)
    samples_per_epoch = config.samples_per_epoch or sum(
        len(episode.actions) for episode in training_episodes
    )
    updates_per_epoch = max(1, int(np.ceil(samples_per_epoch / config.batch_size)))
    last_q_loss = last_value_loss = last_actor_loss = last_mean_weight = 0.0
    for _ in range(config.epochs):
        for _ in range(updates_per_epoch):
            batch = sampler.sample(config.batch_size)
            state = batch.state.to(device_info.device)
            action = batch.action.to(device_info.device)
            next_state = batch.next_state.to(device_info.device)
            goal = batch.goal.to(device_info.device)
            reward = batch.reward.to(device_info.device)
            done = batch.done.to(device_info.device)
            legal_mask = batch.legal_mask.to(device_info.device)
            state_goal = torch.cat([state, goal], dim=-1)

            with torch.no_grad():
                target_q = torch.minimum(
                    target_q1(state_goal)[0].gather(1, action[:, None]).squeeze(1),
                    target_q2(state_goal)[0].gather(1, action[:, None]).squeeze(1),
                )
            predicted_value = value(state, goal)
            value_error = target_q - predicted_value
            expectile_weight = torch.where(
                value_error > 0.0, config.expectile, 1.0 - config.expectile
            )
            value_loss = (expectile_weight * value_error.square()).mean()
            value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
            value_optimizer.step()

            with torch.no_grad():
                td_target = reward + config.discount * (1.0 - done) * value(
                    next_state, goal
                )
            predicted_q1 = q1(state_goal)[0].gather(1, action[:, None]).squeeze(1)
            predicted_q2 = q2(state_goal)[0].gather(1, action[:, None]).squeeze(1)
            q_loss = F.mse_loss(predicted_q1, td_target) + F.mse_loss(
                predicted_q2, td_target
            )
            critic_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_([*q1.parameters(), *q2.parameters()], 10.0)
            critic_optimizer.step()

            with torch.no_grad():
                advantage_weight = torch.exp(
                    config.advantage_temperature * (target_q - value(state, goal))
                ).clamp(max=config.max_advantage_weight)
            logits, _ = actor(state_goal)
            log_probability = F.log_softmax(
                logits.masked_fill(~legal_mask, -torch.inf), dim=1
            ).gather(1, action[:, None]).squeeze(1)
            actor_loss = -(advantage_weight * log_probability).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_optimizer.step()

            with torch.no_grad():
                for target_parameter, parameter in zip(
                    target_q1.parameters(), q1.parameters(), strict=True
                ):
                    target_parameter.lerp_(parameter, config.target_tau)
                for target_parameter, parameter in zip(
                    target_q2.parameters(), q2.parameters(), strict=True
                ):
                    target_parameter.lerp_(parameter, config.target_tau)
            last_q_loss = float(q_loss.detach().cpu())
            last_value_loss = float(value_loss.detach().cpu())
            last_actor_loss = float(actor_loss.detach().cpu())
            last_mean_weight = float(advantage_weight.mean().detach().cpu())

    diagnostic_sampler = make_sampler(
        validation_episodes or training_episodes, config.split_seed + 1
    )
    diagnostic = diagnostic_sampler.sample(min(1024, samples_per_epoch))
    state = diagnostic.state.to(device_info.device)
    goal = diagnostic.goal.to(device_info.device)
    with torch.inference_mode():
        diagnostic_values = value(state, goal).cpu()
    next_values = diagnostic_values[
        torch.tensor([mode == "next" for mode in diagnostic.modes])
    ]
    random_values = diagnostic_values[
        torch.tensor([mode == "random" for mode in diagnostic.modes])
    ]
    result = GoalIQLResult(
        training_episodes=len(training_episodes),
        validation_episodes=len(validation_episodes),
        training_transitions=sum(
            len(episode.actions) for episode in training_episodes
        ),
        validation_transitions=sum(
            len(episode.actions) for episode in validation_episodes
        ),
        q_loss=last_q_loss,
        value_loss=last_value_loss,
        actor_loss=last_actor_loss,
        mean_advantage_weight=last_mean_weight,
        next_goal_value=float(next_values.mean()) if len(next_values) else float("nan"),
        random_goal_value=(
            float(random_values.mean()) if len(random_values) else float("nan")
        ),
    )
    auxiliary: dict[str, object] = {
        "q1_state_dict": q1.state_dict(),
        "q2_state_dict": q2.state_dict(),
        "target_q1_state_dict": target_q1.state_dict(),
        "target_q2_state_dict": target_q2.state_dict(),
        "value_state_dict": value.state_dict(),
    }
    actor.eval()
    return actor, result, auxiliary

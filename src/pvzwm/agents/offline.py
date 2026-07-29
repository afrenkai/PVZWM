from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from pvzwm.agents.ppo import POLICY_ARCHITECTURES, PolicyModel, build_actor_critic
from pvzwm.data.store import ReplayStore
from pvzwm.device import DeviceInfo
from pvzwm.portable import PortableActionCodec


@dataclass(frozen=True, slots=True)
class BehaviorCloneConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 3e-4
    hidden_dim: int = 256
    informative_fraction: float = 0.5
    sampling: str = "informative"
    policy_architecture: str = "flat"


@dataclass(frozen=True, slots=True)
class BehaviorCloneResult:
    transitions: int
    episodes: int
    loss: float
    accuracy: float
    non_wait_accuracy: float


@dataclass(frozen=True, slots=True)
class DiscreteIQLConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 3e-4
    hidden_dim: int = 256
    discount: float = 0.999
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    max_advantage_weight: float = 100.0
    reward_scale: float = 0.01
    target_tau: float = 0.005
    success_fraction: float = 0.5
    informative_fraction: float = 0.5
    policy_architecture: str = "flat"


@dataclass(slots=True)
class DiscreteIQLResult:
    transitions: int
    episodes: int
    successful_episodes: int
    q_loss: float
    value_loss: float
    actor_loss: float
    behavior_accuracy: float
    non_wait_accuracy: float
    mean_advantage_weight: float
    auxiliary_checkpoint: dict[str, object]
    episode_returns: list[float] = field(default_factory=list, init=False)
    episode_wins: list[bool] = field(default_factory=list, init=False)
    episode_levels: list[int] = field(default_factory=list, init=False)

    def summary(self) -> dict[str, float]:
        return {
            "episodes": float(self.episodes),
            "successful_episodes": float(self.successful_episodes),
            "transitions": float(self.transitions),
            "q_loss": self.q_loss,
            "value_loss": self.value_loss,
            "actor_loss": self.actor_loss,
            "behavior_accuracy": self.behavior_accuracy,
            "non_wait_accuracy": self.non_wait_accuracy,
            "mean_advantage_weight": self.mean_advantage_weight,
        }


class StateValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


def _validate_iql_config(config: DiscreteIQLConfig) -> None:
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("IQL epochs and batch size must be positive")
    if config.learning_rate <= 0.0 or config.hidden_dim < 1:
        raise ValueError("IQL learning rate and hidden dimension must be positive")
    if not 0.0 <= config.discount <= 1.0:
        raise ValueError("IQL discount must be in [0, 1]")
    if not 0.0 < config.expectile < 1.0:
        raise ValueError("IQL expectile must be in (0, 1)")
    if config.advantage_temperature <= 0.0 or config.max_advantage_weight < 1.0:
        raise ValueError("IQL advantage weighting parameters are invalid")
    if config.reward_scale <= 0.0 or not 0.0 < config.target_tau <= 1.0:
        raise ValueError("IQL reward scale and target tau must be positive")
    if not 0.0 <= config.success_fraction <= 1.0:
        raise ValueError("IQL success_fraction must be in [0, 1]")
    if not 0.0 <= config.informative_fraction <= 1.0:
        raise ValueError("IQL informative_fraction must be in [0, 1]")
    if config.policy_architecture not in POLICY_ARCHITECTURES:
        raise ValueError(f"policy_architecture must be one of {POLICY_ARCHITECTURES}")


def _assign_sampling_mass(
    weights: torch.Tensor,
    actions: torch.Tensor,
    start: int,
    stop: int,
    mass: float,
    informative_fraction: float,
) -> None:
    if stop <= start or mass <= 0.0:
        return
    indices = torch.arange(start, stop)
    non_wait = actions[start:stop] != 0
    informative_count = int(non_wait.sum())
    wait_count = stop - start - informative_count
    if informative_count and wait_count:
        weights[indices[non_wait]] = mass * informative_fraction / informative_count
        weights[indices[~non_wait]] = mass * (1.0 - informative_fraction) / wait_count
    else:
        weights[start:stop] = mass / (stop - start)


def branch_balanced_sampling_weights(
    actions: torch.Tensor, action_dim: int
) -> torch.Tensor:
    """Give every portable semantic action branch equal sampling mass."""

    codec = PortableActionCodec()
    if action_dim != codec.size:
        raise ValueError("Branch-balanced sampling requires the portable action schema")
    labels: list[tuple[str, int]] = []
    for action in actions.tolist():
        decoded = codec.decode(int(action))
        slot = decoded.slot if decoded.kind.name == "PLANT" else 0
        labels.append((decoded.kind.name, slot))
    counts = Counter(labels)
    branch_count = len(counts)
    return torch.tensor(
        [1.0 / (branch_count * counts[label]) for label in labels],
        dtype=torch.double,
    )


def train_discrete_iql(
    database: str | Path,
    device_info: DeviceInfo,
    config: DiscreteIQLConfig | None = None,
    *,
    seed: int = 0,
    source: str | None = None,
    episode_ids: list[str] | None = None,
    initial_actor_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[PolicyModel, DiscreteIQLResult]:
    """Train legality-masked discrete IQL from complete replay episodes."""
    config = config or DiscreteIQLConfig()
    _validate_iql_config(config)
    torch.manual_seed(seed)
    with ReplayStore(database, read_only=True) as store:
        selected_ids = episode_ids or store.episode_ids(source)
        successful = set(store.successful_episode_ids(source)).intersection(selected_ids)
        success_ids = [episode_id for episode_id in selected_ids if episode_id in successful]
        failure_ids = [episode_id for episode_id in selected_ids if episode_id not in successful]
        success_transitions = list(store.transitions(episode_ids=success_ids))
        failure_transitions = list(store.transitions(episode_ids=failure_ids))
        action_dim = store.infer_action_dim(selected_ids)
    transitions = [*success_transitions, *failure_transitions]
    if not transitions:
        raise ValueError("IQL replay selection is empty")
    if any(item.observation.ndim != 1 for item in transitions):
        raise ValueError("IQL requires vector observations")
    if any(item.legal_mask is None for item in transitions):
        raise ValueError("IQL requires legal-action masks")

    states = torch.from_numpy(
        np.stack([item.observation for item in transitions]).astype(np.float32)
    )
    actions = torch.tensor([item.action for item in transitions], dtype=torch.long)
    rewards = torch.tensor([item.reward for item in transitions], dtype=torch.float32)
    next_states = torch.from_numpy(
        np.stack([item.next_observation for item in transitions]).astype(np.float32)
    )
    dones = torch.tensor(
        [item.terminated for item in transitions], dtype=torch.float32
    )
    legal_masks = torch.from_numpy(
        np.stack([item.legal_mask for item in transitions]).astype(np.bool_)
    )
    if not bool(legal_masks[torch.arange(len(actions)), actions].all()):
        raise ValueError("IQL replay contains an action marked illegal by its mask")

    success_stop = len(success_transitions)
    sampling_weights = torch.zeros(len(transitions), dtype=torch.double)
    if success_transitions and failure_transitions:
        success_mass = config.success_fraction
    else:
        success_mass = 1.0 if success_transitions else 0.0
    _assign_sampling_mass(
        sampling_weights,
        actions,
        0,
        success_stop,
        success_mass,
        config.informative_fraction,
    )
    _assign_sampling_mass(
        sampling_weights,
        actions,
        success_stop,
        len(transitions),
        1.0 - success_mass,
        config.informative_fraction,
    )
    sampler = WeightedRandomSampler(
        sampling_weights,
        num_samples=len(transitions),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        TensorDataset(states, actions, rewards, next_states, dones, legal_masks),
        batch_size=config.batch_size,
        sampler=sampler,
    )

    state_dim = int(states.shape[1])
    actor = build_actor_critic(
        state_dim, action_dim, config.hidden_dim, config.policy_architecture
    ).to(device_info.device)
    if initial_actor_state_dict is not None:
        actor.load_state_dict(initial_actor_state_dict)
    q1 = build_actor_critic(
        state_dim, action_dim, config.hidden_dim, config.policy_architecture
    ).to(device_info.device)
    q2 = build_actor_critic(
        state_dim, action_dim, config.hidden_dim, config.policy_architecture
    ).to(device_info.device)
    target_q1 = build_actor_critic(
        state_dim, action_dim, config.hidden_dim, config.policy_architecture
    ).to(device_info.device)
    target_q2 = build_actor_critic(
        state_dim, action_dim, config.hidden_dim, config.policy_architecture
    ).to(device_info.device)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    target_q1.requires_grad_(False)
    target_q2.requires_grad_(False)
    value = StateValueNetwork(state_dim, config.hidden_dim).to(device_info.device)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.AdamW(
        [*q1.parameters(), *q2.parameters()], lr=config.learning_rate
    )
    value_optimizer = torch.optim.AdamW(value.parameters(), lr=config.learning_rate)

    last_q_loss = 0.0
    last_value_loss = 0.0
    last_actor_loss = 0.0
    last_mean_weight = 0.0
    actor.train()
    q1.train()
    q2.train()
    value.train()
    for _ in range(config.epochs):
        for state, action, reward, next_state, done, legal_mask in loader:
            state = state.to(device_info.device)
            action = action.to(device_info.device)
            reward = reward.to(device_info.device)
            next_state = next_state.to(device_info.device)
            done = done.to(device_info.device)
            legal_mask = legal_mask.to(device_info.device)

            with torch.no_grad():
                target_q = torch.minimum(
                    target_q1(state)[0].gather(1, action[:, None]).squeeze(1),
                    target_q2(state)[0].gather(1, action[:, None]).squeeze(1),
                )
            predicted_value = value(state)
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
                td_target = config.reward_scale * reward + (
                    config.discount * (1.0 - done) * value(next_state)
                )
            predicted_q1 = q1(state)[0].gather(1, action[:, None]).squeeze(1)
            predicted_q2 = q2(state)[0].gather(1, action[:, None]).squeeze(1)
            q_loss = F.mse_loss(predicted_q1, td_target) + F.mse_loss(
                predicted_q2, td_target
            )
            critic_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_([*q1.parameters(), *q2.parameters()], 10.0)
            critic_optimizer.step()

            with torch.no_grad():
                advantage = target_q - value(state)
                advantage_weight = torch.exp(
                    config.advantage_temperature * advantage
                ).clamp(max=config.max_advantage_weight)
            logits, _ = actor(state)
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

    actor.eval()
    predictions: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(states), config.batch_size):
            state = states[start : start + config.batch_size].to(device_info.device)
            legal = legal_masks[start : start + config.batch_size].to(device_info.device)
            logits, _ = actor(state)
            predictions.append(logits.masked_fill(~legal, -torch.inf).argmax(1).cpu())
    predicted = torch.cat(predictions)
    non_wait = actions != 0
    behavior_accuracy = float((predicted == actions).to(torch.float32).mean())
    non_wait_accuracy = (
        float((predicted[non_wait] == actions[non_wait]).to(torch.float32).mean())
        if bool(non_wait.any())
        else 0.0
    )
    auxiliary: dict[str, object] = {
        "algorithm": "discrete_iql",
        "q1_state_dict": q1.state_dict(),
        "q2_state_dict": q2.state_dict(),
        "target_q1_state_dict": target_q1.state_dict(),
        "target_q2_state_dict": target_q2.state_dict(),
        "value_state_dict": value.state_dict(),
    }
    return actor, DiscreteIQLResult(
        transitions=len(transitions),
        episodes=len(selected_ids),
        successful_episodes=len(success_ids),
        q_loss=last_q_loss,
        value_loss=last_value_loss,
        actor_loss=last_actor_loss,
        behavior_accuracy=behavior_accuracy,
        non_wait_accuracy=non_wait_accuracy,
        mean_advantage_weight=last_mean_weight,
        auxiliary_checkpoint=auxiliary,
    )


def train_behavior_clone(
    database: str | Path,
    device_info: DeviceInfo,
    config: BehaviorCloneConfig | None = None,
    *,
    seed: int = 0,
    episode_ids: list[str] | None = None,
) -> tuple[PolicyModel, BehaviorCloneResult]:
    """Distill legal planner actions into an ActorCritic policy initializer."""
    config = config or BehaviorCloneConfig()
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("Behavior-cloning epochs and batch size must be positive")
    if not 0.0 <= config.informative_fraction <= 1.0:
        raise ValueError("informative_fraction must be in [0, 1]")
    if config.sampling not in {"informative", "branch-balanced"}:
        raise ValueError("Behavior-cloning sampling must be informative or branch-balanced")
    if config.policy_architecture not in POLICY_ARCHITECTURES:
        raise ValueError(f"policy_architecture must be one of {POLICY_ARCHITECTURES}")
    torch.manual_seed(seed)
    with ReplayStore(database, read_only=True) as store:
        transitions = list(store.transitions(episode_ids=episode_ids))
        action_dim = store.infer_action_dim(episode_ids)
    if not transitions:
        raise ValueError("Behavior-cloning replay selection is empty")
    if any(item.observation.ndim != 1 for item in transitions):
        raise ValueError("Behavior cloning requires vector observations")
    if any(item.legal_mask is None for item in transitions):
        raise ValueError("Behavior cloning requires legal-action masks")

    states = torch.from_numpy(
        np.stack([item.observation for item in transitions]).astype(np.float32)
    )
    actions = torch.tensor([item.action for item in transitions], dtype=torch.long)
    legal_masks = torch.from_numpy(
        np.stack([item.legal_mask for item in transitions]).astype(np.bool_)
    )
    non_wait = actions != 0
    weights = torch.ones(len(transitions), dtype=torch.double)
    informative_count = int(non_wait.sum())
    wait_count = len(transitions) - informative_count
    if config.sampling == "branch-balanced":
        weights = branch_balanced_sampling_weights(actions, action_dim)
    elif informative_count and wait_count:
        weights[non_wait] = config.informative_fraction / informative_count
        weights[~non_wait] = (1.0 - config.informative_fraction) / wait_count
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(transitions),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        TensorDataset(states, actions, legal_masks),
        batch_size=config.batch_size,
        sampler=sampler,
    )

    model = build_actor_critic(
        int(states.shape[1]),
        action_dim,
        config.hidden_dim,
        config.policy_architecture,
    ).to(device_info.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    last_loss = 0.0
    model.train()
    for _ in range(config.epochs):
        for state, action, legal_mask in loader:
            state = state.to(device_info.device)
            action = action.to(device_info.device)
            legal_mask = legal_mask.to(device_info.device)
            logits, _ = model(state)
            loss = F.cross_entropy(logits.masked_fill(~legal_mask, -torch.inf), action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    model.eval()
    with torch.inference_mode():
        predictions: list[torch.Tensor] = []
        for start in range(0, len(states), config.batch_size):
            state = states[start : start + config.batch_size].to(device_info.device)
            legal = legal_masks[start : start + config.batch_size].to(device_info.device)
            logits, _ = model(state)
            predictions.append(logits.masked_fill(~legal, -torch.inf).argmax(dim=1).cpu())
        predicted = torch.cat(predictions)
    accuracy = float((predicted == actions).to(torch.float32).mean())
    non_wait_accuracy = (
        float((predicted[non_wait] == actions[non_wait]).to(torch.float32).mean())
        if informative_count
        else 0.0
    )
    return model, BehaviorCloneResult(
        transitions=len(transitions),
        episodes=0 if episode_ids is None else len(episode_ids),
        loss=last_loss,
        accuracy=accuracy,
        non_wait_accuracy=non_wait_accuracy,
    )

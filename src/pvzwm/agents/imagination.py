from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pvzwm.agents.ppo import ActorCritic, PPOBatch, ppo_loss
from pvzwm.data.store import ReplayStore
from pvzwm.device import DeviceInfo
from pvzwm.models.flow import RepresentationFlowWorldModel


@dataclass(frozen=True, slots=True)
class FlowPPOTrainConfig:
    updates: int = 1_000
    parallel_envs: int = 64
    imagination_horizon: int = 16
    update_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    hidden_dim: int = 256
    flow_steps: int = 4
    stochastic_dynamics: bool = True
    legality_threshold: float | None = None
    done_threshold: float = 0.5
    max_start_states: int | None = None


@dataclass(slots=True)
class FlowPPOTrainResult:
    losses: list[float]
    imagined_episode_returns: list[float]
    imagined_transitions: int
    mean_legal_fraction: float
    legality_threshold: float
    legality_precision: float
    legality_recall: float
    legality_f1: float

    def summary(self) -> dict[str, float]:
        recent = self.imagined_episode_returns[-100:]
        return {
            "updates": float(len(self.losses)),
            "imagined_transitions": float(self.imagined_transitions),
            "imagined_episodes": float(len(self.imagined_episode_returns)),
            "mean_imagined_return_100": float(np.mean(recent)) if recent else 0.0,
            "mean_legal_fraction": self.mean_legal_fraction,
            "legality_threshold": self.legality_threshold,
            "legality_precision": self.legality_precision,
            "legality_recall": self.legality_recall,
            "legality_f1": self.legality_f1,
            "last_loss": self.losses[-1] if self.losses else 0.0,
        }


def _pixel_starts(
    database: str | Path,
    episode_ids: list[str] | None,
    limit: int | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    starts: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    with ReplayStore(database, read_only=True) as store:
        for transition in store.transitions(limit=limit, episode_ids=episode_ids):
            if transition.observation.ndim != 3:
                raise ValueError("Flow PPO requires channel-first pixel replay")
            if transition.legal_mask is None:
                raise ValueError("Flow PPO replay is missing legal-action masks")
            starts.append(transition.observation)
            legal_masks.append(transition.legal_mask)
    if not starts:
        raise ValueError("Flow PPO replay selection is empty")
    return starts, legal_masks


def _pixel_tensor(items: list[np.ndarray], device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(np.stack(items)).to(device=device, dtype=torch.float32)
    if items[0].dtype == np.uint8:
        value = value.div(255.0)
    return value


@torch.inference_mode()
def _calibrate_legality_threshold(
    world_model: RepresentationFlowWorldModel,
    starts: list[np.ndarray],
    legal_masks: list[np.ndarray],
    device: torch.device,
    rng: np.random.Generator,
    threshold: float | None,
    max_samples: int = 4_096,
) -> tuple[float, float, float, float]:
    sample_count = min(len(starts), max_samples)
    indices = rng.choice(len(starts), size=sample_count, replace=False)
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for offset in range(0, sample_count, 256):
        batch_indices = indices[offset : offset + 256]
        observations = _pixel_tensor(
            [starts[int(index)] for index in batch_indices], device
        )
        latent = world_model.encode(observations)
        probabilities.append(world_model.legal_logits(latent).sigmoid().cpu())
        targets.append(
            torch.from_numpy(
                np.stack([legal_masks[int(index)] for index in batch_indices])
            ).to(torch.bool)
        )
    probability = torch.cat(probabilities).flatten()
    target = torch.cat(targets).flatten()
    candidates = (
        [float(threshold)]
        if threshold is not None
        else [*np.linspace(0.05, 0.95, 19).tolist(), 0.975, 0.99]
    )
    best = (float("-inf"), 0.0, 0.0, candidates[0])
    for candidate in candidates:
        predicted = probability >= candidate
        true_positive = float((predicted & target).sum())
        precision = true_positive / max(float(predicted.sum()), 1.0)
        recall = true_positive / max(float(target.sum()), 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        score = (f1, precision, recall, candidate)
        if score > best:
            best = score
    f1, precision, recall, selected = best
    return float(selected), float(precision), float(recall), float(f1)


def train_flow_ppo(
    device_info: DeviceInfo,
    database: str | Path,
    world_model: RepresentationFlowWorldModel,
    config: FlowPPOTrainConfig | None = None,
    seed: int = 0,
    episode_ids: list[str] | None = None,
) -> tuple[ActorCritic, FlowPPOTrainResult]:
    """Train PPO solely on trajectories imagined by a frozen pixel world model."""
    config = config or FlowPPOTrainConfig()
    if min(
        config.updates,
        config.parallel_envs,
        config.imagination_horizon,
        config.update_epochs,
        config.minibatch_size,
        config.flow_steps,
    ) < 1:
        raise ValueError("Flow PPO counts and horizons must be positive")
    if config.legality_threshold is not None and not (
        0.0 < config.legality_threshold < 1.0
    ):
        raise ValueError("Legality threshold must be between zero and one")
    if not 0.0 < config.done_threshold < 1.0:
        raise ValueError("Done threshold must be between zero and one")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    starts, start_legal_masks = _pixel_starts(
        database, episode_ids=episode_ids, limit=config.max_start_states
    )
    world_model = world_model.to(device_info.device).eval()
    world_model.requires_grad_(False)
    latent_dim = (
        world_model.code_dim * world_model.latent_height * world_model.latent_width
    )
    agent = ActorCritic(
        latent_dim, world_model.action_dim, config.hidden_dim
    ).to(device_info.device)
    optimizer = torch.optim.AdamW(agent.parameters(), lr=config.learning_rate)
    legality_threshold, legality_precision, legality_recall, legality_f1 = (
        _calibrate_legality_threshold(
            world_model,
            starts,
            start_legal_masks,
            device_info.device,
            rng,
            config.legality_threshold,
        )
    )
    result = FlowPPOTrainResult(
        [],
        [],
        0,
        0.0,
        legality_threshold,
        legality_precision,
        legality_recall,
        legality_f1,
    )
    running_returns = torch.zeros(config.parallel_envs, device=device_info.device)
    legal_total = 0.0
    legal_count = 0

    def sample_latents(count: int) -> torch.Tensor:
        indices = rng.integers(0, len(starts), size=count)
        observations = _pixel_tensor(
            [starts[int(index)] for index in indices], device_info.device
        )
        with torch.inference_mode():
            return world_model.encode(observations)

    latent = sample_latents(config.parallel_envs)
    for _ in range(config.updates):
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        old_log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        next_values: list[torch.Tensor] = []
        legal_masks: list[torch.Tensor] = []

        for _ in range(config.imagination_horizon):
            flat_state = latent.flatten(1)
            with torch.inference_mode():
                legal_mask = world_model.legal_mask(
                    latent, threshold=legality_threshold
                )
                logits, value = agent(flat_state)
                logits = logits.masked_fill(~legal_mask, -torch.inf)
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                old_log_probability = distribution.log_prob(action)
                imagined = world_model.imagine_latent(
                    latent,
                    action,
                    steps=config.flow_steps,
                    stochastic=config.stochastic_dynamics,
                )
                done = imagined.done_logit.sigmoid() >= config.done_threshold
                _, next_value = agent(imagined.latent.flatten(1))
                next_value = next_value.masked_fill(done, 0.0)

            states.append(flat_state.detach())
            actions.append(action)
            old_log_probabilities.append(old_log_probability)
            values.append(value)
            rewards.append(imagined.reward)
            dones.append(done.to(torch.float32))
            next_values.append(next_value)
            legal_masks.append(legal_mask)
            legal_total += float(legal_mask.float().mean().cpu())
            legal_count += 1
            result.imagined_transitions += config.parallel_envs
            running_returns += imagined.reward
            for episode_return in running_returns[done].cpu().tolist():
                result.imagined_episode_returns.append(float(episode_return))
            running_returns = running_returns.masked_fill(done, 0.0)
            latent = imagined.latent
            if done.any():
                latent = latent.clone()
                latent[done] = sample_latents(int(done.sum().item()))

        reward_tensor = torch.stack(rewards)
        done_tensor = torch.stack(dones)
        value_tensor = torch.stack(values)
        next_value_tensor = torch.stack(next_values)
        advantages = torch.zeros_like(reward_tensor)
        gae = torch.zeros(config.parallel_envs, device=device_info.device)
        for index in range(config.imagination_horizon - 1, -1, -1):
            active = 1.0 - done_tensor[index]
            delta = (
                reward_tensor[index]
                + config.discount * active * next_value_tensor[index]
                - value_tensor[index]
            )
            gae = delta + config.discount * config.gae_lambda * active * gae
            advantages[index] = gae
        returns = advantages + value_tensor
        flat_advantages = advantages.flatten()
        flat_advantages = (
            flat_advantages - flat_advantages.mean()
        ) / (flat_advantages.std(unbiased=False) + 1e-8)
        tensors = {
            "state": torch.stack(states).flatten(0, 1),
            "action": torch.stack(actions).flatten(),
            "old_log_probability": torch.stack(old_log_probabilities).flatten(),
            "advantage": flat_advantages,
            "returns": returns.flatten(),
            "legal_mask": torch.stack(legal_masks).flatten(0, 1),
        }
        batch_size = len(tensors["action"])
        update_losses: list[float] = []
        for _ in range(config.update_epochs):
            permutation = torch.randperm(batch_size, device=device_info.device)
            for start in range(0, batch_size, config.minibatch_size):
                indices = permutation[start : start + config.minibatch_size]
                batch = PPOBatch(
                    **{name: value[indices] for name, value in tensors.items()}
                )
                optimizer.zero_grad(set_to_none=True)
                losses = ppo_loss(
                    agent,
                    batch,
                    clip_ratio=config.clip_ratio,
                    value_weight=config.value_weight,
                    entropy_weight=config.entropy_weight,
                )
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 10.0)
                optimizer.step()
                update_losses.append(float(losses["loss"].detach().cpu()))
        result.losses.append(float(np.mean(update_losses)))

    result.mean_legal_fraction = legal_total / max(legal_count, 1)
    return agent, result

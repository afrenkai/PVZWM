from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


POLICY_ARCHITECTURES = (
    "flat",
    "spatial",
    "spatial-conv",
    "portable-autoregressive",
    "portable-vdn-autoregressive",
)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        return self.policy(features), self.value(features).squeeze(-1)


class SpatialActorCritic(nn.Module):
    """Portable policy sharing structure across seed slots and board cells."""

    rows = 6
    cols = 9
    cells = rows * cols
    seed_slots = 10
    grid_channels = 10
    portable_state_dim = 585
    portable_action_dim = 650

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if state_dim != self.portable_state_dim or action_dim != self.portable_action_dim:
            raise ValueError("Spatial policy requires portable vector-v2 dimensions 585/650")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        coordinates = []
        for row in range(self.rows):
            for col in range(self.cols):
                coordinates.append(
                    [row / (self.rows - 1), col / (self.cols - 1)]
                )
        self.register_buffer(
            "cell_coordinates", torch.tensor(coordinates, dtype=torch.float32)
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(self.grid_channels + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.slot_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 15, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.plant_cell = nn.Linear(hidden_dim, hidden_dim)
        self.plant_slot = nn.Linear(hidden_dim, hidden_dim)
        self.slot_bias = nn.Linear(hidden_dim, 1)
        self.shovel = nn.Linear(hidden_dim, 1)
        self.board_click = nn.Linear(hidden_dim, 1)
        self.global_actions = nn.Linear(hidden_dim, 2)
        self.value = nn.Linear(hidden_dim, 1)

    def mix_cell_features(self, cell_features: torch.Tensor) -> torch.Tensor:
        return cell_features

    def encode_state(
        self, state: torch.Tensor
    ) -> tuple[
        torch.Size,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_shape = state.shape[:-1]
        flat = state.reshape(-1, self.state_dim)
        grid = flat[:, : self.grid_channels * self.cells].reshape(
            -1, self.grid_channels, self.cells
        ).transpose(1, 2)
        coordinates = self.cell_coordinates.unsqueeze(0).expand(flat.shape[0], -1, -1)
        cell_features = self.cell_encoder(torch.cat([grid, coordinates], dim=-1))
        cell_features = self.mix_cell_features(cell_features)
        packets = flat[:, self.grid_channels * self.cells : -15].reshape(
            -1, self.seed_slots, 3
        )
        slot_features = self.slot_encoder(packets)
        scalars = flat[:, -15:]
        global_features = self.global_encoder(
            torch.cat(
                [cell_features.mean(dim=1), slot_features.mean(dim=1), scalars],
                dim=-1,
            )
        )
        contextual_cells = torch.tanh(cell_features + global_features.unsqueeze(1))
        contextual_slots = torch.tanh(slot_features + global_features.unsqueeze(1))
        return batch_shape, contextual_cells, contextual_slots, global_features

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape, contextual_cells, contextual_slots, global_features = (
            self.encode_state(state)
        )
        plant_cells = self.plant_cell(contextual_cells)
        plant_slots = self.plant_slot(contextual_slots)
        plant_logits = torch.einsum("bsh,bch->bsc", plant_slots, plant_cells)
        plant_logits = plant_logits / self.hidden_dim**0.5
        plant_logits = plant_logits + self.slot_bias(contextual_slots)
        global_logits = self.global_actions(global_features)
        logits = torch.cat(
            [
                global_logits[:, :1],
                plant_logits.flatten(1),
                self.shovel(contextual_cells).squeeze(-1),
                self.board_click(contextual_cells).squeeze(-1),
                global_logits[:, 1:],
            ],
            dim=1,
        )
        value = self.value(global_features).squeeze(-1)
        return (
            logits.reshape(*batch_shape, self.action_dim),
            value.reshape(*batch_shape),
        )


class SpatialConvActorCritic(SpatialActorCritic):
    """Spatial policy with learned local mixing over the logical 6x9 board."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(state_dim, action_dim, hidden_dim)
        self.cell_mixer = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.cell_mixer_norm = nn.LayerNorm(hidden_dim)

    def mix_cell_features(self, cell_features: torch.Tensor) -> torch.Tensor:
        grid = cell_features.transpose(1, 2).reshape(
            -1, self.hidden_dim, self.rows, self.cols
        )
        mixed = self.cell_mixer(grid).flatten(2).transpose(1, 2)
        return self.cell_mixer_norm(cell_features + mixed)


class AutoregressiveSpatialConvActorCritic(SpatialConvActorCritic):
    """Portable joint policy factored as kind, slot, row, then column.

    The returned tensor contains normalized joint log probabilities in the
    existing 650-action vocabulary, so all PPO, BC, masking, and bridge code can
    remain unchanged. Seed slots remain opaque; only the portable action schema
    is encoded.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(state_dim, action_dim, hidden_dim)
        del self.plant_cell
        del self.plant_slot
        del self.slot_bias
        del self.shovel
        del self.board_click
        del self.global_actions
        self.kind_head = nn.Linear(hidden_dim, 5)
        self.plant_slot_query = nn.Linear(hidden_dim, hidden_dim)
        self.plant_cell_key = nn.Linear(hidden_dim, hidden_dim)
        self.slot_head = nn.Linear(hidden_dim, 1)
        self.shovel_cell_head = nn.Linear(hidden_dim, 1)
        self.click_cell_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _cell_log_probabilities(cell_logits: torch.Tensor) -> torch.Tensor:
        """Factor a 6x9 cell distribution into row then conditional column."""

        row_logits = torch.logsumexp(cell_logits, dim=-1)
        row_log_probability = F.log_softmax(row_logits, dim=-1)
        column_log_probability = F.log_softmax(cell_logits, dim=-1)
        return row_log_probability.unsqueeze(-1) + column_log_probability

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape, cells, slots, global_features = self.encode_state(state)
        kind = F.log_softmax(self.kind_head(global_features), dim=-1)
        slot = F.log_softmax(self.slot_head(slots).squeeze(-1), dim=-1)

        slot_query = self.plant_slot_query(slots)
        cell_key = self.plant_cell_key(cells)
        plant_cell_logits = torch.einsum("bsh,bch->bsc", slot_query, cell_key)
        plant_cell_logits = plant_cell_logits / self.hidden_dim**0.5
        plant_cell_logits = plant_cell_logits.reshape(
            -1, self.seed_slots, self.rows, self.cols
        )
        plant_cell = self._cell_log_probabilities(plant_cell_logits)
        plant = (
            kind[:, 1, None, None, None]
            + slot[:, :, None, None]
            + plant_cell
        )

        cell_grid = cells.reshape(-1, self.rows, self.cols, self.hidden_dim)
        shovel_cell = self._cell_log_probabilities(
            self.shovel_cell_head(cell_grid).squeeze(-1)
        )
        click_cell = self._cell_log_probabilities(
            self.click_cell_head(cell_grid).squeeze(-1)
        )
        joint_log_probabilities = torch.cat(
            [
                kind[:, :1],
                plant.flatten(1),
                (kind[:, 2, None, None] + shovel_cell).flatten(1),
                (kind[:, 3, None, None] + click_cell).flatten(1),
                kind[:, 4:],
            ],
            dim=1,
        )
        value = self.value(global_features).squeeze(-1)
        return (
            joint_log_probabilities.reshape(*batch_shape, self.action_dim),
            value.reshape(*batch_shape),
        )


class VDNAutoregressiveSpatialConvActorCritic(
    AutoregressiveSpatialConvActorCritic
):
    """Autoregressive actor with a behavior-centered branching VDN critic.

    The critic factors the portable action schema into kind, opaque seed slot,
    row, and column.  It does not attach game semantics to seed slots or board
    locations.  Conditional factors are activated from the action vocabulary:
    kind is always active, slot only for placement, and row/column only for
    spatial actions.
    """

    factor_sizes = (5, 10, 6, 9)
    factor_count = len(factor_sizes)

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__(state_dim, action_dim, hidden_dim)
        self.kind_advantage = nn.Linear(hidden_dim, self.factor_sizes[0])
        self.slot_advantage = nn.Linear(hidden_dim, 1)
        self.row_advantage = nn.Linear(hidden_dim, 1)
        self.column_advantage = nn.Linear(hidden_dim, 1)

        action_factors = torch.zeros(
            self.portable_action_dim, self.factor_count, dtype=torch.long
        )
        active_factors = torch.zeros(
            self.portable_action_dim, self.factor_count, dtype=torch.bool
        )
        active_factors[:, 0] = True
        action_factors[0, 0] = 0
        for slot in range(self.seed_slots):
            for cell in range(self.cells):
                action = 1 + slot * self.cells + cell
                action_factors[action] = torch.tensor(
                    [1, slot, cell // self.cols, cell % self.cols]
                )
                active_factors[action] = True
        shovel_begin = 1 + self.seed_slots * self.cells
        click_begin = shovel_begin + self.cells
        for cell in range(self.cells):
            row, column = divmod(cell, self.cols)
            action_factors[shovel_begin + cell] = torch.tensor([2, 0, row, column])
            action_factors[click_begin + cell] = torch.tensor([3, 0, row, column])
            active_factors[shovel_begin + cell, [0, 2, 3]] = True
            active_factors[click_begin + cell, [0, 2, 3]] = True
        action_factors[-1, 0] = 4
        self.register_buffer("action_factors", action_factors)
        self.register_buffer("active_factors", active_factors)

    def raw_factor_advantages(self, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_shape, cells, slots, global_features = self.encode_state(state)
        flat_batch = global_features.shape[0]
        grid = cells.reshape(flat_batch, self.rows, self.cols, self.hidden_dim)
        advantages = (
            self.kind_advantage(global_features),
            self.slot_advantage(slots).squeeze(-1),
            self.row_advantage(grid.mean(dim=2)).squeeze(-1),
            self.column_advantage(grid.mean(dim=1)).squeeze(-1),
        )
        return tuple(
            advantage.reshape(*batch_shape, advantage.shape[-1])
            for advantage in advantages
        )

    def centered_factor_advantages(
        self,
        state: torch.Tensor,
        behavior_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Center each critic stream under the frozen rollout policy."""

        raw = self.raw_factor_advantages(state)
        centered: list[torch.Tensor] = []
        offset = 0
        for advantage, size in zip(raw, self.factor_sizes, strict=True):
            probability = behavior_probabilities[..., offset : offset + size]
            centered.append(
                advantage - (probability * advantage).sum(dim=-1, keepdim=True)
            )
            offset += size
        return tuple(centered)

    def factor_distribution(
        self,
        logits: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Marginalize the legal joint policy into frozen factor distributions."""

        masked_logits = logits.masked_fill(~legal_mask, -torch.inf)
        joint_probability = torch.softmax(masked_logits, dim=-1)
        distributions: list[torch.Tensor] = []
        for factor, size in enumerate(self.factor_sizes):
            active = self.active_factors[:, factor]
            source = joint_probability * active.to(joint_probability.dtype)
            marginal = torch.zeros(
                *joint_probability.shape[:-1],
                size,
                dtype=joint_probability.dtype,
                device=joint_probability.device,
            )
            index = self.action_factors[:, factor].expand(
                *joint_probability.shape[:-1], -1
            )
            marginal.scatter_add_(-1, index, source)
            total = marginal.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(marginal, 1.0 / size)
            distributions.append(
                torch.where(total > 0.0, marginal / total.clamp_min(1e-12), uniform)
            )
        return torch.cat(distributions, dim=-1)

    def selected_factor_values(
        self,
        factor_values: tuple[torch.Tensor, ...],
        action: torch.Tensor,
    ) -> torch.Tensor:
        selected = []
        factors = self.action_factors[action]
        for factor, values in enumerate(factor_values):
            selected.append(values.gather(-1, factors[..., factor, None]).squeeze(-1))
        return torch.stack(selected, dim=-1)

    def selected_factor_log_probabilities(
        self,
        factor_probabilities: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        selected = []
        factors = self.action_factors[action]
        offset = 0
        for factor, size in enumerate(self.factor_sizes):
            probability = factor_probabilities[..., offset : offset + size]
            selected.append(
                probability.gather(-1, factors[..., factor, None])
                .squeeze(-1)
                .clamp_min(1e-12)
                .log()
            )
            offset += size
        return torch.stack(selected, dim=-1)

    def factor_importance_weights(
        self,
        state: torch.Tensor,
        behavior_probabilities: torch.Tensor,
        action: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Range-based VDN weights, normalized only across active factors."""

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("VDN importance alpha must be in [0, 1]")
        centered = self.centered_factor_advantages(state, behavior_probabilities)
        ranges = torch.stack(
            [values.amax(dim=-1) - values.amin(dim=-1) for values in centered],
            dim=-1,
        )
        active = self.active_factors[action]
        if alpha == 0.0:
            powered = torch.ones_like(ranges)
        else:
            powered = ranges.clamp_min(0.0).pow(alpha)
        powered = powered * active.to(powered.dtype)
        total = powered.sum(dim=-1, keepdim=True)
        uniform = active.to(powered.dtype) / active.sum(dim=-1, keepdim=True)
        return torch.where(total > 0.0, powered / total.clamp_min(1e-12), uniform)


PolicyModel = (
    ActorCritic
    | SpatialActorCritic
    | SpatialConvActorCritic
    | AutoregressiveSpatialConvActorCritic
    | VDNAutoregressiveSpatialConvActorCritic
)


def build_actor_critic(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    architecture: str = "flat",
) -> PolicyModel:
    if architecture == "flat":
        return ActorCritic(state_dim, action_dim, hidden_dim)
    if architecture == "spatial":
        return SpatialActorCritic(state_dim, action_dim, hidden_dim)
    if architecture == "spatial-conv":
        return SpatialConvActorCritic(state_dim, action_dim, hidden_dim)
    if architecture == "portable-autoregressive":
        return AutoregressiveSpatialConvActorCritic(
            state_dim, action_dim, hidden_dim
        )
    if architecture == "portable-vdn-autoregressive":
        return VDNAutoregressiveSpatialConvActorCritic(
            state_dim, action_dim, hidden_dim
        )
    raise ValueError(f"Unsupported policy architecture: {architecture}")


class RandomNetworkDistillation(nn.Module):
    """Fixed random target plus learned predictor for state novelty."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.target.requires_grad_(False)

    def novelty(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target = self.target(state)
        prediction = self.predictor(state)
        return (prediction - target).square().mean(dim=-1)


class RideNetwork(nn.Module):
    """Learn controllable state change through an inverse-dynamics objective."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.inverse = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def impact(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        current = self.encoder(state)
        following = self.encoder(next_state)
        return torch.linalg.vector_norm(following - current, dim=-1)

    def inverse_loss(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        current = self.encoder(state)
        following = self.encoder(next_state)
        logits = self.inverse(torch.cat([current, following], dim=-1))
        return F.cross_entropy(logits, action)


@dataclass(slots=True)
class PPOBatch:
    state: torch.Tensor
    action: torch.Tensor
    old_log_probability: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor
    legal_mask: torch.Tensor | None = None


@dataclass(slots=True)
class VDNPPOBatch:
    state: torch.Tensor
    action: torch.Tensor
    old_factor_log_probability: torch.Tensor
    factor_advantage: torch.Tensor
    behavior_factor_probability: torch.Tensor
    value_target: torch.Tensor
    legal_mask: torch.Tensor


@dataclass(slots=True)
class GroupPolicyBatch:
    state: torch.Tensor
    action: torch.Tensor
    old_log_probability: torch.Tensor
    behavior_log_probability: torch.Tensor
    reference_log_probability: torch.Tensor
    advantage: torch.Tensor
    trajectory_weight: torch.Tensor
    legal_mask: torch.Tensor


def ppo_loss(
    model: PolicyModel,
    batch: PPOBatch,
    clip_ratio: float = 0.2,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    logits, value = model(batch.state)
    if batch.legal_mask is not None:
        logits = logits.masked_fill(~batch.legal_mask, -torch.inf)
    distribution = torch.distributions.Categorical(logits=logits)
    log_probability = distribution.log_prob(batch.action)
    ratio = (log_probability - batch.old_log_probability).exp()
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    policy_loss = -torch.min(ratio * batch.advantage, clipped * batch.advantage).mean()
    value_loss = F.mse_loss(value, batch.returns)
    entropy = distribution.entropy().mean()
    total = policy_loss + value_weight * value_loss - entropy_weight * entropy
    return {"loss": total, "policy": policy_loss, "value": value_loss, "entropy": entropy}


def vdn_ppo_loss(
    model: VDNAutoregressiveSpatialConvActorCritic,
    batch: VDNPPOBatch,
    clip_ratio: float = 0.2,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    """VDN-PPO with frozen behavior centering and conditional factor masks."""

    logits, value = model(batch.state)
    factor_probability = model.factor_distribution(logits, batch.legal_mask)
    factor_log_probability = model.selected_factor_log_probabilities(
        factor_probability, batch.action
    )
    active = model.active_factors[batch.action]
    ratio = (factor_log_probability - batch.old_factor_log_probability).exp()
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = torch.minimum(
        ratio * batch.factor_advantage,
        clipped * batch.factor_advantage,
    )
    policy_loss = -surrogate.masked_select(active).mean()

    centered = model.centered_factor_advantages(
        batch.state, batch.behavior_factor_probability
    )
    selected_advantage = model.selected_factor_values(centered, batch.action)
    joint_advantage = (selected_advantage * active.to(selected_advantage.dtype)).sum(
        dim=-1
    )
    value_loss = F.mse_loss(value + joint_advantage, batch.value_target)

    offset = 0
    factor_entropies = []
    for factor, size in enumerate(model.factor_sizes):
        probability = factor_probability[..., offset : offset + size]
        factor_entropy = -(probability * probability.clamp_min(1e-12).log()).sum(
            dim=-1
        )
        factor_entropies.append(factor_entropy)
        offset += size
    entropy_by_factor = torch.stack(factor_entropies, dim=-1)
    entropy = entropy_by_factor.masked_select(active).mean()
    total = policy_loss + value_weight * value_loss - entropy_weight * entropy
    return {"loss": total, "policy": policy_loss, "value": value_loss, "entropy": entropy}


def group_policy_loss(
    model: PolicyModel,
    batch: GroupPolicyBatch,
    *,
    algorithm: str,
    clip_ratio: float = 0.2,
    entropy_weight: float = 0.01,
    reference_kl_weight: float = 0.01,
    importance_ratio_cap: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Masked REINFORCE or action-level GRPO with equal trajectory mass."""

    if algorithm not in {"reinforce", "grpo"}:
        raise ValueError("algorithm must be reinforce or grpo")
    if importance_ratio_cap < 1.0:
        raise ValueError("importance_ratio_cap must be at least one")
    logits, _ = model(batch.state)
    logits = logits.masked_fill(~batch.legal_mask, -torch.inf)
    distribution = torch.distributions.Categorical(logits=logits)
    log_probability = distribution.log_prob(batch.action)
    if algorithm == "grpo":
        ratio = (log_probability - batch.old_log_probability).exp()
        clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
        importance = (
            batch.old_log_probability - batch.behavior_log_probability
        ).exp().clamp(max=importance_ratio_cap)
        surrogate = importance * torch.min(
            ratio * batch.advantage, clipped * batch.advantage
        )
        # Schulman's nonnegative k3 estimator using sampled actions. This is the
        # estimator used by GRPO implementations when full-vocabulary KL is costly.
        reference_gap = batch.reference_log_probability - log_probability
        kl = reference_gap.exp() - reference_gap - 1.0
    else:
        surrogate = log_probability * batch.advantage
        kl = torch.zeros_like(surrogate)
    denominator = batch.trajectory_weight.sum().clamp_min(1e-8)
    policy_loss = -(batch.trajectory_weight * surrogate).sum() / denominator
    entropy = (batch.trajectory_weight * distribution.entropy()).sum() / denominator
    kl_loss = (batch.trajectory_weight * kl).sum() / denominator
    total = policy_loss + reference_kl_weight * kl_loss - entropy_weight * entropy
    return {
        "loss": total,
        "policy": policy_loss,
        "entropy": entropy,
        "reference_kl": kl_loss,
    }

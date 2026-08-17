from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pvzwm.domain import PLANT_COSTS, ActionCodec, GameConfig
from pvzwm.models.dynamics import DeterministicDynamics, DynamicsEnsemble


@dataclass(frozen=True, slots=True)
class PlanResult:
    action: int
    score: float
    sequence: tuple[int, ...]


class CategoricalCEMPlanner:
    """Cross-entropy method for PvZ's large categorical action space."""

    def __init__(
        self,
        model: DeterministicDynamics | DynamicsEnsemble,
        action_dim: int,
        horizon: int = 8,
        samples: int = 256,
        elites: int = 32,
        iterations: int = 4,
        discount: float = 0.99,
        uncertainty_penalty: float = 0.1,
        illegal_action_penalty: float = 0.0,
        elite_prior_strength: float = 1.0,
        config: GameConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        if not 0 < elites <= samples:
            raise ValueError("elites must be in (0, samples]")
        if horizon < 1 or iterations < 1:
            raise ValueError("horizon and iterations must be positive")
        if elite_prior_strength <= 0.0:
            raise ValueError("elite_prior_strength must be positive")
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.samples = samples
        self.elites = elites
        self.iterations = iterations
        self.discount = discount
        self.uncertainty_penalty = uncertainty_penalty
        self.illegal_action_penalty = illegal_action_penalty
        self.elite_prior_strength = elite_prior_strength
        self.config = config or GameConfig()
        if not model.predict_legality and ActionCodec(self.config).size != action_dim:
            raise ValueError(
                "Non-clean-room planning requires learned legality dynamics"
            )
        self.device = device or next(model.parameters()).device

    @torch.inference_mode()
    def plan(
        self,
        state: np.ndarray | torch.Tensor,
        legal_mask: np.ndarray | None = None,
        support_mask: np.ndarray | None = None,
    ) -> PlanResult:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).flatten()
        logits = torch.zeros(self.horizon, self.action_dim, device=self.device)
        support = (
            torch.ones(self.action_dim, dtype=torch.bool, device=self.device)
            if support_mask is None
            else torch.as_tensor(support_mask, dtype=torch.bool, device=self.device)
        )
        logits[:, ~support] = -torch.inf
        if legal_mask is not None:
            legal = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device)
            logits[0, ~legal] = -torch.inf

        best_sequence = torch.zeros(self.horizon, dtype=torch.long, device=self.device)
        best_score = -torch.inf
        for _ in range(self.iterations):
            distribution = torch.distributions.Categorical(logits=logits)
            sequences = distribution.sample((self.samples,))
            scores = self._score(state_tensor, sequences, support)
            elite_indices = scores.topk(self.elites).indices
            elite_sequences = sequences[elite_indices]
            iteration_best = elite_indices[0]
            if scores[iteration_best] > best_score:
                best_score = scores[iteration_best]
                best_sequence = sequences[iteration_best]

            counts = torch.zeros_like(logits)
            counts.scatter_add_(1, elite_sequences.transpose(0, 1), torch.ones_like(elite_sequences.T, dtype=counts.dtype))
            # Keep a fixed amount of prior mass per time step. The previous
            # 0.5-per-action pseudocount contributed 325 virtual samples in the
            # authentic 650-action vocabulary and overwhelmed 32 real elites.
            active = torch.isfinite(logits)
            prior = (
                self.elite_prior_strength
                / active.sum(dim=1, keepdim=True).clamp_min(1)
            )
            logits = (counts + prior * active).log()
            logits[:, ~support] = -torch.inf
            if legal_mask is not None:
                logits[0, ~legal] = -torch.inf

        sequence = tuple(int(value) for value in best_sequence.cpu().tolist())
        return PlanResult(sequence[0], float(best_score.cpu()), sequence)

    def _score(
        self,
        initial_state: torch.Tensor,
        sequences: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        states = initial_state.unsqueeze(0).expand(self.samples, -1)
        returns = torch.zeros(self.samples, device=self.device)
        alive = torch.ones(self.samples, device=self.device)
        discount = 1.0
        for time_index in range(self.horizon):
            proposed = sequences[:, time_index]
            legal = self._predicted_legal_mask(states) & support_mask.unsqueeze(0)
            valid = legal.gather(1, proposed.unsqueeze(1)).squeeze(1)
            actions = torch.where(valid, proposed, torch.zeros_like(proposed))
            if self.illegal_action_penalty:
                returns -= (
                    discount
                    * alive
                    * self.illegal_action_penalty
                    * (~valid).to(returns.dtype)
                )
            prediction = self.model(states, actions)
            returns += discount * alive * prediction.reward
            if prediction.variance is not None:
                uncertainty = prediction.variance.mean(dim=-1).sqrt()
                returns -= discount * alive * self.uncertainty_penalty * uncertainty
            alive = alive * (1.0 - torch.sigmoid(prediction.done_logit))
            states = prediction.next_state
            discount *= self.discount
        return returns

    def _predicted_legal_mask(self, states: torch.Tensor) -> torch.Tensor:
        if self.model.predict_legality:
            mask = self.model.legal_mask(states)
            mask[:, 0] = True
            return mask
        cells = self.config.rows * self.config.cols
        occupied = torch.round(states[:, :cells].clamp(0.0, 1.0) * 3.0) != 0
        sun = states[:, 4 * cells].clamp(0.0, 1.0) * 1_000.0
        cooldown_start = 4 * cells + 4 + self.config.rows
        cooldowns = states[:, cooldown_start : cooldown_start + 3].clamp(0.0, 1.0)
        mask = torch.zeros(
            states.shape[0], self.action_dim, dtype=torch.bool, device=states.device
        )
        mask[:, 0] = True
        for kind_index, cost in enumerate(PLANT_COSTS.values()):
            start = 1 + kind_index * cells
            mask[:, start : start + cells] = (
                (~occupied)
                & (sun >= cost).unsqueeze(1)
                & (cooldowns[:, kind_index] <= 0.05).unsqueeze(1)
            )
        shovel_start = 1 + 3 * cells
        mask[:, shovel_start : shovel_start + cells] = occupied
        return mask

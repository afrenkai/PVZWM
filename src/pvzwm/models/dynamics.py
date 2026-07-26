from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class DynamicsPrediction:
    next_state: torch.Tensor
    reward: torch.Tensor
    done_logit: torch.Tensor
    variance: torch.Tensor | None = None
    legal_logits: torch.Tensor | None = None


class DeterministicDynamics(nn.Module):
    """Residual state dynamics with reward and termination heads."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_embed_dim: int = 64,
        predict_legality: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.predict_legality = predict_legality
        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.state_head = nn.Linear(hidden_dim, state_dim)
        nn.init.zeros_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        self.legal_head = (
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            if predict_legality
            else None
        )

    def legal_logits(self, state: torch.Tensor) -> torch.Tensor:
        if self.legal_head is None:
            raise RuntimeError("Dynamics model was not trained to predict legality")
        return self.legal_head(state)

    def legal_mask(self, state: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return self.legal_logits(state).sigmoid() >= threshold

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> DynamicsPrediction:
        features = self.trunk(torch.cat((state, self.action_embedding(action)), dim=-1))
        next_state = state + self.state_head(features)
        return DynamicsPrediction(
            next_state,
            self.reward_head(features).squeeze(-1),
            self.done_head(features).squeeze(-1),
            legal_logits=(
                self.legal_logits(state) if self.legal_head is not None else None
            ),
        )

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        reward_weight: float = 1.0,
        done_weight: float = 0.25,
        legal_weight: float = 0.25,
        state_weight: float = 1.0,
        state_change_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if state.ndim == 3:
            return self.rollout_loss(
                state,
                action,
                reward,
                next_state,
                done,
                legal_mask,
                reward_weight=reward_weight,
                done_weight=done_weight,
                legal_weight=legal_weight,
                state_weight=state_weight,
                state_change_weight=state_change_weight,
            )
        prediction = self(state, action)
        components = self._prediction_loss(
            prediction,
            state,
            reward,
            next_state,
            done,
            legal_mask,
            state_change_weight=state_change_weight,
        )
        return self._weighted_loss(
            components,
            reward_weight=reward_weight,
            done_weight=done_weight,
            legal_weight=legal_weight,
            state_weight=state_weight,
        )

    def rollout_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        reward_weight: float = 1.0,
        done_weight: float = 0.25,
        legal_weight: float = 0.25,
        state_weight: float = 1.0,
        state_change_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Train open-loop rollouts without crossing replay episode boundaries."""
        if state.ndim != 3 or next_state.shape != state.shape:
            raise ValueError("Rollout dynamics require [batch, time, state] tensors")
        if action.shape != state.shape[:2]:
            raise ValueError("Rollout actions must match the batch and time dimensions")
        current = state[:, 0]
        step_components: list[dict[str, torch.Tensor]] = []
        for index in range(state.shape[1]):
            prediction = self(current, action[:, index])
            step_components.append(
                self._prediction_loss(
                    prediction,
                    state[:, index],
                    reward[:, index],
                    next_state[:, index],
                    done[:, index],
                    None if legal_mask is None else legal_mask[:, index],
                    state_change_weight=state_change_weight,
                )
            )
            current = prediction.next_state
        components = {
            name: torch.stack([step[name] for step in step_components]).mean()
            for name in step_components[0]
        }
        return self._weighted_loss(
            components,
            reward_weight=reward_weight,
            done_weight=done_weight,
            legal_weight=legal_weight,
            state_weight=state_weight,
        )

    def _prediction_loss(
        self,
        prediction: DynamicsPrediction,
        state_reference: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        legal_mask: torch.Tensor | None,
        *,
        state_change_weight: float,
    ) -> dict[str, torch.Tensor]:
        if state_change_weight < 0.0:
            raise ValueError("State loss weights cannot be negative")
        state_error = F.smooth_l1_loss(
            prediction.next_state, next_state, reduction="none"
        )
        changed = (next_state - state_reference).abs() > 1e-6
        feature_weights = 1.0 + state_change_weight * changed.to(state_error.dtype)
        state_loss = (state_error * feature_weights).sum() / feature_weights.sum()
        reward_loss = F.smooth_l1_loss(prediction.reward, reward)
        done_loss = F.binary_cross_entropy_with_logits(prediction.done_logit, done)
        legal_loss = state_loss.new_zeros(())
        if self.legal_head is not None:
            if legal_mask is None or prediction.legal_logits is None:
                raise ValueError("Legality-aware dynamics require replay legal masks")
            legal_loss = F.binary_cross_entropy_with_logits(
                prediction.legal_logits, legal_mask.to(prediction.legal_logits.dtype)
            )
        return {
            "state": state_loss,
            "reward": reward_loss,
            "done": done_loss,
            "legal": legal_loss,
            "changed_fraction": changed.to(state_error.dtype).mean(),
        }

    @staticmethod
    def _weighted_loss(
        components: dict[str, torch.Tensor],
        *,
        reward_weight: float,
        done_weight: float,
        legal_weight: float,
        state_weight: float,
    ) -> dict[str, torch.Tensor]:
        if state_weight < 0.0:
            raise ValueError("State loss weights cannot be negative")
        total = (
            state_weight * components["state"]
            + reward_weight * components["reward"]
            + done_weight * components["done"]
            + legal_weight * components["legal"]
        )
        return {"loss": total, **components}


class DynamicsEnsemble(nn.Module):
    """Deep ensemble exposing epistemic variance for conservative planning."""

    def __init__(
        self,
        size: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        predict_legality: bool = False,
    ) -> None:
        super().__init__()
        if size < 2:
            raise ValueError("An uncertainty ensemble requires at least two members")
        self.members = nn.ModuleList(
            DeterministicDynamics(
                state_dim,
                action_dim,
                hidden_dim,
                predict_legality=predict_legality,
            )
            for _ in range(size)
        )

    @property
    def predict_legality(self) -> bool:
        return self.members[0].predict_legality

    def legal_logits(self, state: torch.Tensor) -> torch.Tensor:
        return torch.stack([member.legal_logits(state) for member in self.members]).mean(0)

    def legal_mask(self, state: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return self.legal_logits(state).sigmoid() >= threshold

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> DynamicsPrediction:
        predictions = [member(state, action) for member in self.members]
        states = torch.stack([prediction.next_state for prediction in predictions])
        rewards = torch.stack([prediction.reward for prediction in predictions])
        dones = torch.stack([prediction.done_logit for prediction in predictions])
        legal_predictions = [
            prediction.legal_logits
            for prediction in predictions
            if prediction.legal_logits is not None
        ]
        return DynamicsPrediction(
            states.mean(0),
            rewards.mean(0),
            dones.mean(0),
            states.var(0, unbiased=False),
            (
                torch.stack(legal_predictions).mean(0)
                if legal_predictions
                else None
            ),
        )

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        reward_weight: float = 1.0,
        done_weight: float = 0.25,
        legal_weight: float = 0.25,
        state_weight: float = 1.0,
        state_change_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Train each member on an independent bootstrap of the batch."""
        member_losses: list[dict[str, torch.Tensor]] = []
        for member in self.members:
            indices = torch.randint(
                state.shape[0], (state.shape[0],), device=state.device
            )
            member_losses.append(
                member.loss(
                    state[indices],
                    action[indices],
                    reward[indices],
                    next_state[indices],
                    done[indices],
                    None if legal_mask is None else legal_mask[indices],
                    reward_weight=reward_weight,
                    done_weight=done_weight,
                    legal_weight=legal_weight,
                    state_weight=state_weight,
                    state_change_weight=state_change_weight,
                )
            )
        return {
            name: torch.stack([losses[name] for losses in member_losses]).mean()
            for name in member_losses[0]
        }

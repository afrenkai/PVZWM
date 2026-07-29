from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pvzwm.models.regularizers import sigreg_objective
from pvzwm.models.visreg import visreg_regularizer


def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return value * (1.0 + scale) + shift


class _ConditionalCausalBlock(nn.Module):
    """Causal self-attention with zero-initialized action-conditioned gates."""

    def __init__(
        self,
        latent_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            latent_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 6 * latent_dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.modulation(condition).chunk(6, dim=-1)
        length = value.shape[1]
        causal_mask = torch.ones(
            length,
            length,
            dtype=torch.bool,
            device=value.device,
        ).triu(diagonal=1)
        attention_input = _modulate(
            self.norm_attention(value), attention_shift, attention_scale
        )
        attended, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            attn_mask=causal_mask,
            need_weights=False,
        )
        value = value + attention_gate * attended
        mlp_input = _modulate(self.norm_mlp(value), mlp_shift, mlp_scale)
        return value + mlp_gate * self.mlp(mlp_input)


@dataclass(slots=True)
class LeWMPrediction:
    embedding: torch.Tensor
    reward: torch.Tensor
    done_logit: torch.Tensor


class ActionConditionedLeWorldModel(nn.Module):
    """Causal LeWM predictor with VISReg by default and SIGReg as an ablation."""

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        *,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_history: int = 16,
        projection_head: bool = False,
        prediction_projection_head: bool = False,
        projection_hidden_dim: int | None = None,
        projection_depth: int = 2,
        residual_prediction: bool = False,
    ) -> None:
        super().__init__()
        if latent_dim % num_heads:
            raise ValueError("LeWM latent dimension must be divisible by attention heads")
        if num_layers < 1 or max_history < 1:
            raise ValueError("LeWM layers and maximum history must be positive")
        self.encoder = encoder
        self._representation_frozen = False
        self.residual_prediction = residual_prediction
        projector_hidden_dim = (
            hidden_dim if projection_hidden_dim is None else projection_hidden_dim
        )
        if projector_hidden_dim < 1:
            raise ValueError("LeWM projection hidden dimension must be positive")
        if projection_depth not in {2, 3}:
            raise ValueError("LeWM projection depth must be two or three")

        def projection_mlp() -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Linear(latent_dim, projector_hidden_dim),
                nn.BatchNorm1d(projector_hidden_dim),
                nn.GELU(),
            ]
            if projection_depth == 3:
                layers.extend(
                    [
                        nn.Linear(projector_hidden_dim, projector_hidden_dim),
                        nn.BatchNorm1d(projector_hidden_dim),
                        nn.GELU(),
                    ]
                )
            layers.append(nn.Linear(projector_hidden_dim, latent_dim))
            return nn.Sequential(*layers)

        self.projector: nn.Module | None = (
            projection_mlp() if projection_head else None
        )
        self.prediction_projector: nn.Module | None = (
            projection_mlp() if prediction_projection_head else None
        )
        self.action_embedding = nn.Embedding(action_dim, latent_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_history, latent_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                _ConditionalCausalBlock(
                    latent_dim,
                    num_heads,
                    hidden_dim,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        if residual_prediction:
            final_projection = self.output_projection[-1]
            assert isinstance(final_projection, nn.Linear)
            nn.init.zeros_(final_projection.weight)
            nn.init.zeros_(final_projection.bias)
        self.reward_probe = nn.Linear(latent_dim, 1)
        self.done_probe = nn.Linear(latent_dim, 1)
        self.max_history = max_history

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        leading_shape = observation.shape[:-3]
        flattened = observation.reshape(-1, *observation.shape[-3:])
        embedding = self.encoder(flattened)
        if self.projector is not None:
            embedding = self.projector(embedding)
        return embedding.reshape(*leading_shape, embedding.shape[-1])

    def predict(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
    ) -> LeWMPrediction:
        if embeddings.ndim != 3 or actions.ndim != 2:
            raise ValueError("LeWM prediction expects [batch, time, latent] and actions")
        if embeddings.shape[:2] != actions.shape or embeddings.shape[1] > self.max_history:
            raise ValueError("LeWM embedding/action history is inconsistent or too long")
        value = embeddings + self.position_embedding[:, : embeddings.shape[1]]
        condition = self.action_embedding(actions)
        for block in self.blocks:
            value = block(value, condition)
        predicted = self.output_projection(self.output_norm(value))
        if self.prediction_projector is not None:
            predicted = self.prediction_projector(
                predicted.reshape(-1, predicted.shape[-1])
            ).reshape_as(predicted)
        if self.residual_prediction:
            predicted = embeddings + predicted
        # These probes model task signals without changing the two-term JEPA
        # representation objective through their gradients.
        probe_input = predicted.detach()
        return LeWMPrediction(
            embedding=predicted,
            reward=self.reward_probe(probe_input).squeeze(-1),
            done_logit=self.done_probe(probe_input).squeeze(-1),
        )

    def load_representation_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        keys = [
            key
            for key in current
            if key.startswith("encoder.") or key.startswith("projector.")
        ]
        missing = [key for key in keys if key not in state_dict]
        mismatched = [
            key
            for key in keys
            if key in state_dict and current[key].shape != state_dict[key].shape
        ]
        if missing or mismatched:
            raise ValueError(
                "LeWM representation checkpoint is incompatible; "
                f"missing={missing}, mismatched={mismatched}"
            )
        current.update({key: state_dict[key] for key in keys})
        self.load_state_dict(current)

    def freeze_representation(self) -> None:
        self._representation_frozen = True
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        if self.projector is not None:
            self.projector.requires_grad_(False)
            self.projector.eval()

    def train(self, mode: bool = True) -> ActionConditionedLeWorldModel:
        super().train(mode)
        if self._representation_frozen:
            self.encoder.eval()
            if self.projector is not None:
                self.projector.eval()
        return self

    def loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
        *,
        regularizer: str = "visreg",
        regularization_weight: float = 0.05,
        slices: int = 1024,
        reward_weight: float = 1.0,
        done_weight: float = 0.25,
        loss_mixing: str = "convex",
    ) -> dict[str, torch.Tensor]:
        if not 0.0 <= regularization_weight <= 1.0:
            raise ValueError("regularization_weight must be in [0, 1]")
        source = self.encode(observation)
        target = self.encode(next_observation)
        prediction = self.predict(source, action)
        if regularizer == "visreg":
            prediction_loss = F.mse_loss(prediction.embedding, target)
            # Consecutive source/target times play the role of multiple views;
            # each view is regularized across independent replay windows, as in
            # the released VISReg [views, batch, features] implementation.
            # A transition sequence stores [s0..s(T-1)] and [s1..sT]. Keep
            # the T+1 unique states instead of double-counting middle frames.
            representation_views = torch.cat(
                (source[:, :1], target), dim=1
            ).transpose(0, 1)
            regularization = visreg_regularizer(
                representation_views, slices=slices
            )
            if loss_mixing == "convex":
                latent_loss = (
                    (1.0 - regularization_weight) * prediction_loss
                    + regularization_weight * regularization.total
                )
            elif loss_mixing == "additive":
                latent_loss = (
                    prediction_loss + regularization_weight * regularization.total
                )
            else:
                raise ValueError("LeWM loss mixing must be convex or additive")
            metrics = {
                "prediction": prediction_loss,
                "visreg": regularization.total,
                "center": regularization.center,
                "scale": regularization.scale,
                "shape": regularization.shape,
            }
        elif regularizer == "sigreg":
            latent_loss, metrics = sigreg_objective(
                prediction.embedding,
                target,
                regularization_weight=regularization_weight,
                slices=slices,
            )
            if loss_mixing == "additive":
                latent_loss = (
                    metrics["prediction"]
                    + regularization_weight * metrics["sigreg"]
                )
            elif loss_mixing != "convex":
                raise ValueError("LeWM loss mixing must be convex or additive")
        else:
            raise ValueError("LeWM regularizer must be visreg or sigreg")
        reward_loss = F.smooth_l1_loss(prediction.reward, reward)
        done_loss = F.binary_cross_entropy_with_logits(prediction.done_logit, done)
        total = latent_loss + reward_weight * reward_loss + done_weight * done_loss
        return {
            **metrics,
            "loss": total,
            "reward": reward_loss,
            "done": done_loss,
        }

    def next(
        self,
        embedding_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> LeWMPrediction:
        history = min(self.max_history, embedding_history.shape[1])
        prediction = self.predict(
            embedding_history[:, -history:],
            action_history[:, -history:],
        )
        return LeWMPrediction(
            prediction.embedding[:, -1],
            prediction.reward[:, -1],
            prediction.done_logit[:, -1],
        )

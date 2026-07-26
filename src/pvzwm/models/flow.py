from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pvzwm.models.token import PixelTokenizerBackbone
from pvzwm.models.visreg import visreg_regularizer


@dataclass(slots=True)
class FlowRolloutStep:
    observation: torch.Tensor
    reward: torch.Tensor
    done_logit: torch.Tensor
    latent: torch.Tensor


class RepresentationFlowWorldModel(nn.Module):
    """Compact MIRA-style representation autoencoder and latent flow model.

    The encoder is trained from scratch. A flow-matching Transformer predicts
    the next latent grid from the current grid and action, while the decoder
    maps imagined latents back to pixels.
    """

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_dim: int,
        code_dim: int = 32,
        latent_height: int = 3,
        latent_width: int = 5,
        autoencoder_hidden_dim: int = 64,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if action_dim < 1 or model_dim % num_heads:
            raise ValueError("Action dimension must be positive and model_dim divisible by heads")
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.code_dim = code_dim
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.latent_tokens = latent_height * latent_width
        self.autoencoder = PixelTokenizerBackbone(
            observation_shape,
            code_dim,
            latent_height,
            latent_width,
            autoencoder_hidden_dim,
        )
        self.input_projection = nn.Linear(2 * code_dim, model_dim)
        self.action_embedding = nn.Embedding(action_dim, model_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.position_embedding = nn.Embedding(self.latent_tokens, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.velocity_head = nn.Linear(model_dim, code_dim)
        self.outcome_trunk = nn.Sequential(
            nn.Linear(code_dim + model_dim, model_dim), nn.GELU()
        )
        self.reward_head = nn.Linear(model_dim, 1)
        self.done_head = nn.Linear(model_dim, 1)
        self.legal_head = nn.Sequential(
            nn.Linear(code_dim, model_dim), nn.GELU(), nn.Linear(model_dim, action_dim)
        )

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        latent, _ = self.autoencoder.encode_features(observation)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode_quantized(latent)

    def _tokens(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.flatten(-2).transpose(-1, -2)

    def _grid(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.transpose(-1, -2).reshape(
            len(tokens), self.code_dim, self.latent_height, self.latent_width
        )

    def velocity(
        self,
        noisy_target: torch.Tensor,
        source: torch.Tensor,
        action: torch.Tensor,
        time: torch.Tensor,
        action_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noisy_tokens = self._tokens(noisy_target)
        source_tokens = self._tokens(source)
        hidden = self.input_projection(torch.cat((noisy_tokens, source_tokens), dim=-1))
        positions = torch.arange(self.latent_tokens, device=hidden.device)
        action_condition = self.action_embedding(action)
        if action_drop_mask is not None:
            if action_drop_mask.shape != action.shape:
                raise ValueError("Action drop mask must align with the action batch")
            action_condition = action_condition.masked_fill(
                action_drop_mask[:, None], 0.0
            )
        condition = action_condition + self.time_embedding(time[:, None])
        hidden = hidden + self.position_embedding(positions) + condition[:, None, :]
        return self._grid(self.velocity_head(self.transformer(hidden)))

    def outcomes(
        self, source: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = source.mean(dim=(-2, -1))
        hidden = self.outcome_trunk(
            torch.cat((pooled, self.action_embedding(action)), dim=-1)
        )
        return self.reward_head(hidden).squeeze(-1), self.done_head(hidden).squeeze(-1)

    def legal_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return self.legal_head(latent.mean(dim=(-2, -1)))

    def legal_mask(self, latent: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        mask = self.legal_logits(latent).sigmoid() >= threshold
        mask[..., 0] = True
        return mask

    @torch.inference_mode()
    def imagine_latent(
        self,
        source: torch.Tensor,
        action: torch.Tensor,
        steps: int = 4,
        stochastic: bool = False,
    ) -> FlowRolloutStep:
        if steps < 1:
            raise ValueError("Flow integration steps must be positive")
        latent = torch.randn_like(source) if stochastic else torch.zeros_like(source)
        step_size = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (len(source),), (index + 0.5) * step_size, device=source.device
            )
            latent = latent + step_size * self.velocity(latent, source, action, time)
        reward, done_logit = self.outcomes(source, action)
        return FlowRolloutStep(self.decode(latent), reward, done_logit, latent)

    @torch.inference_mode()
    def generate_next(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        steps: int = 4,
        stochastic: bool = False,
    ) -> FlowRolloutStep:
        source = self.encode(observation)
        return self.imagine_latent(source, action, steps=steps, stochastic=stochastic)

    def loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        reconstruction_weight: float = 1.0,
        flow_weight: float = 1.0,
        visreg_weight: float = 0.01,
        visreg_slices: int = 64,
        reward_weight: float = 1.0,
        done_weight: float = 0.25,
        legal_weight: float = 0.25,
        action_dropout_probability: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        if not 0.0 <= action_dropout_probability <= 1.0:
            raise ValueError("Action dropout probability must be in [0, 1]")
        source = self.encode(observation)
        target = self.encode(next_observation)
        reconstructed_source = self.decode(source)
        reconstructed_target = self.decode(target)
        reconstruction = 0.5 * (
            F.smooth_l1_loss(reconstructed_source, observation)
            + F.smooth_l1_loss(reconstructed_target, next_observation)
        )
        representation_samples = torch.cat((source, target), dim=0).movedim(1, -1)
        regularization = visreg_regularizer(
            representation_samples, slices=visreg_slices
        )

        detached_source = source.detach()
        detached_target = target.detach()
        noise = torch.randn_like(detached_target)
        time = torch.rand(len(target), device=target.device, dtype=target.dtype)
        interpolated = (
            (1.0 - time[:, None, None, None]) * noise
            + time[:, None, None, None] * detached_target
        )
        target_velocity = detached_target - noise
        action_drop_mask = torch.rand(len(action), device=action.device) < action_dropout_probability
        predicted_velocity = self.velocity(
            interpolated,
            detached_source,
            action,
            time,
            action_drop_mask=action_drop_mask,
        )
        flow = F.mse_loss(predicted_velocity, target_velocity)
        predicted_reward, predicted_done = self.outcomes(source, action)
        reward_loss = F.smooth_l1_loss(predicted_reward, reward)
        done_loss = F.binary_cross_entropy_with_logits(predicted_done, done)
        if legal_mask is None:
            legal_loss = source.new_zeros(())
        else:
            legal_targets = legal_mask.to(source.dtype)
            legal_logits = self.legal_logits(source)
            positive_count = legal_targets.sum().clamp_min(1.0)
            negative_targets = 1.0 - legal_targets
            negative_count = negative_targets.sum().clamp_min(1.0)
            positive_loss = (F.softplus(-legal_logits) * legal_targets).sum()
            negative_loss = (F.softplus(legal_logits) * negative_targets).sum()
            # Most PvZ actions are illegal at any given instant. Equal class
            # weighting prevents a superficially accurate WAIT-only classifier.
            legal_loss = 0.5 * (
                positive_loss / positive_count + negative_loss / negative_count
            )
        total = (
            reconstruction_weight * reconstruction
            + flow_weight * flow
            + visreg_weight * regularization.total
            + reward_weight * reward_loss
            + done_weight * done_loss
            + legal_weight * legal_loss
        )
        return {
            "loss": total,
            "reconstruction": reconstruction,
            "flow": flow,
            "visreg": regularization.total,
            "reward": reward_loss,
            "done": done_loss,
            "legal": legal_loss,
        }

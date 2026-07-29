from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class TokenizerOutput:
    encoded: torch.Tensor
    quantized: torch.Tensor
    tokens: torch.Tensor


@dataclass(slots=True)
class TokenRolloutStep:
    context: torch.Tensor
    observation: torch.Tensor
    reward: torch.Tensor
    done_logit: torch.Tensor
    tokens: torch.Tensor


class DivisibleTokenPool(nn.Module):
    """Exact-grid average pooling that avoids MPS adaptive-pool restrictions."""

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.output_size = (height, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output_height, output_width = self.output_size
        height, width = value.shape[-2:]
        target_height = ((height + output_height - 1) // output_height) * output_height
        target_width = ((width + output_width - 1) // output_width) * output_width
        if (target_height, target_width) != (height, width):
            value = F.pad(
                value,
                (0, target_width - width, 0, target_height - height),
                mode="replicate",
            )
        kernel = (target_height // output_height, target_width // output_width)
        return F.avg_pool2d(value, kernel_size=kernel, stride=kernel)


class PixelTokenizerBackbone(nn.Module):
    """Shared convolutional encoder and decoder for discrete pixel tokenizers."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        code_dim: int = 64,
        token_height: int = 3,
        token_width: int = 5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        channels, image_height, image_width = observation_shape
        if min(observation_shape) < 1:
            raise ValueError("Observation dimensions must be positive")
        if code_dim < 1:
            raise ValueError("Tokenizer code dimension must be positive")
        if token_height < 1 or token_width < 1:
            raise ValueError("Token grid dimensions must be positive")
        if hidden_dim % 8:
            raise ValueError("Tokenizer hidden dimension must be divisible by eight")
        self.observation_shape = observation_shape
        self.code_dim = code_dim
        self.token_height = token_height
        self.token_width = token_width
        self.tokens_per_frame = token_height * token_width
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim // 2, 4, stride=2, padding=1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            DivisibleTokenPool(token_height, token_width),
            nn.Conv2d(hidden_dim, code_dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(code_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Upsample(size=(image_height, image_width), mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 5, padding=2),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, channels, 1),
            nn.Sigmoid(),
        )

    def encode_features(self, observation: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        leading_shape = observation.shape[:-3]
        flat = observation.reshape(-1, *self.observation_shape)
        encoded = self.encoder(flat)
        encoded = encoded.reshape(
            *leading_shape, self.code_dim, self.token_height, self.token_width
        )
        return encoded, leading_shape

    def decode_quantized(self, quantized: torch.Tensor) -> torch.Tensor:
        leading_shape = quantized.shape[:-3]
        decoded = self.decoder(
            quantized.reshape(-1, self.code_dim, self.token_height, self.token_width)
        )
        return decoded.reshape(*leading_shape, *self.observation_shape)


class PixelVQTokenizer(PixelTokenizerBackbone):
    """Small learned vector quantizer with a fixed spatial token grid."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        vocab_size: int = 256,
        code_dim: int = 64,
        token_height: int = 3,
        token_width: int = 5,
        hidden_dim: int = 64,
    ) -> None:
        if vocab_size < 2:
            raise ValueError("The VQ codebook requires at least two entries")
        super().__init__(
            observation_shape, code_dim, token_height, token_width, hidden_dim
        )
        self.vocab_size = vocab_size
        self.codebook = nn.Embedding(vocab_size, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / vocab_size, 1.0 / vocab_size)

    def encode(self, observation: torch.Tensor) -> TokenizerOutput:
        encoded, leading_shape = self.encode_features(observation)
        flat_encoded = encoded.reshape(
            -1, self.code_dim, self.token_height, self.token_width
        )
        vectors = flat_encoded.permute(0, 2, 3, 1).reshape(-1, self.code_dim)
        vectors_float = vectors.float()
        codebook_float = self.codebook.weight.float()
        distances = (
            vectors_float.square().sum(dim=1, keepdim=True)
            + codebook_float.square().sum(dim=1)
            - 2.0 * vectors_float @ codebook_float.t()
        )
        tokens = distances.argmin(dim=-1)
        quantized = self.codebook(tokens).view(
            len(flat_encoded), self.token_height, self.token_width, self.code_dim
        )
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        quantized = quantized.reshape(
            *leading_shape, self.code_dim, self.token_height, self.token_width
        )
        tokens = tokens.reshape(*leading_shape, self.tokens_per_frame)
        return TokenizerOutput(encoded, quantized, tokens)

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        leading_shape = tokens.shape[:-1]
        quantized = self.codebook(tokens).reshape(
            *leading_shape, self.token_height, self.token_width, self.code_dim
        )
        quantized = quantized.movedim(-1, -3).contiguous()
        return self.decode_quantized(quantized)


class PixelFSQTokenizer(PixelTokenizerBackbone):
    """Finite scalar quantizer following the official Google FSQ equations."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        levels: tuple[int, ...] = (8, 8, 4),
        token_height: int = 3,
        token_width: int = 5,
        hidden_dim: int = 64,
        epsilon: float = 1e-3,
    ) -> None:
        if not levels or any(level < 2 for level in levels):
            raise ValueError("Every FSQ dimension needs at least two levels")
        super().__init__(
            observation_shape,
            len(levels),
            token_height,
            token_width,
            hidden_dim,
        )
        self.vocab_size = 1
        for level in levels:
            self.vocab_size *= level
        self.epsilon = epsilon
        level_tensor = torch.tensor(levels, dtype=torch.long)
        basis = torch.ones_like(level_tensor)
        if len(levels) > 1:
            basis[1:] = torch.cumprod(level_tensor[:-1], dim=0)
        self.register_buffer("levels", level_tensor)
        self.register_buffer("basis", basis)

    def _quantize(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = encoded.movedim(-3, -1)
        levels = self.levels.to(device=values.device, dtype=values.dtype)
        half_levels = (levels - 1.0) * (1.0 - self.epsilon) / 2.0
        offset = torch.where(self.levels.remainder(2) == 1, 0.0, 0.5).to(values.dtype)
        shift = torch.tan(offset / half_levels)
        bounded = torch.tanh(values + shift) * half_levels - offset
        rounded = bounded + (bounded.round() - bounded).detach()
        half_width = torch.div(self.levels, 2, rounding_mode="floor").to(values.dtype)
        normalized = rounded / half_width
        digits = (normalized * half_width + half_width).round().to(torch.long)
        digits = torch.minimum(
            torch.maximum(digits, torch.zeros_like(digits)), self.levels - 1
        )
        tokens = (digits * self.basis).sum(dim=-1)
        return normalized.movedim(-1, -3).contiguous(), tokens

    def encode(self, observation: torch.Tensor) -> TokenizerOutput:
        encoded, leading_shape = self.encode_features(observation)
        quantized, tokens = self._quantize(encoded)
        return TokenizerOutput(
            encoded,
            quantized,
            tokens.reshape(*leading_shape, self.tokens_per_frame),
        )

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        digits = torch.div(
            tokens[..., None], self.basis, rounding_mode="floor"
        ).remainder(self.levels)
        half_width = torch.div(self.levels, 2, rounding_mode="floor")
        normalized = (digits - half_width).to(self.decoder[0].weight.dtype) / half_width
        leading_shape = tokens.shape[:-1]
        quantized = normalized.reshape(
            *leading_shape, self.token_height, self.token_width, self.code_dim
        ).movedim(-1, -3).contiguous()
        return self.decode_quantized(quantized)


class CausalTokenWorldModel(nn.Module):
    """IRIS-style discrete tokenizer and action-interleaved causal Transformer."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_dim: int,
        vocab_size: int = 256,
        code_dim: int = 64,
        quantizer: str = "vq",
        fsq_levels: tuple[int, ...] = (8, 8, 4),
        token_height: int = 3,
        token_width: int = 5,
        tokenizer_hidden_dim: int = 64,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_transitions: int = 16,
    ) -> None:
        super().__init__()
        if action_dim < 1 or max_transitions < 1:
            raise ValueError("Action dimension and context length must be positive")
        if model_dim % num_heads:
            raise ValueError("Transformer model dimension must be divisible by its heads")
        self.action_dim = action_dim
        self.max_transitions = max_transitions
        self.quantizer_kind = quantizer
        if quantizer == "vq":
            self.tokenizer: PixelVQTokenizer | PixelFSQTokenizer = PixelVQTokenizer(
                observation_shape,
                vocab_size,
                code_dim,
                token_height,
                token_width,
                tokenizer_hidden_dim,
            )
        elif quantizer == "fsq":
            self.tokenizer = PixelFSQTokenizer(
                observation_shape,
                fsq_levels,
                token_height,
                token_width,
                tokenizer_hidden_dim,
            )
        else:
            raise ValueError(f"Unsupported token quantizer: {quantizer}")
        self.vocab_size = self.tokenizer.vocab_size
        self.tokens_per_frame = self.tokenizer.tokens_per_frame
        self.max_tokens = (
            (max_transitions + 1) * self.tokens_per_frame + max_transitions
        )
        self.embedding = nn.Embedding(self.vocab_size + action_dim, model_dim)
        self.position_embedding = nn.Embedding(self.max_tokens, model_dim)
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
        self.token_head = nn.Linear(model_dim, self.vocab_size)
        self.reward_head = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 1)
        )
        self.done_head = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 1)
        )

    def _hidden(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] > self.max_tokens:
            raise ValueError("Token context has an invalid shape or exceeds the configured limit")
        positions = torch.arange(ids.shape[1], device=ids.device)
        embedded = self.embedding(ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device),
            diagonal=1,
        )
        return self.transformer(embedded, mask=causal_mask)

    def _interleave(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if frames.shape[1] != actions.shape[1] + 1:
            raise ValueError("A token sequence needs one more frame than action")
        action_ids = actions.unsqueeze(-1) + self.vocab_size
        blocks = torch.cat((frames[:, :-1], action_ids), dim=-1).flatten(1)
        return torch.cat((blocks, frames[:, -1]), dim=-1)

    def loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
        tokenizer_weight: float = 1.0,
        codebook_weight: float = 1.0,
        commitment_beta: float = 1.0,
        reward_weight: float = 1.0,
        done_weight: float = 0.25,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        if observation.ndim != 5:
            raise ValueError("Token model loss expects [batch, time, channels, height, width]")
        frames = torch.cat((observation[:, :1], next_observation), dim=1)
        encoded = self.tokenizer.encode(frames)
        straight_through = encoded.encoded + (encoded.quantized - encoded.encoded).detach()
        reconstruction = self.tokenizer.decode_quantized(straight_through)
        reconstruction_loss = F.l1_loss(reconstruction, frames)
        if self.quantizer_kind == "vq":
            codebook_loss = F.mse_loss(
                encoded.quantized, encoded.encoded.detach()
            ) + commitment_beta * F.mse_loss(
                encoded.encoded, encoded.quantized.detach()
            )
        else:
            codebook_loss = encoded.encoded.new_zeros(())

        ids = self._interleave(encoded.tokens.detach(), action)
        hidden = self._hidden(ids)
        next_ids = ids[:, 1:]
        observation_targets = next_ids < self.vocab_size
        token_logits = self.token_head(hidden[:, :-1][observation_targets])
        token_loss = F.cross_entropy(token_logits, next_ids[observation_targets])

        action_positions = ids >= self.vocab_size
        action_hidden = hidden[action_positions]
        predicted_reward = self.reward_head(action_hidden).squeeze(-1)
        predicted_done = self.done_head(action_hidden).squeeze(-1)
        reward_loss = F.smooth_l1_loss(predicted_reward, reward.flatten())
        done_loss = F.binary_cross_entropy_with_logits(predicted_done, done.flatten())

        counts = torch.bincount(
            encoded.tokens.flatten(), minlength=self.vocab_size
        ).to(dtype=torch.float32)
        probabilities = counts / counts.sum().clamp_min(1.0)
        perplexity = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        )
        utilization = (counts > 0).to(dtype=torch.float32).mean()
        total = (
            token_loss
            + tokenizer_weight * (reconstruction_loss + codebook_weight * codebook_loss)
            + reward_weight * reward_loss
            + done_weight * done_loss
        )
        return {
            "loss": total,
            "token_cross_entropy": token_loss,
            "tokenizer_reconstruction": reconstruction_loss,
            "codebook": codebook_loss,
            "codebook_perplexity": perplexity,
            "codebook_utilization": utilization,
            "reward": reward_loss,
            "done": done_loss,
        }

    def encode_tokens(self, observation: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(observation).tokens

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode_tokens(tokens)

    def initial_context(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encode_tokens(observation)

    def _trim_context(self, context: torch.Tensor) -> torch.Tensor:
        required = 1 + self.tokens_per_frame
        block = self.tokens_per_frame + 1
        while context.shape[1] + required > self.max_tokens:
            if context.shape[1] <= block:
                raise ValueError("Configured token context is too short for one generated frame")
            context = context[:, block:]
        return context

    def generate_next(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        temperature: float = 0.0,
    ) -> TokenRolloutStep:
        context = self._trim_context(context)
        context = torch.cat((context, action[:, None] + self.vocab_size), dim=1)
        action_hidden = self._hidden(context)[:, -1]
        predicted_reward = self.reward_head(action_hidden).squeeze(-1)
        predicted_done = self.done_head(action_hidden).squeeze(-1)
        generated: list[torch.Tensor] = []
        for _ in range(self.tokens_per_frame):
            logits = self.token_head(self._hidden(context)[:, -1])
            if temperature > 0:
                probabilities = F.softmax(logits / temperature, dim=-1)
                token = torch.multinomial(probabilities, 1).squeeze(-1)
            else:
                token = logits.argmax(dim=-1)
            generated.append(token)
            context = torch.cat((context, token[:, None]), dim=1)
        tokens = torch.stack(generated, dim=1)
        return TokenRolloutStep(
            context,
            self.decode_tokens(tokens),
            predicted_reward,
            predicted_done,
            tokens,
        )

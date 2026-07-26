from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn

from pvzwm.portable import PortableActionCodec, PortableActionKind


@dataclass(frozen=True, slots=True)
class EvolutionEpisode:
    won: bool
    episode_return: float
    length: int
    truncated: bool
    descriptor: np.ndarray
    zero_sun_plant_fraction: float = 0.0


@dataclass(frozen=True, slots=True)
class CandidateResult:
    quality: float
    win_rate: float
    mean_return: float
    truncation_rate: float
    descriptor: np.ndarray
    zero_sun_plant_fraction: float


def behavior_descriptor(actions: Sequence[int], max_steps: int) -> np.ndarray:
    """Describe a portable policy without using pixels or privileged game state."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    codec = PortableActionCodec()
    counts = np.zeros(5, dtype=np.float64)
    plant_slots: set[int] = set()
    plant_lanes: set[int] = set()
    plant_columns: list[int] = []
    for action_id in actions:
        action = codec.decode(int(action_id))
        counts[int(action.kind)] += 1.0
        if action.kind == PortableActionKind.PLANT:
            plant_slots.add(action.slot)
            plant_lanes.add(action.row)
            plant_columns.append(action.col)
    denominator = max(1, len(actions))
    mean_column = (
        float(np.mean(plant_columns)) / max(1, codec.cols - 1)
        if plant_columns
        else 0.0
    )
    return np.asarray(
        [
            counts[int(PortableActionKind.WAIT)] / denominator,
            counts[int(PortableActionKind.PLANT)] / denominator,
            counts[int(PortableActionKind.COLLECT)] / denominator,
            counts[int(PortableActionKind.SHOVEL)] / denominator,
            len(plant_slots) / codec.seed_slots,
            len(plant_lanes) / codec.rows,
            mean_column,
            min(1.0, len(actions) / max_steps),
        ],
        dtype=np.float64,
    )


def zero_sun_plant_fraction(
    actions: Sequence[int], sun_before_actions: Sequence[float]
) -> float:
    """Diagnostic only: this value must not contribute to evolutionary fitness."""

    if len(actions) != len(sun_before_actions):
        raise ValueError("sun_before_actions must align with actions")
    codec = PortableActionCodec()
    plants = 0
    zero_sun_plants = 0
    for action_id, sun in zip(actions, sun_before_actions, strict=True):
        if codec.decode(int(action_id)).kind == PortableActionKind.PLANT:
            plants += 1
            zero_sun_plants += int(sun <= 0.0)
    return zero_sun_plants / max(1, plants)


def aggregate_candidate(
    episodes: Sequence[EvolutionEpisode],
) -> CandidateResult:
    """Aggregate diagnostics while optimizing only mean environment return."""

    if not episodes:
        raise ValueError("A candidate needs at least one episode")
    win_rate = float(np.mean([episode.won for episode in episodes]))
    mean_return = float(np.mean([episode.episode_return for episode in episodes]))
    truncation_rate = float(np.mean([episode.truncated for episode in episodes]))
    descriptor = np.mean(
        np.stack([episode.descriptor for episode in episodes]), axis=0
    )
    zero_sun_fraction = float(
        np.mean([episode.zero_sun_plant_fraction for episode in episodes])
    )
    return CandidateResult(
        quality=mean_return,
        win_rate=win_rate,
        mean_return=mean_return,
        truncation_rate=truncation_rate,
        descriptor=descriptor,
        zero_sun_plant_fraction=zero_sun_fraction,
    )


def centered_ranks(values: Sequence[float]) -> np.ndarray:
    """Return tie-aware ranks in [-0.5, 0.5]."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a nonempty vector")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    if array.size == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks / (array.size - 1) - 0.5


def descriptor_cell(descriptor: np.ndarray, bins: int) -> tuple[int, ...]:
    """Quantize a normalized behavior descriptor for MAP-Elites competition."""

    values = np.asarray(descriptor, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("descriptor must be a nonempty vector")
    if bins < 2:
        raise ValueError("bins must be at least two")
    if not np.all(np.isfinite(values)):
        raise ValueError("descriptor values must be finite")
    clipped = np.clip(values, 0.0, 1.0)
    return tuple(np.minimum((clipped * bins).astype(np.int64), bins - 1).tolist())


def mutate_ordered_choice(
    choice: Sequence[int],
    available: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Mutate a unique ordered choice by replacement or slot permutation."""

    current = tuple(int(item) for item in choice)
    options = tuple(dict.fromkeys(int(item) for item in available))
    if not current or len(set(current)) != len(current):
        raise ValueError("choice must be nonempty and unique")
    if len(options) < len(current) or not set(current).issubset(options):
        raise ValueError("available choices must contain the current choice")
    replacements = tuple(item for item in options if item not in current)
    mutated = list(current)
    permute = len(current) > 1 and (not replacements or bool(rng.integers(0, 2)))
    if permute:
        left, right = rng.choice(len(current), size=2, replace=False).tolist()
        mutated[left], mutated[right] = mutated[right], mutated[left]
    else:
        index = int(rng.integers(0, len(current)))
        mutated[index] = int(rng.choice(replacements))
    return tuple(mutated)


def policy_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Evolve the actor and shared representation, leaving the critic untouched."""

    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("value.")
    )


def parameter_vector(model: nn.Module, names: Sequence[str]) -> torch.Tensor:
    parameters = dict(model.named_parameters())
    if not names:
        raise ValueError("At least one parameter must be selected")
    return nn.utils.parameters_to_vector([parameters[name] for name in names]).detach().clone()


def set_parameter_vector(
    model: nn.Module, names: Sequence[str], vector: torch.Tensor
) -> None:
    parameters = dict(model.named_parameters())
    selected = [parameters[name] for name in names]
    expected = sum(parameter.numel() for parameter in selected)
    if vector.numel() != expected:
        raise ValueError(f"Parameter vector has {vector.numel()} values; expected {expected}")
    with torch.no_grad():
        nn.utils.vector_to_parameters(vector, selected)


def seeded_noise(
    size: int,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if size < 1:
        raise ValueError("Noise size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(size, generator=generator, dtype=dtype).to(device)


def antithetic_gradient(
    noises: Sequence[torch.Tensor],
    utilities: Sequence[float],
    sigma: float,
) -> torch.Tensor:
    """Estimate an ascent direction; utilities are ordered [+eps..., -eps...]."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    pair_count = len(noises)
    if pair_count < 1 or len(utilities) != 2 * pair_count:
        raise ValueError("Antithetic utilities must contain one positive and negative per noise")
    utility = np.asarray(utilities, dtype=np.float64)
    gradient = torch.zeros_like(noises[0])
    for index, noise in enumerate(noises):
        if noise.shape != gradient.shape:
            raise ValueError("All noise vectors must have equal shape")
        coefficient = float(utility[index] - utility[pair_count + index])
        gradient.add_(noise, alpha=coefficient)
    return gradient / (pair_count * sigma)


@dataclass(slots=True)
class AdamAscent:
    size: int
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    step_number: int = field(init=False, default=0)
    first_moment: torch.Tensor | None = field(init=False, default=None)
    second_moment: torch.Tensor | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("Adam state size must be positive")
        self.step_number = 0
        self.first_moment = None
        self.second_moment = None

    def step(
        self, parameters: torch.Tensor, gradient: torch.Tensor, learning_rate: float
    ) -> torch.Tensor:
        if learning_rate <= 0.0 or parameters.shape != gradient.shape:
            raise ValueError("Invalid Adam ascent update")
        if parameters.numel() != self.size:
            raise ValueError("Parameter vector does not match Adam state size")
        if self.first_moment is None:
            self.first_moment = torch.zeros_like(parameters)
            self.second_moment = torch.zeros_like(parameters)
        assert self.second_moment is not None
        self.step_number += 1
        self.first_moment.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
        self.second_moment.mul_(self.beta2).addcmul_(
            gradient, gradient, value=1.0 - self.beta2
        )
        first = self.first_moment / (1.0 - self.beta1**self.step_number)
        second = self.second_moment / (1.0 - self.beta2**self.step_number)
        return parameters + learning_rate * first / (second.sqrt() + self.epsilon)

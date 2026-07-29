import numpy as np
import pytest
import torch

from pvzwm.agents.evolution import (
    AdamAscent,
    EvolutionEpisode,
    aggregate_candidate,
    antithetic_gradient,
    behavior_descriptor,
    centered_ranks,
    descriptor_cell,
    mutate_ordered_choice,
    parameter_vector,
    policy_parameter_names,
    set_parameter_vector,
    zero_sun_plant_fraction,
)
from pvzwm.agents.ppo import ActorCritic
from pvzwm.portable import PortableActionCodec


def test_behavior_descriptor_tracks_portable_strategy() -> None:
    codec = PortableActionCodec()
    actions = [0, codec.plant(2, 1, 3), codec.plant(4, 5, 7), codec.collect_action]
    descriptor = behavior_descriptor(actions, max_steps=8)

    assert descriptor.tolist() == pytest.approx(
        [0.25, 0.5, 0.25, 0.0, 0.2, 2 / 6, 5 / 8, 0.5]
    )
    assert zero_sun_plant_fraction(actions, [50.0, 25.0, 0.0, 0.0]) == 0.5


def test_quality_is_exactly_mean_environment_return() -> None:
    descriptor = np.zeros(8)
    result = aggregate_candidate(
        [
            EvolutionEpisode(True, -500.0, 10, False, descriptor),
            EvolutionEpisode(False, 100.0, 10, True, descriptor),
        ]
    )
    assert result.quality == pytest.approx(-200.0)
    assert result.win_rate == pytest.approx(0.5)
    assert result.truncation_rate == pytest.approx(0.5)


def test_centered_ranks_average_ties() -> None:
    ranks = centered_ranks([1.0, 1.0, 3.0])
    assert ranks.tolist() == pytest.approx([-0.25, -0.25, 0.5])


def test_descriptor_cells_and_ordered_choice_mutations_are_bounded() -> None:
    assert descriptor_cell(np.asarray([0.0, 0.24, 1.0]), bins=4) == (0, 0, 3)
    rng = np.random.default_rng(7)
    original = (0, 1, 2)
    for _ in range(20):
        mutated = mutate_ordered_choice(original, range(6), rng)
        assert len(mutated) == len(original)
        assert len(set(mutated)) == len(mutated)
        assert set(mutated).issubset(set(range(6)))
        assert mutated != original


def test_antithetic_gradient_and_adam_move_toward_better_perturbation() -> None:
    noises = [torch.tensor([1.0, -2.0])]
    gradient = antithetic_gradient(noises, [0.5, -0.5], sigma=0.25)
    assert gradient.tolist() == pytest.approx([4.0, -8.0])
    parameters = torch.zeros(2)
    updated = AdamAscent(2).step(parameters, gradient, learning_rate=0.1)
    assert updated[0] > 0.0
    assert updated[1] < 0.0


def test_policy_vector_excludes_critic_and_round_trips() -> None:
    model = ActorCritic(4, 3, hidden_dim=8)
    names = policy_parameter_names(model)
    assert names
    assert not any(name.startswith("value.") for name in names)
    original_value = model.value.weight.detach().clone()
    vector = parameter_vector(model, names)
    set_parameter_vector(model, names, vector + 0.1)
    assert torch.equal(model.value.weight, original_value)
    assert torch.allclose(parameter_vector(model, names), vector + 0.1)

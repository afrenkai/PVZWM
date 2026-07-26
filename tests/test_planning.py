import numpy as np
import torch

from pvzwm.agents.planning import CategoricalCEMPlanner
from pvzwm.domain import ActionCodec, GameConfig
from pvzwm.env import PVZEnv
from pvzwm.models.dynamics import DeterministicDynamics


def test_planner_reconstructs_legality_and_respects_support() -> None:
    config = GameConfig()
    env = PVZEnv(config, seed=0)
    observation, _ = env.reset(seed=0)
    model = DeterministicDynamics(
        observation.vector(config).size, ActionCodec(config).size, hidden_dim=16
    )
    planner = CategoricalCEMPlanner(
        model,
        ActionCodec(config).size,
        horizon=2,
        samples=8,
        elites=2,
        iterations=1,
        config=config,
        device=torch.device("cpu"),
    )
    state = torch.from_numpy(observation.vector(config)).unsqueeze(0)
    predicted = planner._predicted_legal_mask(state).squeeze(0).numpy()
    np.testing.assert_array_equal(predicted, env.legal_action_mask())

    wait_only = np.zeros(ActionCodec(config).size, dtype=np.bool_)
    wait_only[0] = True
    result = planner.plan(
        observation.vector(config), env.legal_action_mask(), wait_only
    )
    assert result.action == 0
    assert result.sequence == (0, 0)


def test_planner_accepts_non_clean_room_action_space_with_learned_legality() -> None:
    model = DeterministicDynamics(
        6, 5, hidden_dim=16, predict_legality=True
    )
    planner = CategoricalCEMPlanner(
        model,
        5,
        horizon=2,
        samples=8,
        elites=2,
        iterations=1,
        device=torch.device("cpu"),
    )
    state = np.zeros(6, dtype=np.float32)
    mask = planner._predicted_legal_mask(torch.from_numpy(state).unsqueeze(0))
    assert mask.shape == (1, 5)
    assert bool(mask[0, 0])
    result = planner.plan(state, np.asarray([True, False, False, False, False]))
    assert result.action == 0


def test_cem_elite_histogram_can_concentrate_in_large_action_space() -> None:
    torch.manual_seed(7)
    model = DeterministicDynamics(
        4, 64, hidden_dim=8, predict_legality=True
    )
    planner = CategoricalCEMPlanner(
        model,
        64,
        horizon=3,
        samples=512,
        elites=16,
        iterations=4,
        elite_prior_strength=1.0,
        device=torch.device("cpu"),
    )
    weights = torch.tensor([4.0, 2.0, 1.0])
    planner._score = lambda _state, sequences, _support: (  # type: ignore[method-assign]
        sequences.eq(63).to(torch.float32).mul(weights).sum(dim=1)
    )
    result = planner.plan(
        np.zeros(4, dtype=np.float32),
        np.ones(64, dtype=np.bool_),
    )
    assert result.sequence == (63, 63, 63)

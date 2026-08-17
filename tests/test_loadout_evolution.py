import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from pvzwm.agents.dqn import QNetwork
from pvzwm.agents.ppo import build_actor_critic

_SPEC = importlib.util.spec_from_file_location(
    "evolve_portable_loadout", Path("scripts/evolve_portable_loadout.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_load_initial_loadouts_preserves_order_and_removes_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "archive.json"
    path.write_text(
        json.dumps(
            {
                "archive": [
                    {"loadout": [16, 1, 0, 7, 5, 3]},
                    {"loadout": [6, 3, 0, 7, 5, 1]},
                    {"loadout": [16, 1, 0, 7, 5, 3]},
                ]
            }
        )
    )
    assert _MODULE.load_initial_loadouts(path) == (
        (16, 1, 0, 7, 5, 3),
        (6, 3, 0, 7, 5, 1),
    )


def test_load_initial_loadouts_rejects_malformed_archive(tmp_path: Path) -> None:
    path = tmp_path / "archive.json"
    path.write_text('{"archive": [{"not_loadout": []}]}')
    with pytest.raises(ValueError, match="invalid entry"):
        _MODULE.load_initial_loadouts(path)


@pytest.mark.parametrize("agent", ["dqn", "dyna-dqn"])
def test_build_controller_supports_value_based_agents(agent: str) -> None:
    source = QNetwork(8, 5, 16)
    model, restored_agent = _MODULE.build_controller(
        {
            "agent": agent,
            "state_dim": 8,
            "action_dim": 5,
            "hidden_dim": 16,
            "state_dict": source.state_dict(),
        },
        torch.device("cpu"),
    )
    scores = _MODULE.controller_scores(model, restored_agent, torch.zeros(8))
    assert restored_agent == agent
    assert scores.shape == (5,)


def test_build_controller_preserves_ppo_support() -> None:
    source = build_actor_critic(8, 5, 16, "flat")
    model, agent = _MODULE.build_controller(
        {
            "agent": "ppo",
            "state_dim": 8,
            "action_dim": 5,
            "hidden_dim": 16,
            "policy_architecture": "flat",
            "state_dict": source.state_dict(),
        },
        torch.device("cpu"),
    )
    assert _MODULE.controller_scores(model, agent, torch.zeros(8)).shape == (5,)


def test_build_controller_rejects_incompatible_agent() -> None:
    with pytest.raises(ValueError, match="PPO, DQN, or Dyna-DQN"):
        _MODULE.build_controller({"agent": "bc"}, torch.device("cpu"))


def test_select_action_respects_legality_for_argmax() -> None:
    action = _MODULE.select_action(
        torch.tensor([10.0, 2.0, 1.0]),
        torch.tensor([False, True, True]),
        stochastic=False,
        epsilon=0.0,
    )
    assert action == 1


def test_select_action_epsilon_one_samples_only_legal_actions() -> None:
    torch.manual_seed(17)
    actions = {
        _MODULE.select_action(
            torch.tensor([10.0, 2.0, 1.0]),
            torch.tensor([False, True, True]),
            stochastic=False,
            epsilon=1.0,
        )
        for _ in range(32)
    }
    assert actions == {1, 2}

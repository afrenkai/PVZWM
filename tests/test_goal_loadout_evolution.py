import importlib.util
from pathlib import Path
import sys

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "evolve_goal_iql_loadout", Path("scripts/evolve_goal_iql_loadout.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_summary_to_candidate_result_uses_environment_return_only() -> None:
    result = _MODULE.summary_to_candidate_result(
        {
            "episode_records": [
                {
                    "won": False,
                    "return": -9.5,
                    "steps": 12,
                    "terminated": True,
                    "truncated": False,
                    "behavior_descriptor": [0.5, 0.25, 0.25],
                    "zero_sun_plant_fraction_diagnostic": 0.4,
                },
                {
                    "won": True,
                    "return": 100.0,
                    "steps": 20,
                    "terminated": True,
                    "truncated": False,
                    "behavior_descriptor": [0.25, 0.5, 0.25],
                    "zero_sun_plant_fraction_diagnostic": 0.2,
                },
            ]
        }
    )
    assert result.quality == result.mean_return == 45.25
    assert result.win_rate == 0.5
    np.testing.assert_allclose(result.descriptor, [0.375, 0.375, 0.25])
    assert np.isclose(result.zero_sun_plant_fraction, 0.3)


def test_candidate_rank_prioritizes_terminal_wins_then_return() -> None:
    losing = _MODULE.CandidateResult(
        quality=10.0,
        win_rate=0.0,
        mean_return=10.0,
        truncation_rate=0.0,
        descriptor=np.zeros(2),
        zero_sun_plant_fraction=0.0,
    )
    winning = _MODULE.CandidateResult(
        quality=-10.0,
        win_rate=0.5,
        mean_return=-10.0,
        truncation_rate=0.0,
        descriptor=np.zeros(2),
        zero_sun_plant_fraction=0.0,
    )
    assert _MODULE.candidate_rank(winning) > _MODULE.candidate_rank(losing)


def test_evaluation_command_preserves_ordered_loadout_and_uses_uv() -> None:
    spec = _MODULE.GoalLoadoutEvalSpec(
        uv=Path("uv"),
        evaluator=Path("scripts/eval_goal_iql_frontier.py"),
        database=Path("replay.duckdb"),
        checkpoint=Path("worker.pt"),
        latent_goal_index=Path("goals.pt"),
        jepa_checkpoint=Path("jepa.pt"),
        calibration_reports=(Path("verified.json"), Path("clean.json")),
        binary=Path("pvzwm-bridge"),
        resources=Path("resources"),
        level=21,
        episode_steps=3600,
        max_options=32,
        max_option_steps=136,
        device="cuda",
        stochastic=False,
    )
    command = _MODULE.evaluation_command(
        spec, (16, 3, 0, 7, 5, 1), 123, 2, Path("result.json")
    )
    assert command[:2] == ["uv", "run"]
    assert command[command.index("--seed-types") + 1] == "16,3,0,7,5,1"
    assert command.count("--calibration-report") == 2
    assert command[command.index("--seed") + 1] == "123"

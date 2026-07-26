import numpy as np
import pytest

from pvzwm.agents.goal_selector import (
    FrontierGoal,
    fit_competence_threshold,
    select_frontier_goal,
)


def test_competence_threshold_is_fit_from_heldout_outcomes() -> None:
    threshold = fit_competence_threshold(
        [-12.0, -8.0, -3.0, -1.0], [False, False, True, True]
    )
    assert threshold == pytest.approx(-3.0)


def test_frontier_selector_prefers_depth_only_within_learned_competence() -> None:
    goals = [
        FrontierGoal(np.zeros(2), b"a", 10),
        FrontierGoal(np.ones(2), b"b", 40),
        FrontierGoal(np.full(2, 2.0), b"c", 80),
    ]
    selected = select_frontier_goal(
        goals,
        [-1.0, -2.0, -20.0],
        minimum_depth=0,
        competence_threshold=-5.0,
    )
    assert selected is not None
    assert selected[0].key == b"b"
    bounded = select_frontier_goal(
        goals,
        [-1.0, -2.0, -3.0],
        minimum_depth=0,
        maximum_depth=40,
        competence_threshold=-5.0,
    )
    assert bounded is not None
    assert bounded[0].key == b"b"
    assert select_frontier_goal(
        goals,
        [-1.0, -2.0, -20.0],
        minimum_depth=40,
        competence_threshold=-5.0,
    ) is None

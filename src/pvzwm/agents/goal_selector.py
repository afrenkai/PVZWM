from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class FrontierGoal:
    state: np.ndarray
    key: bytes
    depth: int


def fit_competence_threshold(
    predicted_values: Sequence[float], successes: Sequence[bool]
) -> float:
    """Fit a value cutoff by balanced accuracy on held-out engine outcomes."""

    values = np.asarray(predicted_values, dtype=np.float64)
    labels = np.asarray(successes, dtype=np.bool_)
    if values.ndim != 1 or labels.shape != values.shape or values.size == 0:
        raise ValueError("Competence calibration requires aligned nonempty vectors")
    if not np.all(np.isfinite(values)):
        raise ValueError("Competence values must be finite")
    thresholds = np.unique(values)
    best_threshold = float(thresholds[0])
    best_score = -np.inf
    best_coverage = -1
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((~labels).sum()))
    for threshold in thresholds:
        predicted = values >= threshold
        true_positive_rate = float((predicted & labels).sum()) / positives
        true_negative_rate = float((~predicted & ~labels).sum()) / negatives
        score = 0.5 * (true_positive_rate + true_negative_rate)
        coverage = int(predicted.sum())
        if score > best_score or (score == best_score and coverage > best_coverage):
            best_score = score
            best_coverage = coverage
            best_threshold = float(threshold)
    return best_threshold


def select_frontier_goal(
    candidates: Sequence[FrontierGoal],
    predicted_values: Sequence[float],
    *,
    minimum_depth: int,
    competence_threshold: float,
    blocked_keys: set[bytes] | None = None,
    maximum_depth: int | None = None,
) -> tuple[FrontierGoal, float] | None:
    """Choose the deepest predicted-reachable generic archive goal."""

    values = np.asarray(predicted_values, dtype=np.float64)
    if values.shape != (len(candidates),):
        raise ValueError("Every frontier candidate requires one predicted value")
    if not np.all(np.isfinite(values)):
        raise ValueError("Frontier values must be finite")
    blocked = set() if blocked_keys is None else blocked_keys
    eligible = [
        (candidate, float(value))
        for candidate, value in zip(candidates, values, strict=True)
        if candidate.depth > minimum_depth
        and (maximum_depth is None or candidate.depth <= maximum_depth)
        and candidate.key not in blocked
        and value >= competence_threshold
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item[0].depth, item[1]))

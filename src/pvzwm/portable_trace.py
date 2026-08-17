from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

_PROTOCOL_FIELDS = {"ok", "reward", "last_action", "ticks", "frame"}
_UNORDERED_LIST_FIELDS = {
    "available_seed_types",
    "legal_actions",
    "plants",
    "zombies",
    "projectiles",
    "coins",
    "grid_items",
    "mowers",
}


def _canonical_value(value: Any, *, field: str = "") -> Any:
    if isinstance(value, Mapping):
        return {
            key: _canonical_value(item, field=key)
            for key, item in sorted(value.items())
            if key not in _PROTOCOL_FIELDS
        }
    if isinstance(value, list):
        items = [_canonical_value(item) for item in value]
        if field in _UNORDERED_LIST_FIELDS:
            return sorted(
                items,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":"), allow_nan=False
                ),
            )
        return items
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Trace snapshots cannot contain non-finite floats")
        return 0.0 if value == 0.0 else value
    return value


def canonical_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove transport fields and deterministically order gameplay state."""

    canonical = _canonical_value(payload)
    if not isinstance(canonical, dict):
        raise TypeError("A portable snapshot must be a mapping")
    return canonical


def snapshot_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        canonical_snapshot(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def expand_tape_command(command: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Expand a compact command with ``repeat`` into single-tick requests."""

    request = dict(command)
    repeat = int(request.pop("repeat", 1))
    if repeat < 1:
        raise ValueError("Tape repeat must be positive")
    if request.get("cmd") not in {"reset", "step", "observe"}:
        raise ValueError(f"Unsupported tape command: {request.get('cmd')!r}")
    for _ in range(repeat):
        yield request.copy()


def first_snapshot_difference(
    expected: Any,
    actual: Any,
    *,
    float_tolerance: float = 0.0,
    path: str = "$",
) -> str | None:
    """Return a concise path to the first semantic difference."""

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            return (
                f"{path}: keys differ; missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(expected_keys):
            difference = first_snapshot_difference(
                expected[key],
                actual[key],
                float_tolerance=float_tolerance,
                path=f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if (
        isinstance(expected, Sequence)
        and not isinstance(expected, (str, bytes))
        and isinstance(actual, Sequence)
        and not isinstance(actual, (str, bytes))
    ):
        if len(expected) != len(actual):
            return f"{path}: lengths differ ({len(expected)} != {len(actual)})"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = first_snapshot_difference(
                expected_item,
                actual_item,
                float_tolerance=float_tolerance,
                path=f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if abs(float(expected) - float(actual)) <= float_tolerance:
            return None
    elif expected == actual:
        return None
    return f"{path}: {expected!r} != {actual!r}"

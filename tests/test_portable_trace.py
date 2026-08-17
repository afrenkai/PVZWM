from __future__ import annotations

from pvzwm.portable_trace import (
    canonical_snapshot,
    expand_tape_command,
    first_snapshot_difference,
    snapshot_digest,
)


def test_canonical_snapshot_ignores_transport_and_entity_order() -> None:
    left = {
        "ok": True,
        "reward": 3.0,
        "plants": [
            {"type": 1, "row": 2, "col": 1},
            {"type": 0, "row": 2, "col": 0},
        ],
        "legal_actions": [2, 0, 1],
        "sun": 50,
    }
    right = {
        "last_action": 9,
        "plants": list(reversed(left["plants"])),
        "legal_actions": [1, 2, 0],
        "sun": 50,
    }
    assert canonical_snapshot(left) == canonical_snapshot(right)
    assert snapshot_digest(left) == snapshot_digest(right)


def test_expand_tape_command_produces_independent_requests() -> None:
    expanded = list(expand_tape_command({"cmd": "step", "action": 0, "repeat": 3}))
    assert expanded == [{"cmd": "step", "action": 0}] * 3
    assert expanded[0] is not expanded[1]


def test_first_snapshot_difference_reports_exact_path_and_tolerance() -> None:
    expected = {"zombies": [{"x": 500.0, "health": 270}]}
    actual = {"zombies": [{"x": 500.00001, "health": 269}]}
    assert (
        first_snapshot_difference(expected, actual, float_tolerance=0.001)
        == "$.zombies[0].health: 270 != 269"
    )

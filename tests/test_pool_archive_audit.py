import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "audit_pool_archive", Path("scripts/audit_pool_archive.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_payload_helpers_preserve_stacked_plant_types() -> None:
    payload = {
        "plants": [
            {"type": 16, "row": 2, "col": 4},
            {"type": 0, "row": 2, "col": 4},
            {"type": 1, "row": 0, "col": 0},
        ],
        "seed_packets": [{"slot": 2, "type": 0}],
    }

    assert _MODULE.plants_at(payload, 2, 4) == (0, 16)
    assert _MODULE.packet_type(payload, 2) == 0
    assert _MODULE.packet_type(payload, 3) is None


def test_deepest_unique_cells_deduplicates_prefixes() -> None:
    payload = {
        "cells": [
            {"seed": 1, "actions": [1]},
            {"seed": 1, "actions": [1, 2, 3]},
            {"seed": 1, "actions": [1, 2, 3]},
            {"seed": 2, "actions": [4, 5]},
        ]
    }

    selected = _MODULE.deepest_unique_cells(payload, 2)

    assert [len(cell["actions"]) for cell in selected] == [3, 2]


def test_deepest_unique_cells_can_require_seed_slot() -> None:
    slot_two_action = 1 + 2 * 54 + 7
    payload = {
        "cells": [
            {"seed": 1, "actions": [1, 2, 3, 4]},
            {"seed": 2, "actions": [0, slot_two_action, 0]},
        ]
    }

    selected = _MODULE.deepest_unique_cells(
        payload, 2, require_plant_slot=2
    )

    assert [cell["seed"] for cell in selected] == [2]

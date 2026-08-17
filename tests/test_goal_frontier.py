import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_goal_iql_frontier", Path("scripts/eval_goal_iql_frontier.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_parse_seed_types_preserves_order() -> None:
    assert _MODULE.parse_seed_types("16,3,0,7,5,1") == (16, 3, 0, 7, 5, 1)


@pytest.mark.parametrize("value", ["1,1", "", "48,1"])
def test_parse_seed_types_rejects_invalid_decks(value: str) -> None:
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        _MODULE.parse_seed_types(value)


def test_behavior_descriptor_is_reported_from_executed_actions() -> None:
    source = Path("scripts/eval_goal_iql_frontier.py").read_text()
    assert '"behavior_descriptor": behavior_descriptor(' in source
    assert '"zero_sun_plant_fraction_diagnostic": zero_sun_plant_fraction(' in source

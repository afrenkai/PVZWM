import argparse
import importlib.util
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "evolve_portable_policy", Path("scripts/evolve_portable_policy.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_parse_seed_types_preserves_authentic_slot_order() -> None:
    assert _MODULE.parse_seed_types("16,1,0,7,5,3") == (16, 1, 0, 7, 5, 3)


@pytest.mark.parametrize("value", ["", "1,1", "48", "-1", "a,b"])
def test_parse_seed_types_rejects_invalid_decks(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _MODULE.parse_seed_types(value)

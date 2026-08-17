from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "create_goty_adventure_save",
    Path("scripts/create_goty_adventure_save.py"),
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2", 2), ("1-2", 2), ("5-10", 50)],
)
def test_parse_adventure_level(value: str, expected: int) -> None:
    assert _MODULE.parse_adventure_level(value) == expected


@pytest.mark.parametrize("value", ["1", "1-1", "0-2", "5-11", "2-3-4"])
def test_parse_adventure_level_rejects_invalid_targets(value: str) -> None:
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        _MODULE.parse_adventure_level(value)

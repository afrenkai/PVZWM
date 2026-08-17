from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from pvzwm.agents.go_explore import ArchiveCell

_SPEC = importlib.util.spec_from_file_location(
    "verify_portable_archive", Path("scripts/verify_portable_archive.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_unique_cells_filter_depth_and_duplicate_prefixes() -> None:
    cells = [
        ArchiveCell(b"a", 1, (1, 2), 0.0),
        ArchiveCell(b"b", 1, (1, 2), 1.0),
        ArchiveCell(b"c", 2, (1, 2, 3), 0.0),
        ArchiveCell(b"d", 3, (1, 2, 3, 4), 0.0),
    ]
    selected = _MODULE.unique_cells_in_depth_range(cells, 2, 3)
    assert [(cell.seed, cell.actions) for cell in selected] == [
        (2, (1, 2, 3)),
        (1, (1, 2)),
    ]

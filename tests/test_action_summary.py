import importlib.util
from pathlib import Path

from pvzwm.portable import PortableActionCodec


_SPEC = importlib.util.spec_from_file_location(
    "summarize_portable_actions", Path("scripts/summarize_portable_actions.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_plant_action_distribution_reports_every_slot_and_row() -> None:
    codec = PortableActionCodec()
    result = _MODULE.plant_action_distribution(
        [codec.plant(0, 2, 4), codec.plant(0, 2, 5), codec.plant(3, 5, 8), 0],
        codec,
    )
    assert result["total"] == 3
    assert result["distinct_cells"] == 3
    assert len(result["by_slot"]) == codec.seed_slots
    assert len(result["by_row"]) == codec.rows
    assert result["by_slot"][0]["count"] == 2
    assert result["by_slot"][3]["count"] == 1
    assert result["by_row"][2]["count"] == 2
    assert result["by_row"][5]["count"] == 1

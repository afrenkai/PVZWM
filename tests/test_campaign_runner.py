import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_full_campaign", Path("scripts/run_full_campaign.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_authentic_progress_ignores_rehearsal_and_requires_order() -> None:
    frontier, completed, done = _MODULE.authentic_progress(
        11,
        13,
        [1, 12, 11, 11, 12, 13],
        [True, True, False, True, True, True],
    )
    assert (frontier, completed, done) == (13, 13, True)


def test_stage_id_covers_full_adventure() -> None:
    assert _MODULE.stage_id(1) == "1-1"
    assert _MODULE.stage_id(10) == "1-10"
    assert _MODULE.stage_id(11) == "2-1"
    assert _MODULE.stage_id(50) == "5-10"


def test_campaign_runner_preserves_checkpoint_policy_architecture() -> None:
    assert _MODULE.checkpoint_policy_architecture({}) == "flat"
    assert (
        _MODULE.checkpoint_policy_architecture({"policy_architecture": "spatial"})
        == "spatial"
    )
    assert (
        _MODULE.checkpoint_policy_architecture(
            {"policy_architecture": "spatial-conv"}
        )
        == "spatial-conv"
    )
    assert (
        _MODULE.checkpoint_policy_architecture(
            {"policy_architecture": "portable-autoregressive"}
        )
        == "portable-autoregressive"
    )
    assert (
        _MODULE.checkpoint_policy_architecture(
            {"policy_architecture": "portable-vdn-autoregressive"}
        )
        == "portable-vdn-autoregressive"
    )


def test_level_loadout_archive_resolves_authentic_stage_placeholders() -> None:
    assert _MODULE.level_loadout_archive(None, 21) is None
    assert _MODULE.level_loadout_archive("archives/level-{level}.json", 21) == Path(
        "archives/level-21.json"
    )
    assert _MODULE.level_loadout_archive("archives/{stage}.json", 21) == Path(
        "archives/3-1.json"
    )

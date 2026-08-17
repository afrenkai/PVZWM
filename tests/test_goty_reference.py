from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pvzwm.goty_reference import (
    build_acceptance_plan,
    build_evidence_record,
    build_reference_manifest,
    record_acceptance_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCRIPT = PROJECT_ROOT / "scripts" / "manifest_goty_reference.py"
RECORD_SCRIPT = PROJECT_ROOT / "scripts" / "record_goty_acceptance.py"


def _write_reference(root: Path, *, executable: bool = True) -> None:
    root.mkdir()
    (root / "main.pak").write_bytes(b"goty-assets")
    (root / "properties").mkdir()
    (root / "properties" / "partner.xml").write_text("<Properties/>")
    if executable:
        (root / "PlantsVsZombies.exe").write_bytes(b"goty-executable")


def test_reference_manifest_is_stable_and_privacy_safe(tmp_path: Path) -> None:
    root = tmp_path / "private-install"
    _write_reference(root)

    first = build_reference_manifest(
        root,
        version="1.2.0.1073",
        distribution="steam-windows",
    )
    second = build_reference_manifest(
        root,
        version="1.2.0.1073",
        distribution="steam-windows",
    )

    assert first["content_id"] == second["content_id"]
    assert first["ready_for_black_box_comparison"] is True
    assert first["baseline_kind"] == "installation"
    assert first["game_executables"] == ["PlantsVsZombies.exe"]
    assert first["main_paks"] == ["main.pak"]
    assert str(root) not in json.dumps(first)


def test_assets_only_manifest_is_not_a_black_box_reference(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    _write_reference(root, executable=False)

    manifest = build_reference_manifest(root, version="1.2.0.1073")

    assert manifest["baseline_kind"] == "assets-only"
    assert manifest["ready_for_black_box_comparison"] is False


def test_acceptance_plan_covers_flows_campaign_and_secondary_modes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "install"
    _write_reference(root)
    manifest = build_reference_manifest(root, version="1.2.0.1073")

    plan = build_acceptance_plan(manifest)
    scenario_ids = [scenario["id"] for scenario in plan["scenarios"]]

    assert plan["scenario_count"] == 134
    assert len(scenario_ids) == len(set(scenario_ids))
    assert "flow/zombatar" in scenario_ids
    assert "adventure/1-1" in scenario_ids
    assert "adventure/5-10" in scenario_ids
    assert "mode/01-survival-day" in scenario_ids
    assert "mode/70-i-zombie-endless" in scenario_ids

    updated = record_acceptance_result(
        plan,
        "adventure/1-1",
        "mismatch",
        reference_artifacts=["reference/1-1.png"],
        candidate_artifacts=["candidate/1-1.png"],
        notes="Different reward transition.",
    )
    scenario = next(
        item for item in updated["scenarios"] if item["id"] == "adventure/1-1"
    )
    assert scenario["result"] == "mismatch"
    assert scenario["notes"] == "Different reward transition."
    assert next(
        item for item in plan["scenarios"] if item["id"] == "adventure/1-1"
    )["result"] == "pending"


def test_reference_manifest_cli_writes_manifest_and_plan(tmp_path: Path) -> None:
    root = tmp_path / "install"
    _write_reference(root)
    manifest_path = tmp_path / "reference.json"
    acceptance_path = tmp_path / "acceptance.json"

    result = subprocess.run(
        [
            sys.executable,
            str(MANIFEST_SCRIPT),
            str(root),
            "--version",
            "1.2.0.1073",
            "--distribution",
            "steam-windows",
            "--output",
            str(manifest_path),
            "--acceptance-output",
            str(acceptance_path),
            "--require-executable",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"ready_for_black_box_comparison": true' in result.stdout
    assert json.loads(manifest_path.read_text())["file_count"] == 3
    assert json.loads(acceptance_path.read_text())["scenario_count"] == 134


def test_reference_manifest_rejects_symbolic_links(tmp_path: Path) -> None:
    root = tmp_path / "install"
    _write_reference(root)
    (root / "alias.pak").symlink_to(root / "main.pak")

    with pytest.raises(ValueError, match="symbolic link"):
        build_reference_manifest(root, version="1.2.0.1073")


def test_acceptance_cli_hashes_local_evidence(tmp_path: Path) -> None:
    root = tmp_path / "install"
    _write_reference(root)
    manifest = build_reference_manifest(root, version="1.2.0.1073")
    plan_path = tmp_path / "comparison" / "acceptance.json"
    plan_path.parent.mkdir()
    plan_path.write_text(json.dumps(build_acceptance_plan(manifest)))
    reference_frame = plan_path.parent / "reference-1-1.png"
    candidate_frame = plan_path.parent / "candidate-1-1.png"
    reference_frame.write_bytes(b"reference-frame")
    candidate_frame.write_bytes(b"candidate-frame")

    result = subprocess.run(
        [
            sys.executable,
            str(RECORD_SCRIPT),
            str(plan_path),
            "adventure/1-1",
            "--result",
            "match",
            "--reference-artifact",
            str(reference_frame),
            "--candidate-artifact",
            str(candidate_frame),
            "--notes",
            "Matched reward transition.",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"match": 1' in result.stdout
    plan = json.loads(plan_path.read_text())
    scenario = next(
        item for item in plan["scenarios"] if item["id"] == "adventure/1-1"
    )
    assert scenario["result"] == "match"
    assert scenario["reference_artifacts"][0]["path"] == "reference-1-1.png"
    assert len(scenario["reference_artifacts"][0]["sha256"]) == 64


def test_evidence_must_stay_with_local_acceptance_plan(tmp_path: Path) -> None:
    evidence_root = tmp_path / "comparison"
    evidence_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="beneath"):
        build_evidence_record(outside, evidence_root=evidence_root)

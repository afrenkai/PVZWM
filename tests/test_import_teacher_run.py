from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORT_SCRIPT = PROJECT_ROOT / "scripts" / "import_teacher_run.py"


def _legacy_run(root: Path) -> Path:
    run = root / "legacy"
    checkpoint = run / "checkpoints" / "after-1-1"
    checkpoint.mkdir(parents=True)
    (run / "teacher.dmo").write_bytes(b"demo")
    (checkpoint / "pvzwm-checkpoint.json").write_text(
        json.dumps({"completed_level": 1}),
        encoding="utf-8",
    )
    (checkpoint / "save.dat").write_bytes(b"save")
    (run / "checkpoints.jsonl").write_text("checkpoint\n", encoding="utf-8")
    return run


def test_legacy_run_import_is_non_destructive_and_resumable(tmp_path: Path) -> None:
    source = _legacy_run(tmp_path)
    campaign = tmp_path / "world1.campaign"
    source_before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    result = subprocess.run(
        [
            sys.executable,
            str(IMPORT_SCRIPT),
            str(source),
            "--save-slot",
            "world1",
            "--campaign",
            str(campaign),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert (campaign / "segments/segment-0001.imported/teacher.dmo").read_bytes() == b"demo"
    assert (campaign / "checkpoints/after-1-1/save.dat").read_bytes() == b"save"
    assert json.loads(
        (campaign / "pvzwm-teacher-campaign.json").read_text(encoding="utf-8")
    )["save_slot"] == "world1"
    assert {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    } == source_before


def test_legacy_run_import_never_overwrites_campaign(tmp_path: Path) -> None:
    source = _legacy_run(tmp_path)
    campaign = tmp_path / "world1.campaign"
    campaign.mkdir()
    marker = campaign / "keep"
    marker.write_text("untouched", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(IMPORT_SCRIPT),
            str(source),
            "--save-slot",
            "world1",
            "--campaign",
            str(campaign),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert marker.read_text(encoding="utf-8") == "untouched"

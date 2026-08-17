from __future__ import annotations

import json
import os
import stat
import struct
import subprocess
import sys
from pathlib import Path

from pvzwm.teacher_demo import DEMO_FILE_ID, DEMO_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify_teacher_replay.py"
VERIFY_CAMPAIGN_SCRIPT = (
    PROJECT_ROOT / "scripts" / "verify_teacher_campaign_replay.py"
)


def _write_demo(
    path: Path,
    update_count: int,
    markers: list[tuple[str, int]],
) -> None:
    marker_buffer = bytearray(struct.pack("<I", len(markers)))
    for name, update in markers:
        encoded_name = name.encode()
        marker_buffer.extend(struct.pack("<H", len(encoded_name)))
        marker_buffer.extend(encoded_name)
        marker_buffer.extend(struct.pack("<I", update))
    path.write_bytes(
        struct.pack(
            "<IIIQiH",
            DEMO_FILE_ID,
            DEMO_VERSION,
            1234,
            1_700_000_000,
            -18_000,
            0,
        )
        + struct.pack("<I", len(marker_buffer))
        + marker_buffer
        + struct.pack("<I", update_count)
        + b"\x01"
    )


def _write_fake_binary(path: Path) -> None:
    path.write_text(
        """#!/bin/sh
report=
play_target=0
blackout=0
hidden=0
save_directory=
for argument in "$@"; do
    case "$argument" in
        -demo-report=*) report=${argument#-demo-report=} ;;
        -play-to=*) play_target=${argument#-play-to=} ;;
        -savedir=*) save_directory=${argument#-savedir=} ;;
        -demo-blackout) blackout=1 ;;
        -demo-hidden) hidden=1 ;;
    esac
done
test "$play_target" = 4322
test "$blackout" = "${EXPECT_BLACKOUT:-0}"
test "$hidden" = "${EXPECT_HIDDEN:-0}"
if test "${EXPECT_INITIAL_SAVE:-0}" = 1; then
    test -f "$save_directory/start-state"
fi
printf '%s\\n' '{' \
    '  "schema_version": 1,' \
    '  "status": "target_reached",' \
    '  "actual_update": 4322,' \
    '  "expected_update": 5000,' \
    '  "verification_target": 4322' \
    '}' > "$report"
"""
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_native_teacher_replay_report_is_validated(tmp_path: Path) -> None:
    session = tmp_path / "run"
    session.mkdir()
    _write_demo(
        session / "teacher.dmo",
        5000,
        [("pvzwm/checkpoint/after-1-3", 4321)],
    )
    binary = tmp_path / "pvz-portable"
    _write_fake_binary(binary)
    resources = tmp_path / "resources"
    resources.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(session),
            "--binary",
            str(binary),
            "--resources",
            str(resources),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert '"valid": true' in result.stdout
    assert '"actual_update": 4322' in result.stdout
    assert '"blackout": false' in result.stdout
    assert '"hidden": false' in result.stdout
    assert '"verification_target": 4322' in result.stdout


def test_native_teacher_replay_blackout_and_hidden_are_opt_in(
    tmp_path: Path,
) -> None:
    session = tmp_path / "run"
    session.mkdir()
    _write_demo(
        session / "teacher.dmo",
        5000,
        [("pvzwm/checkpoint/after-1-3", 4321)],
    )
    binary = tmp_path / "pvz-portable"
    _write_fake_binary(binary)
    resources = tmp_path / "resources"
    resources.mkdir()
    environment = os.environ.copy()
    environment["EXPECT_BLACKOUT"] = "1"
    environment["EXPECT_HIDDEN"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(session),
            "--binary",
            str(binary),
            "--resources",
            str(resources),
            "--blackout",
            "--hidden",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert '"valid": true' in result.stdout
    assert '"blackout": true' in result.stdout
    assert '"hidden": true' in result.stdout


def test_every_teacher_campaign_segment_is_replayed(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    for number in (1, 2):
        segment = campaign / "segments" / f"segment-{number:04d}.test"
        segment.mkdir(parents=True)
        _write_demo(
            segment / "teacher.dmo",
            5000,
            [(f"pvzwm/checkpoint/after-1-{number}", 4321)],
        )
    binary = tmp_path / "pvz-portable"
    _write_fake_binary(binary)
    resources = tmp_path / "resources"
    resources.mkdir()
    environment = os.environ.copy()
    environment["EXPECT_BLACKOUT"] = "1"
    environment["EXPECT_HIDDEN"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_CAMPAIGN_SCRIPT),
            str(campaign),
            "--binary",
            str(binary),
            "--resources",
            str(resources),
            "--blackout",
            "--hidden",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert len(report["segments"]) == 2
    assert all(segment["valid"] for segment in report["segments"])


def test_teacher_replay_copies_and_preserves_segment_start_save(
    tmp_path: Path,
) -> None:
    session = tmp_path / "segment"
    session.mkdir()
    _write_demo(
        session / "teacher.dmo",
        5000,
        [("pvzwm/checkpoint/after-1-3", 4321)],
    )
    initial_save = session / "start-save"
    initial_save.mkdir()
    marker = initial_save / "start-state"
    marker.write_bytes(b"immutable")
    binary = tmp_path / "pvz-portable"
    _write_fake_binary(binary)
    resources = tmp_path / "resources"
    resources.mkdir()
    environment = os.environ.copy()
    environment["EXPECT_INITIAL_SAVE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(session),
            "--binary",
            str(binary),
            "--resources",
            str(resources),
            "--initial-save",
            str(initial_save),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["initial_save"] == str(initial_save)
    assert report["initial_save_sha256"]
    assert marker.read_bytes() == b"immutable"

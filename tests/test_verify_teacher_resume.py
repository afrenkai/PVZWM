from __future__ import annotations

import json
import os
import stat
import struct
import subprocess
import sys
from pathlib import Path

from pvzwm.teacher_demo import PROFILE_VERSION, USER_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify_teacher_resume.py"


def _write_checkpoint(checkpoint: Path) -> None:
    userdata = checkpoint / "userdata"
    userdata.mkdir(parents=True)
    name = b"Lol"
    (userdata / "users.dat").write_bytes(
        struct.pack("<IH", PROFILE_VERSION, 1)
        + struct.pack("<H", len(name))
        + name
        + struct.pack("<II", 1, 1)
    )
    (userdata / "user1.dat").write_bytes(
        struct.pack("<IIII", USER_VERSION, 4, 0, 0)
    )
    (checkpoint / "pvzwm-checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pvzwm-human-teacher-checkpoint",
                "completed_level": 3,
                "next_adventure_level": 4,
            }
        ),
        encoding="utf-8",
    )


def _write_fake_bridge(path: Path) -> None:
    path.write_text(
        """#!/bin/sh
save_directory=
for argument in "$@"; do
    case "$argument" in
        -savedir=*) save_directory=${argument#-savedir=} ;;
    esac
done
test -f "$save_directory/userdata/user1.dat" || exit 3
printf '%s\\n' 'PVZWM {"ok":true,"ready":true,"protocol_version":4,"renderer":"metal","action_count":650,"profile_count":1,"profile":{"name":"Lol","id":1,"adventure_level":4,"coins":0,"finished_adventure":0}}'
IFS= read -r reset_command
printf '%s\\n' 'PVZWM {"ok":true,"area":1,"stage":4,"legal_actions":[0]}'
IFS= read -r quit_command
printf '%s\\n' 'PVZWM {"ok":true,"bye":true}'
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_native_checkpoint_resume_report_is_validated(tmp_path: Path) -> None:
    session = tmp_path / "run"
    checkpoint = session / "checkpoints" / "after-1-3"
    _write_checkpoint(checkpoint)
    original_user = (checkpoint / "userdata" / "user1.dat").read_bytes()
    binary = tmp_path / "pvzwm-metal-bridge"
    _write_fake_bridge(binary)
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
            "--expected-checkpoints",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    report = json.loads(result.stdout)
    assert report["valid"] is True
    assert report["renderer"] == "metal"
    assert report["checkpoints"][0]["loaded_profile"]["adventure_level"] == 4
    assert report["checkpoints"][0]["resumed_stage"] == {"area": 1, "stage": 4}
    assert (checkpoint / "userdata" / "user1.dat").read_bytes() == original_user

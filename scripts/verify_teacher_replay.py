#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pvzwm.teacher_demo import read_demo_summary


def _directory_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a PVZWM human teacher demo through the native game."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument(
        "--initial-save",
        type=Path,
        help="immutable save snapshot present when this demo segment began",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--blackout",
        action="store_true",
        help="suppress intermediate Metal presentation while fast-forwarding",
    )
    parser.add_argument(
        "--hidden",
        action="store_true",
        help="keep the native replay window hidden",
    )
    args = parser.parse_args()

    session = args.session.resolve()
    binary = args.binary.resolve()
    resources = args.resources.resolve()
    initial_save = args.initial_save.resolve() if args.initial_save else None
    initial_save_digest = (
        _directory_digest(initial_save) if initial_save is not None else None
    )
    demo = read_demo_summary(session / "teacher.dmo")
    last_checkpoint_update = max(
        (marker.update for marker in demo.markers),
        default=demo.update_count,
    )
    verify_to_end = last_checkpoint_update >= demo.update_count
    verification_target = (
        demo.update_count if verify_to_end else last_checkpoint_update + 1
    )

    with tempfile.TemporaryDirectory(prefix="pvzwm-teacher-replay-") as temporary:
        replay_root = Path(temporary)
        save_dir = replay_root / "save"
        teacher_session_dir = replay_root / "teacher-session"
        report_path = replay_root / "demo-report.json"
        if initial_save is None:
            save_dir.mkdir()
        else:
            shutil.copytree(initial_save, save_dir)
        teacher_session_dir.mkdir()

        command = [
            str(binary),
            f"-resdir={resources}",
            f"-savedir={save_dir}",
            f"-teacher-session-dir={teacher_session_dir}",
            "-windowed",
            f"-play={demo.path}",
            (
                "-play-to-end"
                if verify_to_end
                else f"-play-to={verification_target}"
            ),
            f"-demo-report={report_path}",
        ]
        if args.blackout:
            command.append("-demo-blackout")
        if args.hidden:
            command.append("-demo-hidden")
        environment = os.environ.copy()
        environment.setdefault("SDL_AUDIODRIVER", "dummy")
        result = subprocess.run(
            command,
            check=False,
            env=environment,
            stdout=sys.stderr,
            timeout=args.timeout,
        )

        if not report_path.is_file():
            raise SystemExit(
                f"Native replay exited {result.returncode} without a demo report"
            )
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        valid = (
            result.returncode == 0
            and report.get("status")
            == ("complete" if verify_to_end else "target_reached")
            and report.get("actual_update") == verification_target
            and report.get("expected_update") == demo.update_count
            and report.get("verification_target")
            == (0 if verify_to_end else verification_target)
            and (
                initial_save is None
                or _directory_digest(initial_save) == initial_save_digest
            )
        )
        print(
            json.dumps(
                {
                    "valid": valid,
                    "session": str(session),
                    "binary": str(binary),
                    "renderer": "native",
                    "blackout": args.blackout,
                    "hidden": args.hidden,
                    "initial_save": str(initial_save) if initial_save else None,
                    "initial_save_sha256": initial_save_digest,
                    "last_checkpoint_update": last_checkpoint_update,
                    "verification_target": verification_target,
                    "demo": demo.as_dict(),
                    "replay": report,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if not valid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

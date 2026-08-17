#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

VERIFY_REPLAY = Path(__file__).with_name("verify_teacher_replay.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay every independently recorded segment of a teacher campaign."
    )
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--blackout", action="store_true")
    parser.add_argument("--hidden", action="store_true")
    args = parser.parse_args()

    campaign = args.campaign.resolve()
    demo_paths: list[Path] = []
    legacy_demo = campaign / "teacher.dmo"
    if legacy_demo.is_file():
        demo_paths.append(legacy_demo)
    demo_paths.extend(sorted((campaign / "segments").glob("*/teacher.dmo")))
    if not demo_paths:
        raise SystemExit(f"Teacher campaign has no demo segments: {campaign}")

    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for demo_path in demo_paths:
        command = [
            sys.executable,
            str(VERIFY_REPLAY),
            str(demo_path.parent),
            "--binary",
            str(args.binary),
            "--resources",
            str(args.resources),
            "--timeout",
            str(args.timeout),
        ]
        initial_save = demo_path.parent / "start-save"
        if initial_save.is_dir():
            command.extend(("--initial-save", str(initial_save)))
        if args.blackout:
            command.append("--blackout")
        if args.hidden:
            command.append("--hidden")
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        try:
            report: dict[str, Any] = json.loads(result.stdout)
        except json.JSONDecodeError:
            report = {
                "valid": False,
                "session": str(demo_path.parent),
                "stdout": result.stdout,
            }
        report["segment"] = str(demo_path.parent.relative_to(campaign))
        reports.append(report)
        if result.returncode != 0 or not report.get("valid"):
            errors.append(f"Replay failed: {demo_path}")

    print(
        json.dumps(
            {
                "valid": not errors,
                "campaign": str(campaign),
                "renderer": "native",
                "blackout": args.blackout,
                "hidden": args.hidden,
                "errors": errors,
                "segments": reports,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

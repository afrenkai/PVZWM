#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pvzwm.teacher_demo import (
    SPECIAL_ADVENTURE_LEVELS,
    read_checkpoint_profiles,
    read_demo_summary,
    validate_complete_adventure,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a PVZWM human teacher run."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--expected-checkpoints", type=int)
    parser.add_argument(
        "--require-complete-adventure",
        action="store_true",
        help="require the exact clean-profile 1-1 through 5-10 sequence",
    )
    args = parser.parse_args()

    session = args.session.resolve()
    demo_paths: list[Path] = []
    legacy_demo = session / "teacher.dmo"
    if legacy_demo.is_file():
        demo_paths.append(legacy_demo)
    demo_paths.extend(
        sorted((session / "segments").glob("*/teacher.dmo"))
    )
    if not demo_paths:
        raise SystemExit(f"Teacher run has no demo segments: {session}")
    demos = tuple(read_demo_summary(path) for path in demo_paths)
    checkpoint_manifests = sorted(
        (session / "checkpoints").glob("*/pvzwm-checkpoint.json")
    )
    checkpoints: list[dict[str, Any]] = []
    errors: list[str] = []
    marker_updates: dict[str, list[int]] = {}
    for demo in demos:
        for marker in demo.markers:
            marker_updates.setdefault(marker.name, []).append(marker.update)
    for marker, updates in marker_updates.items():
        if len(updates) > 1:
            errors.append(f"Duplicate checkpoint marker in demo segments: {marker}")
    for manifest in checkpoint_manifests:
        checkpoint: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
        checkpoint["path"] = str(manifest.parent)
        try:
            profiles = read_checkpoint_profiles(manifest.parent)
        except (OSError, ValueError) as error:
            profiles = ()
            errors.append(f"Invalid checkpoint save {manifest.parent.name}: {error}")
        checkpoint["save_profiles"] = [
            profile.as_dict() for profile in profiles
        ]
        completed_level = checkpoint.get("completed_level")
        if isinstance(completed_level, int):
            checkpoint["special_stage"] = SPECIAL_ADVENTURE_LEVELS.get(
                completed_level
            )

        marker = str(checkpoint.get("demo_marker", ""))
        if marker not in marker_updates:
            errors.append(f"Checkpoint marker is absent from demo: {marker}")
        elif checkpoint.get("demo_update") not in marker_updates[marker]:
            errors.append(f"Checkpoint update does not match demo marker: {marker}")
        next_level = checkpoint.get("next_adventure_level")
        if profiles and not any(
            profile.adventure_level == next_level for profile in profiles
        ):
            errors.append(
                f"Checkpoint {manifest.parent.name} advertises adventure level "
                f"{next_level}, but no saved profile contains that level"
            )
        checkpoints.append(checkpoint)
    checkpoints.sort(
        key=lambda checkpoint: (
            checkpoint.get("completed_level")
            if isinstance(checkpoint.get("completed_level"), int)
            else 10**9
        )
    )
    if not checkpoints:
        errors.append("Teacher run has no checkpoints")
    if args.expected_checkpoints is not None and (
        len(checkpoints) != args.expected_checkpoints
    ):
        errors.append(
            f"Expected {args.expected_checkpoints} checkpoints, found {len(checkpoints)}"
        )
    if args.require_complete_adventure:
        errors.extend(validate_complete_adventure(checkpoints, demos))

    print(
        json.dumps(
            {
                "session": str(session),
                "valid": not errors,
                "complete_adventure": args.require_complete_adventure,
                "errors": errors,
                "demo": demos[0].as_dict() if len(demos) == 1 else None,
                "demos": [demo.as_dict() for demo in demos],
                "checkpoints": checkpoints,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

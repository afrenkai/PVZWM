#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pvzwm.portable import PortableBridgeClient, PortableProcessConfig
from pvzwm.teacher_demo import read_checkpoint_profiles


def _directory_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_checkpoint(
    manifest_path: Path,
    *,
    binary: Path,
    resources: Path,
    timeout: float,
    temporary_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    checkpoint = manifest_path.parent.resolve()
    manifest: dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    profiles = read_checkpoint_profiles(checkpoint)
    expected_level = int(manifest["next_adventure_level"])
    source_digest = _directory_digest(checkpoint)
    save_directory = temporary_root / checkpoint.name
    shutil.copytree(checkpoint, save_directory)

    client = PortableBridgeClient(
        PortableProcessConfig(
            binary=binary,
            resources=resources,
            xvfb_run=None,
            frame_width=0,
            frame_height=0,
            startup_timeout_seconds=timeout,
            save_directory=save_directory,
        )
    )
    try:
        ready = client.ready
        resumed = client.request(
            {
                "cmd": "reset",
                "level": expected_level,
                "seed": 0,
                "frame_width": 0,
                "frame_height": 0,
            }
        )
    finally:
        client.close()

    errors: list[str] = []
    if ready.get("renderer") != "metal":
        errors.append(
            f"native resume verifier selected {ready.get('renderer')!r}, not Metal"
        )
    loaded = ready.get("profile")
    if not isinstance(loaded, dict):
        errors.append("native engine did not select a saved profile")
        loaded = {}

    matching_profiles = [
        profile
        for profile in profiles
        if profile.name == loaded.get("name")
        and profile.profile_id == loaded.get("id")
        and profile.adventure_level == loaded.get("adventure_level")
        and profile.finished_adventure == loaded.get("finished_adventure")
    ]
    if not matching_profiles:
        errors.append("native engine profile does not match checkpoint save bytes")
    if loaded.get("adventure_level") != expected_level:
        errors.append(
            "native engine loaded Adventure level "
            f"{loaded.get('adventure_level')}; expected {expected_level}"
        )
    completed_level = int(manifest["completed_level"])
    if ready.get("profile_count") != len(profiles):
        errors.append(
            f"native engine loaded {ready.get('profile_count')} profiles; "
            f"expected {len(profiles)}"
        )
    expected_area = (expected_level - 1) // 10 + 1
    expected_stage = (expected_level - 1) % 10 + 1
    if (resumed.get("area"), resumed.get("stage")) != (
        expected_area,
        expected_stage,
    ):
        errors.append(
            "native engine resumed stage "
            f"{resumed.get('area')}-{resumed.get('stage')}; "
            f"expected {expected_area}-{expected_stage}"
        )
    if _directory_digest(checkpoint) != source_digest:
        errors.append("source checkpoint changed during verification")

    return (
        {
            "checkpoint": checkpoint.name,
            "path": str(checkpoint),
            "manifest_level": expected_level,
            "completed_level": completed_level,
            "profile_count": ready.get("profile_count"),
            "loaded_profile": loaded,
            "resumed_stage": {
                "area": resumed.get("area"),
                "stage": resumed.get("stage"),
            },
            "source_sha256": source_digest,
            "valid": not errors,
        },
        errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reopen every teacher checkpoint through the native engine."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--expected-checkpoints", type=int)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    session = args.session.resolve()
    binary = args.binary.resolve()
    resources = args.resources.resolve()
    manifests = sorted((session / "checkpoints").glob("*/pvzwm-checkpoint.json"))
    errors: list[str] = []
    if not manifests:
        errors.append("Teacher run has no checkpoints")
    if (
        args.expected_checkpoints is not None
        and len(manifests) != args.expected_checkpoints
    ):
        errors.append(
            f"Expected {args.expected_checkpoints} checkpoints, found {len(manifests)}"
        )

    checkpoints: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pvzwm-teacher-resume-") as temporary:
        temporary_root = Path(temporary)
        for manifest_path in manifests:
            try:
                checkpoint, checkpoint_errors = _verify_checkpoint(
                    manifest_path,
                    binary=binary,
                    resources=resources,
                    timeout=args.timeout,
                    temporary_root=temporary_root,
                )
            except (OSError, RuntimeError, ValueError) as error:
                checkpoint = {
                    "checkpoint": manifest_path.parent.name,
                    "path": str(manifest_path.parent),
                    "valid": False,
                }
                checkpoint_errors = [str(error)]
            checkpoints.append(checkpoint)
            errors.extend(
                f"{manifest_path.parent.name}: {error}"
                for error in checkpoint_errors
            )

    print(
        json.dumps(
            {
                "valid": not errors,
                "session": str(session),
                "binary": str(binary),
                "renderer": "metal",
                "errors": errors,
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

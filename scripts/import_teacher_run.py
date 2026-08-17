#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from pathlib import Path

SAVE_SLOT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def _directory_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _default_teacher_root() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library/Application Support/PVZWM/teacher-runs"
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return data_home / "pvzwm/teacher-runs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import an immutable legacy teacher run into a resumable campaign."
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--save-slot", required=True)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()

    if SAVE_SLOT_PATTERN.fullmatch(args.save_slot) is None:
        raise SystemExit("Invalid save slot name")
    source = args.run.resolve()
    source_demo = source / "teacher.dmo"
    source_checkpoints = source / "checkpoints"
    if not source_demo.is_file() or not source_checkpoints.is_dir():
        raise SystemExit("Source must contain teacher.dmo and checkpoints/")
    if not any(source_checkpoints.glob("*/pvzwm-checkpoint.json")):
        raise SystemExit("Source teacher run has no checkpoint manifests")

    campaign = (
        args.campaign.resolve()
        if args.campaign is not None
        else (_default_teacher_root() / f"{args.save_slot}.campaign").resolve()
    )
    if campaign.exists():
        raise SystemExit(f"Teacher campaign already exists: {campaign}")
    campaign.parent.mkdir(parents=True, exist_ok=True)

    source_digest = _directory_digest(source)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{campaign.name}.", dir=campaign.parent)
    )
    try:
        segment = temporary / "segments" / "segment-0001.imported"
        segment.mkdir(parents=True)
        (segment / "start-save").mkdir()
        shutil.copy2(source_demo, segment / "teacher.dmo")
        shutil.copytree(source_checkpoints, temporary / "checkpoints")
        source_index = source / "checkpoints.jsonl"
        if source_index.is_file():
            shutil.copy2(source_index, temporary / "checkpoints.jsonl")
        (temporary / "pvzwm-teacher-campaign.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pvzwm-human-teacher-campaign",
                    "save_slot": args.save_slot,
                    "imported_run": str(source),
                    "imported_run_sha256": source_digest,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, campaign)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    if _directory_digest(source) != source_digest:
        raise SystemExit("Source teacher run changed during import")
    print(
        json.dumps(
            {
                "valid": True,
                "source": str(source),
                "source_sha256": source_digest,
                "campaign": str(campaign),
                "save_slot": args.save_slot,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

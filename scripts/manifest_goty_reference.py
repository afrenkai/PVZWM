#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pvzwm.goty_reference import (
    build_acceptance_plan,
    build_reference_manifest,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hash a legally owned Plants vs. Zombies GOTY installation and "
            "optionally initialize its black-box acceptance plan."
        )
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--distribution", default="unknown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acceptance-output", type=Path)
    parser.add_argument(
        "--require-executable",
        action="store_true",
        help="fail unless the root contains main.pak, the game executable, and a version",
    )
    args = parser.parse_args()

    ignored = tuple(
        path
        for path in (args.output, args.acceptance_output)
        if path is not None
    )
    manifest = build_reference_manifest(
        args.root,
        version=args.version,
        distribution=args.distribution,
        ignored_paths=ignored,
    )
    if args.require_executable and not manifest["ready_for_black_box_comparison"]:
        raise SystemExit(
            "Reference is incomplete: provide an installation containing main.pak "
            "and PlantsVsZombies.exe (or the legacy macOS app), plus --version."
        )

    if args.output is not None:
        write_json_atomic(args.output, manifest)
    if args.acceptance_output is not None:
        write_json_atomic(
            args.acceptance_output,
            build_acceptance_plan(manifest),
        )

    print(
        json.dumps(
            {
                "baseline_kind": manifest["baseline_kind"],
                "content_id": manifest["content_id"],
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "declared_version": manifest["declared_version"],
                "game_executables": manifest["game_executables"],
                "main_paks": manifest["main_paks"],
                "ready_for_black_box_comparison": manifest[
                    "ready_for_black_box_comparison"
                ],
                "manifest": str(args.output) if args.output is not None else None,
                "acceptance": (
                    str(args.acceptance_output)
                    if args.acceptance_output is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

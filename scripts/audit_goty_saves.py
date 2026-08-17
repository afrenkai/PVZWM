#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from pvzwm.goty_saves import (
    MIDLEVEL_SAVEABLE_MODES,
    SaveTarget,
    adventure_save_targets,
    secondary_save_targets,
    validate_v4_save,
)
from pvzwm.portable import PortableBridgeClient, PortableProcessConfig
from pvzwm.portable_trace import canonical_snapshot, first_snapshot_difference


def parse_integer_spec(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ".." in item:
            bounds = item.split("..")
            if len(bounds) != 2:
                raise argparse.ArgumentTypeError(f"invalid range: {item!r}")
            start, stop = (int(bound.strip()) for bound in bounds)
            if stop < start:
                raise argparse.ArgumentTypeError(
                    f"range must be increasing: {item!r}"
                )
            values.extend(range(start, stop + 1))
        else:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("integer specification contains duplicates")
    return tuple(values)


def parse_restart_targets(value: str) -> tuple[SaveTarget, ...]:
    targets: list[SaveTarget] = []
    for item in value.split(","):
        kind, separator, raw_value = item.strip().partition(":")
        if not separator:
            raise argparse.ArgumentTypeError(
                "restart targets use adventure:<level> or mode:<id>"
            )
        try:
            identifier = int(raw_value)
            if kind == "adventure":
                target = adventure_save_targets((identifier,))[0]
            elif kind == "mode":
                target = secondary_save_targets((identifier,))[0]
            else:
                raise ValueError(f"unknown target kind {kind!r}")
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        targets.append(target)
    if not targets:
        raise argparse.ArgumentTypeError("expected at least one restart target")
    return tuple(targets)


def make_client(
    binary: Path,
    resources: Path,
    save_directory: Path,
) -> PortableBridgeClient:
    return PortableBridgeClient(
        PortableProcessConfig(
            binary=binary,
            resources=resources,
            save_directory=save_directory,
            xvfb_run=None,
            frame_width=0,
            frame_height=0,
        )
    )


def require_identical(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    context: str,
) -> None:
    difference = first_snapshot_difference(
        canonical_snapshot(expected),
        canonical_snapshot(actual),
    )
    if difference is not None:
        raise RuntimeError(f"{context}: {difference}")


def request_pair(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient,
    command: dict[str, Any],
    *,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_payload = reference.request(command)
    candidate_payload = candidate.request(command)
    require_identical(reference_payload, candidate_payload, context=context)
    return reference_payload, candidate_payload


def profile_id(client: PortableBridgeClient) -> int:
    profile = client.ready.get("profile")
    if not isinstance(profile, dict) or not isinstance(profile.get("id"), int):
        raise TypeError(f"bridge has no persisted profile identity: {profile!r}")
    return int(profile["id"])


def audit_in_process(
    *,
    reference_binary: Path,
    candidate_binary: Path,
    resources: Path,
    targets: tuple[SaveTarget, ...],
    checkpoint: int,
    continuation: int,
    seed: int,
) -> int:
    comparisons = 0
    with (
        tempfile.TemporaryDirectory(prefix="pvzwm-save-reference-") as reference_root,
        tempfile.TemporaryDirectory(prefix="pvzwm-save-candidate-") as candidate_root,
    ):
        reference_directory = Path(reference_root)
        candidate_directory = Path(candidate_root)
        reference = make_client(reference_binary, resources, reference_directory)
        candidate = make_client(candidate_binary, resources, candidate_directory)
        try:
            candidate_profile_id = profile_id(candidate)
            for target in targets:
                request_pair(
                    reference,
                    candidate,
                    target.reset_command(seed=seed),
                    context=f"{target.label} reset",
                )
                before, _ = request_pair(
                    reference,
                    candidate,
                    {
                        "cmd": "step",
                        "action": 0,
                        "ticks": checkpoint,
                        "frame_width": 0,
                        "frame_height": 0,
                    },
                    context=f"{target.label} checkpoint",
                )
                if before.get("done", False):
                    raise RuntimeError(
                        f"{target.label} terminated before the save checkpoint"
                    )

                candidate.request({"cmd": "save"})
                save_path = target.save_path(
                    candidate_directory,
                    profile_id=candidate_profile_id,
                )
                save_errors = validate_v4_save(save_path)
                if save_errors:
                    raise RuntimeError(
                        f"{target.label} invalid save: {'; '.join(save_errors)}"
                    )

                loaded = candidate.request(
                    {
                        "cmd": "load",
                        "game_mode": target.game_mode,
                        "frame_width": 0,
                        "frame_height": 0,
                    }
                )
                require_identical(
                    before,
                    loaded,
                    context=f"{target.label} immediate reload",
                )
                comparisons += 1

                continued_reference, continued_candidate = request_pair(
                    reference,
                    candidate,
                    {
                        "cmd": "step",
                        "action": 0,
                        "ticks": continuation,
                        "frame_width": 0,
                        "frame_height": 0,
                    },
                    context=f"{target.label} post-reload continuation",
                )
                comparisons += 1
                print(
                    json.dumps(
                        {
                            "target": target.label,
                            "game_mode": target.game_mode,
                            "level": target.level,
                            "checkpoint": checkpoint,
                            "continuation": continuation,
                            "save_bytes": save_path.stat().st_size,
                            "reload_identical": True,
                            "continuation_identical": True,
                            "terminal_after_continuation": bool(
                                continued_reference.get("done", False)
                                or continued_candidate.get("done", False)
                            ),
                        },
                        allow_nan=False,
                    )
                )
        finally:
            candidate.close()
            reference.close()
    return comparisons


def audit_process_restart(
    *,
    reference_binary: Path,
    candidate_binary: Path,
    resources: Path,
    target: SaveTarget,
    checkpoint: int,
    continuation: int,
    seed: int,
) -> int:
    with (
        tempfile.TemporaryDirectory(
            prefix="pvzwm-restart-reference-"
        ) as reference_root,
        tempfile.TemporaryDirectory(
            prefix="pvzwm-restart-candidate-"
        ) as candidate_root,
    ):
        reference_directory = Path(reference_root)
        candidate_directory = Path(candidate_root)
        reference = make_client(reference_binary, resources, reference_directory)
        source = make_client(candidate_binary, resources, candidate_directory)
        resumed: PortableBridgeClient | None = None
        try:
            source_profile = source.ready.get("profile")
            if not isinstance(source_profile, dict):
                raise TypeError(
                    f"{target.label} source has no profile: {source_profile!r}"
                )
            expected_profile = dict(source_profile)
            if target.level is not None:
                expected_profile["adventure_level"] = target.level
            request_pair(
                reference,
                source,
                target.reset_command(seed=seed),
                context=f"{target.label} restart reset",
            )
            before, _ = request_pair(
                reference,
                source,
                {
                    "cmd": "step",
                    "action": 0,
                    "ticks": checkpoint,
                    "frame_width": 0,
                    "frame_height": 0,
                },
                context=f"{target.label} restart checkpoint",
            )
            if before.get("done", False):
                raise RuntimeError(
                    f"{target.label} terminated before the restart checkpoint"
                )
            source.request({"cmd": "save"})
            save_path = target.save_path(
                candidate_directory,
                profile_id=profile_id(source),
            )
            save_errors = validate_v4_save(save_path)
            if save_errors:
                raise RuntimeError(
                    f"{target.label} invalid restart save: {'; '.join(save_errors)}"
                )

            source.close()
            resumed = make_client(candidate_binary, resources, candidate_directory)
            if resumed.ready.get("profile") != expected_profile:
                raise RuntimeError(
                    f"{target.label} profile changed across restart: "
                    f"{expected_profile!r} != {resumed.ready.get('profile')!r}"
                )
            loaded = resumed.request(
                {
                    "cmd": "load",
                    "game_mode": target.game_mode,
                    "frame_width": 0,
                    "frame_height": 0,
                }
            )
            require_identical(
                before,
                loaded,
                context=f"{target.label} process restart reload",
            )
            request_pair(
                reference,
                resumed,
                {
                    "cmd": "step",
                    "action": 0,
                    "ticks": continuation,
                    "frame_width": 0,
                    "frame_height": 0,
                },
                context=f"{target.label} process restart continuation",
            )
            print(
                json.dumps(
                    {
                        "target": target.label,
                        "process_restart": True,
                        "profile_identical": True,
                        "reload_identical": True,
                        "continuation_identical": True,
                    }
                )
            )
        finally:
            if resumed is not None:
                resumed.close()
            source.close()
            reference.close()
    return 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Require exact native mid-level save/reload state and deterministic "
            "continuation against an uninterrupted bridge."
        )
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument(
        "--levels",
        type=parse_integer_spec,
        default=parse_integer_spec("1..50"),
    )
    parser.add_argument(
        "--modes",
        type=parse_integer_spec,
        default=MIDLEVEL_SAVEABLE_MODES,
    )
    parser.add_argument("--checkpoint", type=int, default=500)
    parser.add_argument("--continuation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--restart-targets",
        type=parse_restart_targets,
        default=parse_restart_targets("adventure:31,mode:35"),
    )
    args = parser.parse_args()
    if args.checkpoint < 1 or args.continuation < 1:
        raise ValueError("checkpoint and continuation must be positive")

    targets = (
        *adventure_save_targets(args.levels),
        *secondary_save_targets(args.modes),
    )
    comparisons = audit_in_process(
        reference_binary=args.binary,
        candidate_binary=args.candidate_binary,
        resources=args.resources,
        targets=targets,
        checkpoint=args.checkpoint,
        continuation=args.continuation,
        seed=args.seed,
    )
    for target in args.restart_targets:
        comparisons += audit_process_restart(
            reference_binary=args.binary,
            candidate_binary=args.candidate_binary,
            resources=args.resources,
            target=target,
            checkpoint=args.checkpoint,
            continuation=args.continuation,
            seed=args.seed,
        )

    print(
        json.dumps(
            {
                "targets": len(targets),
                "adventure_levels": len(args.levels),
                "secondary_modes": len(args.modes),
                "process_restarts": len(args.restart_targets),
                "comparisons": comparisons,
                "checkpoint": args.checkpoint,
                "continuation": args.continuation,
                "save_format_valid": True,
                "state_identical": True,
            }
        )
    )


if __name__ == "__main__":
    main()

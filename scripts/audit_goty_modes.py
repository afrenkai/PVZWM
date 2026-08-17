#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from pvzwm.goty_modes import GAME_MODE_BY_ID, validate_game_mode_snapshot
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


def _request(
    client: PortableBridgeClient,
    *,
    game_mode: int,
    checkpoint: int,
    previous_checkpoint: int,
    seed: int,
) -> dict[str, Any]:
    if checkpoint == 0:
        return client.request(
            {
                "cmd": "reset",
                "game_mode": game_mode,
                "seed": seed,
                "frame_width": 0,
                "frame_height": 0,
            }
        )
    return client.request(
        {
            "cmd": "step",
            "action": 0,
            "ticks": checkpoint - previous_checkpoint,
            "frame_width": 0,
            "frame_height": 0,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every native GOTY secondary mode and optionally require "
            "exact gameplay state from a second renderer."
        )
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument(
        "--modes",
        type=parse_integer_spec,
        default=parse_integer_spec("1..70"),
    )
    parser.add_argument(
        "--checkpoints",
        type=parse_integer_spec,
        default=parse_integer_spec("0,500,2000"),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if tuple(sorted(args.checkpoints)) != args.checkpoints:
        raise ValueError("checkpoints must be increasing")
    if args.checkpoints[0] != 0:
        raise ValueError("checkpoints must start at zero")
    unknown_modes = sorted(set(args.modes) - GAME_MODE_BY_ID.keys())
    if unknown_modes:
        raise ValueError(f"unsupported secondary game modes: {unknown_modes}")

    with (
        tempfile.TemporaryDirectory(prefix="pvzwm-modes-reference-") as reference_save,
        tempfile.TemporaryDirectory(prefix="pvzwm-modes-candidate-") as candidate_save,
    ):
        common = {
            "resources": args.resources,
            "xvfb_run": None,
            "frame_width": 0,
            "frame_height": 0,
        }
        reference = PortableBridgeClient(
            PortableProcessConfig(
                binary=args.binary,
                save_directory=Path(reference_save),
                **common,
            )
        )
        candidate = (
            PortableBridgeClient(
                PortableProcessConfig(
                    binary=args.candidate_binary,
                    save_directory=Path(candidate_save),
                    **common,
                )
            )
            if args.candidate_binary is not None
            else None
        )

        comparisons = 0
        try:
            for game_mode in args.modes:
                spec = GAME_MODE_BY_ID[game_mode]
                previous_checkpoint = 0
                for checkpoint in args.checkpoints:
                    reference_payload = _request(
                        reference,
                        game_mode=game_mode,
                        checkpoint=checkpoint,
                        previous_checkpoint=previous_checkpoint,
                        seed=args.seed,
                    )
                    errors = validate_game_mode_snapshot(
                        spec,
                        reference_payload,
                        checkpoint=checkpoint,
                    )
                    state_difference = None
                    if candidate is not None:
                        candidate_payload = _request(
                            candidate,
                            game_mode=game_mode,
                            checkpoint=checkpoint,
                            previous_checkpoint=previous_checkpoint,
                            seed=args.seed,
                        )
                        errors.extend(
                            f"candidate: {error}"
                            for error in validate_game_mode_snapshot(
                                spec,
                                candidate_payload,
                                checkpoint=checkpoint,
                            )
                        )
                        state_difference = first_snapshot_difference(
                            canonical_snapshot(reference_payload),
                            canonical_snapshot(candidate_payload),
                        )
                        if state_difference is not None:
                            errors.append(f"candidate state: {state_difference}")

                    record = {
                        "game_mode": game_mode,
                        "name": spec.name,
                        "family": spec.family,
                        "checkpoint": checkpoint,
                        "invariants_valid": not errors,
                        "state_identical": (
                            None if candidate is None else state_difference is None
                        ),
                        "errors": errors,
                    }
                    print(json.dumps(record, allow_nan=False))
                    if errors:
                        raise RuntimeError(
                            f"mode audit failed at {spec.name}, tick {checkpoint}: "
                            f"{'; '.join(errors)}"
                        )
                    comparisons += 1
                    previous_checkpoint = checkpoint
        finally:
            if candidate is not None:
                candidate.close()
            reference.close()

    print(
        json.dumps(
            {
                "modes": len(args.modes),
                "comparisons": comparisons,
                "checkpoints": list(args.checkpoints),
                "invariants_valid": True,
                "state_identical": True if args.candidate_binary else None,
            }
        )
    )


if __name__ == "__main__":
    main()

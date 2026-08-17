from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from pvzwm.campaign import campaign_stages
from pvzwm.goty_campaign import (
    adventure_level_number,
    validate_adventure_snapshot,
)
from pvzwm.portable import (
    PortableActionCodec,
    PortableBridgeClient,
    PortableProcessConfig,
)
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
    level: int,
    checkpoint: int,
    previous_checkpoint: int,
    seed: int,
) -> dict[str, Any]:
    if checkpoint == 0:
        return client.request(
            {
                "cmd": "reset",
                "level": level,
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


def _paired_request(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient | None,
    command: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    reference_payload = reference.request(command)
    if candidate is not None:
        candidate_payload = candidate.request(command)
        difference = first_snapshot_difference(
            canonical_snapshot(reference_payload),
            canonical_snapshot(candidate_payload),
        )
        if difference is not None:
            raise RuntimeError(f"{context}: candidate state diverged: {difference}")
    return reference_payload


def _reset_pair(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient | None,
    *,
    level: int,
    seed: int,
) -> dict[str, Any]:
    return _paired_request(
        reference,
        candidate,
        {
            "cmd": "reset",
            "level": level,
            "seed": seed,
            "frame_width": 0,
            "frame_height": 0,
        },
        context=f"level {level} reset",
    )


def _step_pair(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient | None,
    *,
    level: int,
    action: int,
    ticks: int,
) -> dict[str, Any]:
    return _paired_request(
        reference,
        candidate,
        {
            "cmd": "step",
            "action": action,
            "ticks": ticks,
            "frame_width": 0,
            "frame_height": 0,
        },
        context=f"level {level} action {action}",
    )


def exercise_vasebreaker(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient | None,
    *,
    seed: int,
) -> dict[str, Any]:
    level = 35
    payload = _reset_pair(reference, candidate, level=level, seed=seed)
    pots = [item for item in payload["grid_items"] if item["type"] == 7]
    if not pots:
        raise RuntimeError("4-5 Vasebreaker reset did not create a scary pot")
    pot = pots[0]
    action = PortableActionCodec().board_click(int(pot["row"]), int(pot["col"]))
    before_count = len(pots)
    for _ in range(20):
        payload = _step_pair(
            reference,
            candidate,
            level=level,
            action=action,
            ticks=1,
        )
        remaining = [
            item
            for item in payload["grid_items"]
            if item["type"] == 7
            and item["row"] == pot["row"]
            and item["col"] == pot["col"]
        ]
        if not remaining:
            break
    else:
        raise RuntimeError("4-5 board click did not break the selected vase")
    return {
        "level": "4-5",
        "mode": "vasebreaker",
        "action": action,
        "before_vases": before_count,
        "after_vases": sum(
            item["type"] == 7 for item in payload["grid_items"]
        ),
        "interaction_valid": True,
    }


def exercise_whack_a_zombie(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient | None,
    *,
    seed: int,
) -> dict[str, Any]:
    level = 15
    payload = _reset_pair(reference, candidate, level=level, seed=seed)
    codec = PortableActionCodec()
    clicks = range(codec.board_click_begin, codec.collect_action)
    damage_observed = False
    action_used = 0
    for _ in range(60):
        if payload.get("done", False):
            break
        if payload["zombies"]:
            for action in clicks:
                health_before = int(payload["zombie_health"])
                count_before = int(payload["zombies_count"])
                payload = _step_pair(
                    reference,
                    candidate,
                    level=level,
                    action=action,
                    ticks=1,
                )
                if (
                    int(payload["zombie_health"]) < health_before
                    or int(payload["zombies_count"]) < count_before
                ):
                    damage_observed = True
                    action_used = action
                    break
            if damage_observed:
                break
        payload = _step_pair(
            reference,
            candidate,
            level=level,
            action=0,
            ticks=25,
        )
    if not damage_observed:
        raise RuntimeError("2-5 board clicks never damaged a surfaced zombie")
    return {
        "level": "2-5",
        "mode": "whack-a-zombie",
        "action": action_used,
        "interaction_valid": True,
    }


def exercise_conveyor(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient | None,
    *,
    level: int,
    stage_id: str,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    payload = _reset_pair(reference, candidate, level=level, seed=seed)
    for _ in range(200):
        plant_actions = [
            action
            for action in payload["legal_actions"]
            if PortableActionCodec.plant_begin
            <= action
            < PortableActionCodec.shovel_begin
        ]
        if plant_actions:
            break
        if payload.get("done", False):
            raise RuntimeError(
                f"{stage_id} terminated before its conveyor supplied a seed"
            )
        payload = _step_pair(
            reference,
            candidate,
            level=level,
            action=0,
            ticks=10,
        )
    else:
        raise RuntimeError(f"{stage_id} conveyor supplied no plant action")

    action = plant_actions[0]
    packets_before = payload["seed_packets"]
    entities_before = (
        payload["plants_count"],
        len(payload["projectiles"]),
    )
    payload = _step_pair(
        reference,
        candidate,
        level=level,
        action=action,
        ticks=1,
    )
    entities_after = (
        payload["plants_count"],
        len(payload["projectiles"]),
    )
    if payload["seed_packets"] == packets_before and entities_after == entities_before:
        raise RuntimeError(f"{stage_id} legal conveyor placement had no visible effect")
    return {
        "level": stage_id,
        "mode": mode,
        "action": action,
        "interaction_valid": True,
    }


def exercise_special_interactions(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    results = [
        exercise_whack_a_zombie(reference, candidate, seed=seed),
        exercise_vasebreaker(reference, candidate, seed=seed),
    ]
    stages = campaign_stages()
    for stage in stages:
        if stage.mode not in {
            "wallnut-bowling",
            "little-trouble",
            "bungee-blitz",
            "conveyor",
            "boss",
        }:
            continue
        results.append(
            exercise_conveyor(
                reference,
                candidate,
                level=adventure_level_number(stage),
                stage_id=stage.stage_id,
                mode=stage.mode,
                seed=seed,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every GOTY Adventure stage and optionally require exact "
            "gameplay state from a second renderer/backend"
        )
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument(
        "--levels",
        type=parse_integer_spec,
        default=parse_integer_spec("1..50"),
    )
    parser.add_argument(
        "--checkpoints",
        type=parse_integer_spec,
        default=parse_integer_spec("0,500,2000"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--exercise-special-actions",
        action="store_true",
        help=(
            "Break a vase, hit a surfaced zombie, and place one seed from every "
            "Adventure conveyor"
        ),
    )
    args = parser.parse_args()

    if tuple(sorted(args.checkpoints)) != args.checkpoints:
        raise ValueError("checkpoints must be increasing")
    if args.checkpoints[0] != 0:
        raise ValueError("checkpoints must start at zero")
    if any(level < 1 or level > 50 for level in args.levels):
        raise ValueError("Adventure levels must be in [1, 50]")

    stage_by_level = {
        adventure_level_number(stage): stage for stage in campaign_stages()
    }
    reference_save = tempfile.TemporaryDirectory(prefix="pvzwm-campaign-reference-")
    candidate_save = tempfile.TemporaryDirectory(prefix="pvzwm-campaign-candidate-")
    common = {
        "resources": args.resources,
        "xvfb_run": None,
        "frame_width": 0,
        "frame_height": 0,
    }
    reference = PortableBridgeClient(
        PortableProcessConfig(
            binary=args.binary,
            save_directory=Path(reference_save.name),
            **common,
        )
    )
    candidate = (
        PortableBridgeClient(
            PortableProcessConfig(
                binary=args.candidate_binary,
                save_directory=Path(candidate_save.name),
                **common,
            )
        )
        if args.candidate_binary is not None
        else None
    )

    comparisons = 0
    interaction_results: list[dict[str, Any]] = []
    try:
        for level in args.levels:
            stage = stage_by_level[level]
            previous_checkpoint = 0
            for checkpoint in args.checkpoints:
                reference_payload = _request(
                    reference,
                    level=level,
                    checkpoint=checkpoint,
                    previous_checkpoint=previous_checkpoint,
                    seed=args.seed,
                )
                errors = validate_adventure_snapshot(
                    stage,
                    reference_payload,
                    checkpoint=checkpoint,
                )
                state_difference = None
                if candidate is not None:
                    candidate_payload = _request(
                        candidate,
                        level=level,
                        checkpoint=checkpoint,
                        previous_checkpoint=previous_checkpoint,
                        seed=args.seed,
                    )
                    candidate_errors = validate_adventure_snapshot(
                        stage,
                        candidate_payload,
                        checkpoint=checkpoint,
                    )
                    errors.extend(
                        f"candidate: {error}" for error in candidate_errors
                    )
                    state_difference = first_snapshot_difference(
                        canonical_snapshot(reference_payload),
                        canonical_snapshot(candidate_payload),
                    )
                    if state_difference is not None:
                        errors.append(f"candidate state: {state_difference}")

                record = {
                    "level": stage.stage_id,
                    "mode": stage.mode,
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
                        f"campaign audit failed at {stage.stage_id}, "
                        f"tick {checkpoint}: {'; '.join(errors)}"
                    )
                comparisons += 1
                previous_checkpoint = checkpoint
        if args.exercise_special_actions:
            interaction_results = exercise_special_interactions(
                reference,
                candidate,
                seed=args.seed,
            )
            for result in interaction_results:
                print(json.dumps(result, allow_nan=False))
    finally:
        if candidate is not None:
            candidate.close()
        reference.close()
        candidate_save.cleanup()
        reference_save.cleanup()

    print(
        json.dumps(
            {
                "levels": len(args.levels),
                "comparisons": comparisons,
                "checkpoints": list(args.checkpoints),
                "special_interactions": len(interaction_results),
                "invariants_valid": True,
                "state_identical": True if candidate is not None else None,
            }
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from pvzwm.goty_playthrough import NativeBoardDriver, recommended_seed_types
from pvzwm.portable import PortableBridgeClient, PortableProcessConfig
from pvzwm.portable_trace import canonical_snapshot, first_snapshot_difference


def parse_integer_spec(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ".." in item:
            start_text, stop_text = item.split("..", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise argparse.ArgumentTypeError(f"range must be increasing: {item!r}")
            values.extend(range(start, stop + 1))
        else:
            values.append(int(item))
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("levels must be nonempty and unique")
    return tuple(values)


def parse_seed_types(value: str) -> tuple[int, ...]:
    seed_types = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seed_types or len(seed_types) > 10:
        raise argparse.ArgumentTypeError("seed loadout must contain 1 to 10 types")
    if len(seed_types) != len(set(seed_types)):
        raise argparse.ArgumentTypeError("seed loadout contains duplicates")
    if any(seed_type < 0 or seed_type >= 48 for seed_type in seed_types):
        raise argparse.ArgumentTypeError("seed types must be in [0, 48)")
    return seed_types


def request_pair(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient | None,
    command: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    payload = reference.request(command)
    if candidate is not None:
        candidate_payload = candidate.request(command)
        difference = first_snapshot_difference(
            canonical_snapshot(payload),
            canonical_snapshot(candidate_payload),
        )
        if difference is not None:
            raise RuntimeError(f"{context}: renderer state diverged: {difference}")
    return payload


def play_level(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient | None,
    *,
    level: int,
    seed: int,
    ticks_per_action: int,
    max_ticks: int,
    seed_types: tuple[int, ...] | None,
    trace_decisions: bool = False,
) -> dict[str, Any]:
    reset_command: dict[str, Any] = {
        "cmd": "reset",
        "level": level,
        "seed": seed,
        "frame_width": 0,
        "frame_height": 0,
    }
    if seed_types is not None:
        reset_command["seed_types"] = list(seed_types)
    payload = request_pair(
        reference,
        candidate,
        reset_command,
        context=f"level {level} reset",
    )
    selected_seed_types = seed_types
    if selected_seed_types is None and bool(payload.get("choose_seeds", False)):
        selected_seed_types = recommended_seed_types(payload)
        if len(selected_seed_types) != int(payload["seed_bank_capacity"]):
            raise RuntimeError(
                f"level {level}: could not construct a complete legal loadout"
            )
        reset_command["seed_types"] = list(selected_seed_types)
        payload = request_pair(
            reference,
            candidate,
            reset_command,
            context=f"level {level} terrain-aware loadout reset",
        )
    driver = NativeBoardDriver()
    ticks = 0
    actions = 0
    planted = 0
    collected = 0
    clicked = 0
    while not bool(payload.get("done", False)) and ticks < max_ticks:
        action = driver.choose_action(payload)
        if action not in {int(item) for item in payload["legal_actions"]}:
            raise RuntimeError(f"level {level}: driver selected illegal action {action}")
        if trace_decisions and action != 0:
            print(
                json.dumps(
                    {
                        "trace": "before_action",
                        "level_number": level,
                        "ticks": ticks,
                        "action": action,
                        "vase_seed_row": driver._vase_seed_row,
                        "vase_seed_type": driver._vase_seed_type,
                        "cursor_type": payload.get("cursor_type"),
                        "cursor_seed_type": payload.get("cursor_seed_type"),
                        "plants": [
                            {
                                "type": int(plant["type"]),
                                "row": int(plant["row"]),
                                "col": int(plant["col"]),
                            }
                            for plant in payload.get("plants", [])
                        ],
                        "zombies": [
                            {
                                "type": int(zombie["type"]),
                                "row": int(zombie["row"]),
                                "x": float(zombie["x"]),
                            }
                            for zombie in payload.get("zombies", [])
                            if not bool(zombie.get("mind_controlled", False))
                        ],
                        "coins": [
                            {
                                "type": int(coin["type"]),
                                "being_collected": bool(
                                    coin.get("being_collected", False)
                                ),
                            }
                            for coin in payload.get("coins", [])
                        ],
                        "pots": [
                            {
                                "row": int(item["row"]),
                                "col": int(item["col"]),
                                "seed_type": int(item.get("seed_type", -1)),
                                "zombie_type": int(item.get("zombie_type", -1)),
                            }
                            for item in payload.get("grid_items", [])
                            if int(item.get("type", -1)) == 7
                        ],
                    },
                    allow_nan=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        if 1 <= action < driver.codec.shovel_begin:
            planted += 1
        elif driver.codec.board_click_begin <= action < driver.codec.collect_action:
            clicked += 1
        elif action == driver.codec.collect_action:
            collected += 1
        step_ticks = min(ticks_per_action, max_ticks - ticks)
        payload = request_pair(
            reference,
            candidate,
            {
                "cmd": "step",
                "action": action,
                "ticks": step_ticks,
                "frame_width": 0,
                "frame_height": 0,
            },
            context=f"level {level} tick {ticks} action {action}",
        )
        ticks += step_ticks
        actions += 1

    record = {
        "level": f"{int(payload['area'])}-{int(payload['stage'])}",
        "level_number": level,
        "ticks": ticks,
        "actions": actions,
        "planted": planted,
        "collected": collected,
        "board_clicks": clicked,
        "won": bool(payload.get("won", False)),
        "lost": int(payload.get("board_result", 0)) == 2,
        "terminal": bool(payload.get("done", False)),
        "final_wave": int(payload.get("wave", -1)),
        "spawned_waves": int(payload.get("total_spawned_waves", -1)),
        "renderer_state_identical": candidate is not None,
        "seed_types": selected_seed_types,
    }
    if not record["won"]:
        record["failure_state"] = {
            "sun": int(payload.get("sun", 0)),
            "seed_packets": payload.get("seed_packets", []),
            "plants": [
                {
                    "type": int(plant["type"]),
                    "row": int(plant["row"]),
                    "col": int(plant["col"]),
                }
                for plant in payload.get("plants", [])
            ],
            "zombies": [
                {
                    "type": int(zombie["type"]),
                    "row": int(zombie["row"]),
                    "x": float(zombie["x"]),
                    "body_health": int(zombie["body_health"]),
                    "helm_health": int(zombie["helm_health"]),
                }
                for zombie in payload.get("zombies", [])
            ],
            "mowers": payload.get("mowers", []),
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Drive authentic GOTY Adventure boards through the native player "
            "action surface and require legal terminal wins"
        )
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument(
        "--levels",
        type=parse_integer_spec,
        default=parse_integer_spec("1..10"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seed-types",
        type=parse_seed_types,
        help="explicit legal seed loadout; only valid when auditing one level",
    )
    parser.add_argument("--ticks-per-action", type=int, default=10)
    parser.add_argument("--max-ticks", type=int, default=120_000)
    parser.add_argument(
        "--trace-decisions",
        action="store_true",
        help="write non-wait driver decisions and visible board state to stderr",
    )
    args = parser.parse_args()

    if any(level < 1 or level > 50 for level in args.levels):
        raise ValueError("Adventure levels must be in [1, 50]")
    if args.ticks_per_action < 1 or args.max_ticks < 1:
        raise ValueError("tick counts must be positive")
    if args.seed_types is not None and len(args.levels) != 1:
        raise ValueError("--seed-types requires exactly one level")

    reference_save = tempfile.TemporaryDirectory(prefix="pvzwm-play-reference-")
    candidate_save = tempfile.TemporaryDirectory(prefix="pvzwm-play-candidate-")
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

    records: list[dict[str, Any]] = []
    try:
        for level in args.levels:
            record = play_level(
                reference,
                candidate,
                level=level,
                seed=args.seed,
                ticks_per_action=args.ticks_per_action,
                max_ticks=args.max_ticks,
                seed_types=args.seed_types,
                trace_decisions=args.trace_decisions,
            )
            print(json.dumps(record, allow_nan=False), flush=True)
            records.append(record)
            if not record["won"]:
                raise RuntimeError(
                    f"level {record['level']} did not reach a legal win: {record}"
                )
    finally:
        if candidate is not None:
            candidate.close()
        reference.close()
        candidate_save.cleanup()
        reference_save.cleanup()

    print(
        json.dumps(
            {
                "levels": len(records),
                "wins": sum(bool(record["won"]) for record in records),
                "all_terminal": all(bool(record["terminal"]) for record in records),
                "renderer_state_identical": candidate is not None,
            }
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from pvzwm.portable import (
    PortableActionCodec,
    PortableActionKind,
    PortableEnv,
    PortableProcessConfig,
)


def plants_at(payload: dict[str, object], row: int, col: int) -> tuple[int, ...]:
    plants = payload.get("plants", [])
    if not isinstance(plants, list):
        return ()
    return tuple(
        sorted(
            int(plant["type"])
            for plant in plants
            if isinstance(plant, dict)
            and int(plant.get("row", -1)) == row
            and int(plant.get("col", -1)) == col
        )
    )


def packet_type(payload: dict[str, object], slot: int) -> int | None:
    packets = payload.get("seed_packets", [])
    if not isinstance(packets, list):
        return None
    for packet in packets:
        if isinstance(packet, dict) and int(packet.get("slot", -1)) == slot:
            return int(packet["type"])
    return None


def deepest_unique_cells(
    payload: dict[str, object],
    limit: int,
    require_plant_slot: int | None = None,
) -> list[dict[str, object]]:
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Archive payload has no cell list")
    ordered = sorted(
        (cell for cell in cells if isinstance(cell, dict)),
        key=lambda cell: len(cell.get("actions", [])),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for cell in ordered:
        actions = tuple(int(action) for action in cell.get("actions", []))
        if require_plant_slot is not None:
            begin = (
                PortableActionCodec.plant_begin
                + require_plant_slot * PortableActionCodec.cells
            )
            if not any(begin <= action < begin + PortableActionCodec.cells for action in actions):
                continue
        identity = (int(cell["seed"]), actions)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(cell)
        if len(selected) == limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay deep Go-Explore prefixes and audit generic plant stacking"
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--level", type=int, default=21)
    parser.add_argument("--prefixes", type=int, default=12)
    parser.add_argument(
        "--require-plant-slot",
        type=int,
        choices=range(PortableActionCodec.seed_slots),
        help="Audit only prefixes containing an action from this seed slot",
    )
    parser.add_argument("--ticks-per-step", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prefixes < 1:
        raise SystemExit("--prefixes must be positive")

    with gzip.open(args.archive, "rt", encoding="utf-8") as handle:
        archive_payload = json.load(handle)
    selected = deepest_unique_cells(
        archive_payload, args.prefixes, args.require_plant_slot
    )
    codec = PortableActionCodec()
    archived_required_slot_actions: list[dict[str, object]] = []
    if args.require_plant_slot is not None:
        for cell in selected:
            actions = [int(action) for action in cell.get("actions", [])]
            for step, action in enumerate(actions):
                decoded = codec.decode(action)
                if (
                    decoded.kind != PortableActionKind.PLANT
                    or decoded.slot != args.require_plant_slot
                ):
                    continue
                prior_same_cell: list[dict[str, int]] = []
                for prior_step, prior_action in enumerate(actions[:step]):
                    prior = codec.decode(prior_action)
                    if (
                        prior.kind == PortableActionKind.PLANT
                        and prior.row == decoded.row
                        and prior.col == decoded.col
                    ):
                        prior_same_cell.append(
                            {"step": prior_step, "slot": prior.slot}
                        )
                archived_required_slot_actions.append(
                    {
                        "seed": int(cell["seed"]),
                        "prefix_depth": len(actions),
                        "step": step,
                        "action": action,
                        "slot": decoded.slot,
                        "row": decoded.row,
                        "col": decoded.col,
                        "prior_plant_actions_same_cell": prior_same_cell,
                    }
                )
    stack_pairs: Counter[tuple[tuple[int, ...], int | None]] = Counter()
    selected_types: Counter[int | None] = Counter()
    selected_rows: Counter[int] = Counter()
    events: list[dict[str, object]] = []
    invalid_events: list[dict[str, object]] = []
    restored_prefixes = 0
    invalid_prefixes = 0
    maximum_restored_depth = 0
    plant_actions = 0
    stack_actions = 0

    config = PortableProcessConfig(
        binary=args.binary,
        resources=args.resources,
        level=args.level,
        ticks_per_step=args.ticks_per_step,
        max_steps=3_600,
        frame_width=0,
        frame_height=0,
        reward_mode="clipped-damage",
    )
    for cell in selected:
        actions = [int(action) for action in cell.get("actions", [])]
        environment = PortableEnv(config)
        valid = True
        try:
            observation, _ = environment.reset(seed=int(cell["seed"]))
            for step, action in enumerate(actions):
                if not environment.legal_action_mask()[action]:
                    decoded = codec.decode(action)
                    invalid_events.append(
                        {
                            "seed": int(cell["seed"]),
                            "prefix_depth": len(actions),
                            "step": step,
                            "reason": "action_became_illegal",
                            "action": action,
                            "kind": decoded.kind.name,
                            "slot": decoded.slot,
                            "row": decoded.row,
                            "col": decoded.col,
                            "sun": observation.payload.get("sun"),
                            "wave": observation.payload.get("wave"),
                            "plants_count": observation.payload.get("plants_count"),
                            "legal_action_count": int(
                                environment.legal_action_mask().sum()
                            ),
                        }
                    )
                    valid = False
                    break
                decoded = codec.decode(action)
                before_payload = observation.payload
                before_types: tuple[int, ...] = ()
                chosen_type: int | None = None
                if decoded.kind == PortableActionKind.PLANT:
                    plant_actions += 1
                    before_types = plants_at(
                        before_payload, decoded.row, decoded.col
                    )
                    chosen_type = packet_type(before_payload, decoded.slot)
                    selected_types[chosen_type] += 1
                    selected_rows[decoded.row] += 1
                observation, reward, terminated, truncated, _ = environment.step(action)
                if decoded.kind == PortableActionKind.PLANT and before_types:
                    after_types = plants_at(
                        observation.payload, decoded.row, decoded.col
                    )
                    stack_actions += 1
                    stack_pairs[(before_types, chosen_type)] += 1
                    events.append(
                        {
                            "seed": int(cell["seed"]),
                            "prefix_depth": len(actions),
                            "step": step,
                            "slot": decoded.slot,
                            "row": decoded.row,
                            "col": decoded.col,
                            "selected_type": chosen_type,
                            "before_types": list(before_types),
                            "after_types": list(after_types),
                            "reward": reward,
                        }
                    )
                if terminated or truncated:
                    valid = step + 1 == len(actions)
                    if not valid:
                        invalid_events.append(
                            {
                                "seed": int(cell["seed"]),
                                "prefix_depth": len(actions),
                                "step": step,
                                "reason": "episode_ended_before_prefix",
                                "won": observation.won,
                                "lost": observation.lost,
                            }
                        )
                    break
            if valid:
                restored_prefixes += 1
                maximum_restored_depth = max(maximum_restored_depth, len(actions))
            else:
                invalid_prefixes += 1
        finally:
            environment.close()

    cells = archive_payload.get("cells", [])
    result = {
        "archive": str(args.archive),
        "level": args.level,
        "archive_cells": len(cells) if isinstance(cells, list) else 0,
        "archive_maximum_depth": max(
            (len(cell.get("actions", [])) for cell in cells if isinstance(cell, dict)),
            default=0,
        ),
        "prefixes_requested": args.prefixes,
        "required_plant_slot": args.require_plant_slot,
        "archived_required_slot_actions": archived_required_slot_actions,
        "prefixes_replayed": len(selected),
        "restored_prefixes": restored_prefixes,
        "invalid_prefixes": invalid_prefixes,
        "maximum_restored_depth": maximum_restored_depth,
        "plant_actions": plant_actions,
        "stack_actions": stack_actions,
        "selected_seed_types": [
            {"type": seed_type, "count": count}
            for seed_type, count in selected_types.most_common()
        ],
        "selected_rows": [
            {"row": row, "count": count}
            for row, count in sorted(selected_rows.items())
        ],
        "stack_pairs": [
            {
                "before_types": list(before_types),
                "selected_type": seed_type,
                "count": count,
            }
            for (before_types, seed_type), count in stack_pairs.most_common()
        ],
        "stack_events": events,
        "invalid_events": invalid_events,
        "reward_shaping_used": False,
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from pvzwm.goty_modes import GAME_MODE_BY_ID
from pvzwm.goty_playthrough import NativeBoardDriver, recommended_mode_seed_types
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
        raise argparse.ArgumentTypeError("modes must be nonempty and unique")
    return tuple(values)


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


def mode_terminal(game_mode: int, payload: dict[str, Any]) -> bool:
    """Return whether the complete secondary mode, not one stage, has ended."""

    if int(payload.get("board_result", 0)) == 2:
        return True
    if not bool(payload.get("won", False)):
        return False
    if 1 <= game_mode <= 10:
        return int(payload.get("survival_stage", -1)) >= 4
    return True


def play_mode(
    reference: PortableBridgeClient,
    candidate: PortableBridgeClient | None,
    *,
    game_mode: int,
    seed: int,
    ticks_per_action: int,
    max_ticks: int,
    trace_decisions: bool = False,
) -> dict[str, Any]:
    spec = GAME_MODE_BY_ID[game_mode]
    reset_command: dict[str, Any] = {
        "cmd": "reset",
        "game_mode": game_mode,
        "seed": seed,
        "frame_width": 0,
        "frame_height": 0,
    }
    if game_mode == 5:
        # Roof Cleaners are ordinary Crazy Dave purchases that persist on a
        # human GOTY profile. Model that equipment rather than testing an
        # artificially unequipped throwaway profile.
        reset_command["purchased_roof_cleaner"] = 1
    if game_mode in {3, 4}:
        # Pool Cleaners are the corresponding persistent Crazy Dave purchase
        # for Survival: Pool and Survival: Fog.
        reset_command["purchased_pool_cleaner"] = 1
    if game_mode in {2, 4}:
        # Completed human profiles can carry one purchased packet slot into
        # secondary modes. Survival: Night/Fog use it for their upgrade cores.
        reset_command["packet_upgrades"] = 1
    if game_mode in {2, 4}:
        # The late Survival modes also exercise Crazy Dave seed upgrades. This
        # exposes the human-legal Gloom-shroom and Cattail strategies instead
        # of constraining the gate to a base-seed throwaway profile.
        reset_command["purchased_seed_upgrades"] = 1
    payload = request_pair(
        reference,
        candidate,
        reset_command,
        context=f"{spec.name} reset",
    )
    selected_seed_types: tuple[int, ...] | None = None
    if bool(payload.get("choose_seeds", False)):
        selected_seed_types = recommended_mode_seed_types(payload)
        # Seeing Stars begins with Starfruit already fixed in slot zero. The
        # bridge currently cannot replace that preselected packet, so retain
        # the game's valid default loadout for this one challenge.
        if game_mode == 22:
            selected_seed_types = tuple(
                int(packet["type"]) for packet in payload.get("seed_packets", [])
            )
        else:
            payload = request_pair(
                reference,
                candidate,
                {
                    **reset_command,
                    "seed_types": list(selected_seed_types),
                },
                context=f"{spec.name} optimized loadout",
            )

    driver = NativeBoardDriver()
    ticks = 0
    actions = 0
    planted = 0
    collected = 0
    clicked = 0
    interactions = 0
    planted_by_seed_type: Counter[int] = Counter()
    planted_by_survival_stage: Counter[int] = Counter()
    planted_by_stage_and_seed_type: Counter[tuple[int, int]] = Counter()
    previous_survival_stage = int(payload.get("survival_stage", -1))
    if trace_decisions:
        print(
            json.dumps(
                {
                    "event": "reset",
                    "game_mode": game_mode,
                    "sun": int(payload.get("sun", -1)),
                    "seed_packets": payload.get("seed_packets", []),
                    "plants": payload.get("plants", []),
                    "grid_items": payload.get("grid_items", []),
                    "legal_actions": payload.get("legal_actions", []),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    while not mode_terminal(game_mode, payload) and ticks < max_ticks:
        interaction = driver.choose_interaction(payload)
        if interaction is not None:
            interaction_command: dict[str, Any] = {
                "cmd": "interact",
                "interaction": interaction,
                "frame_width": 0,
                "frame_height": 0,
            }
            if interaction == "survival_repick":
                selected_seed_types = recommended_mode_seed_types(
                    payload,
                    survival_repick=True,
                )
                if not selected_seed_types:
                    raise RuntimeError(f"{spec.name}: survival repick has no loadout")
                interaction_command["seed_types"] = list(selected_seed_types)
            payload = request_pair(
                reference,
                candidate,
                interaction_command,
                context=f"{spec.name} interaction {interaction}",
            )
            interactions += 1
            if trace_decisions:
                print(
                    json.dumps(
                        {
                            "event": "interaction",
                            "interaction": interaction,
                            "ticks": ticks,
                            "survival_stage": int(payload.get("survival_stage", -1)),
                            "sun": int(payload.get("sun", -1)),
                            "plants": len(payload.get("plants", [])),
                            "zombies": len(payload.get("zombies", [])),
                            "mowers_ready": int(payload.get("mowers_ready", -1)),
                            "mowers_triggered": int(
                                payload.get("mowers_triggered", -1)
                            ),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            continue
        intermediate_survival_win = (
            1 <= game_mode <= 10
            and bool(payload.get("won", False))
            and int(payload.get("survival_stage", -1)) < 4
        )
        action = 0 if intermediate_survival_win else driver.choose_action(payload)
        legal = {int(item) for item in payload["legal_actions"]}
        if action not in legal:
            raise RuntimeError(f"{spec.name}: driver selected illegal action {action}")
        if driver.codec.plant_begin <= action < driver.codec.shovel_begin:
            planted += 1
            planted_by_survival_stage[int(payload.get("survival_stage", -1))] += 1
            planted_slot = driver.codec.decode(action).slot
            planted_seed_type = next(
                (
                    int(packet["type"])
                    for packet in payload.get("seed_packets", [])
                    if int(packet["slot"]) == planted_slot
                ),
                -1,
            )
            planted_by_seed_type[planted_seed_type] += 1
            planted_by_stage_and_seed_type[
                (int(payload.get("survival_stage", -1)), planted_seed_type)
            ] += 1
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
            context=f"{spec.name} tick {ticks} action {action}",
        )
        ticks += step_ticks
        actions += 1
        survival_stage = int(payload.get("survival_stage", -1))
        if trace_decisions and (
            action not in {0, driver.codec.collect_action}
            or survival_stage != previous_survival_stage
            or int(payload.get("board_result", 0)) == 2
        ):
            decoded = driver.codec.decode(action)
            print(
                json.dumps(
                    {
                        "event": "step",
                        "ticks": ticks,
                        "action": action,
                        "action_kind": decoded.kind,
                        "slot": decoded.slot,
                        "row": decoded.row,
                        "col": decoded.col,
                        "survival_stage": survival_stage,
                        "wave": int(payload.get("wave", -1)),
                        "sun": int(payload.get("sun", -1)),
                        "plants": len(payload.get("plants", [])),
                        "zombies": len(payload.get("zombies", [])),
                        "mowers_ready": int(payload.get("mowers_ready", -1)),
                        "mowers_triggered": int(payload.get("mowers_triggered", -1)),
                        "zombie_state": payload.get("zombies", [])
                        if int(payload.get("board_result", 0)) == 2
                        else None,
                        "plant_layout": [
                            [
                                int(plant.get("type", -1)),
                                int(plant.get("row", -1)),
                                int(plant.get("col", -1)),
                            ]
                            for plant in payload.get("plants", [])
                        ],
                        "brain_rows": [
                            int(item.get("row", -1))
                            for item in payload.get("grid_items", [])
                            if int(item.get("type", -1)) == 12
                        ],
                        "initial_grid_item_state": payload.get("grid_items", [])
                        if ticks == step_ticks
                        else None,
                        "board_result": int(payload.get("board_result", 0)),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
        previous_survival_stage = survival_stage

    completed = mode_terminal(game_mode, payload)
    engine_won = bool(payload.get("won", False))
    return {
        "game_mode": game_mode,
        "name": spec.name,
        "family": spec.family,
        "ticks": ticks,
        "actions": actions,
        "planted": planted,
        "planted_by_seed_type": dict(sorted(planted_by_seed_type.items())),
        "planted_by_survival_stage": dict(sorted(planted_by_survival_stage.items())),
        "planted_by_stage_and_seed_type": {
            stage: {
                seed_type: count
                for (item_stage, seed_type), count in sorted(
                    planted_by_stage_and_seed_type.items()
                )
                if item_stage == stage
            }
            for stage in sorted(
                {item_stage for item_stage, _ in planted_by_stage_and_seed_type}
            )
        },
        "collected": collected,
        "board_clicks": clicked,
        "interactions": interactions,
        "won": completed and engine_won,
        "engine_won": engine_won,
        "lost": int(payload.get("board_result", 0)) == 2,
        "terminal": completed,
        "final_wave": int(payload.get("wave", -1)),
        "spawned_waves": int(payload.get("total_spawned_waves", -1)),
        "challenge_state": int(payload.get("challenge_state", -1)),
        "survival_stage": int(payload.get("survival_stage", -1)),
        "next_survival_stage_counter": int(
            payload.get("next_survival_stage_counter", -1)
        ),
        "last_stand_button_visible": bool(
            payload.get("last_stand_button_visible", False)
        ),
        "last_stand_button_disabled": bool(
            payload.get("last_stand_button_disabled", True)
        ),
        "board_buttons_interactable": bool(
            payload.get("board_buttons_interactable", False)
        ),
        "renderer_state_identical": candidate is not None,
        "seed_types": selected_seed_types,
        "packet_upgrades": int(payload.get("packet_upgrades", 0)),
        "purchased_seed_upgrades": bool(
            reset_command.get("purchased_seed_upgrades", 0)
        ),
        "purchased_pool_cleaner": bool(reset_command.get("purchased_pool_cleaner", 0)),
        "purchased_roof_cleaner": bool(reset_command.get("purchased_roof_cleaner", 0)),
        "final_mowers_ready": int(payload.get("mowers_ready", -1)),
        "mowers_triggered": int(payload.get("mowers_triggered", -1)),
        "final_mowers": payload.get("mowers", []),
        "final_breach_zombies": [
            zombie
            for zombie in payload.get("zombies", [])
            if float(zombie.get("x", 9999.0)) < 150.0
        ],
        "final_plant_types": dict(
            sorted(
                Counter(
                    int(plant.get("type", -1)) for plant in payload.get("plants", [])
                ).items()
            )
        ),
        "final_plants_by_row": dict(
            sorted(
                Counter(
                    int(plant.get("row", -1)) for plant in payload.get("plants", [])
                ).items()
            )
        ),
        "final_zombies_by_row": dict(
            sorted(
                Counter(
                    int(zombie.get("row", -1))
                    for zombie in payload.get("zombies", [])
                    if not bool(zombie.get("mind_controlled", False))
                ).items()
            )
        ),
        "final_legal_action_count": len(payload.get("legal_actions", [])),
        "remaining_brain_rows": [
            int(item.get("row", -1))
            for item in payload.get("grid_items", [])
            if int(item.get("type", -1)) == 12
        ],
    }


def play_isolated_mode(
    *,
    binary: Path,
    candidate_binary: Path | None,
    resources: Path,
    game_mode: int,
    seed: int,
    ticks_per_action: int,
    max_ticks: int,
    trace_decisions: bool = False,
) -> dict[str, Any]:
    """Run one mode in fresh native processes and private save directories."""

    common = {
        "resources": resources,
        "xvfb_run": None,
        "frame_width": 0,
        "frame_height": 0,
    }
    with (
        tempfile.TemporaryDirectory(
            prefix=f"pvzwm-mode-{game_mode}-reference-"
        ) as reference_save,
        tempfile.TemporaryDirectory(
            prefix=f"pvzwm-mode-{game_mode}-candidate-"
        ) as candidate_save,
    ):
        reference = PortableBridgeClient(
            PortableProcessConfig(
                binary=binary,
                save_directory=Path(reference_save),
                **common,
            )
        )
        candidate = (
            PortableBridgeClient(
                PortableProcessConfig(
                    binary=candidate_binary,
                    save_directory=Path(candidate_save),
                    **common,
                )
            )
            if candidate_binary is not None
            else None
        )
        try:
            return play_mode(
                reference,
                candidate,
                game_mode=game_mode,
                seed=seed,
                ticks_per_action=ticks_per_action,
                max_ticks=max_ticks,
                trace_decisions=trace_decisions,
            )
        finally:
            if candidate is not None:
                candidate.close()
            reference.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Drive finite GOTY secondary modes through the native player-action "
            "surface and require authentic terminal wins."
        )
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--modes", type=parse_integer_spec, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ticks-per-action", type=int, default=10)
    parser.add_argument("--max-ticks", type=int, default=360_000)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="report non-winning modes without failing on the first one",
    )
    parser.add_argument(
        "--trace-decisions",
        action="store_true",
        help="write non-wait decisions and Survival stage transitions to stderr",
    )
    args = parser.parse_args()

    unknown_modes = sorted(set(args.modes) - GAME_MODE_BY_ID.keys())
    if unknown_modes:
        raise ValueError(f"unsupported secondary game modes: {unknown_modes}")
    if args.ticks_per_action < 1 or args.max_ticks < 1:
        raise ValueError("tick counts must be positive")

    records: list[dict[str, Any]] = []
    for game_mode in args.modes:
        record = play_isolated_mode(
            binary=args.binary,
            candidate_binary=args.candidate_binary,
            resources=args.resources,
            game_mode=game_mode,
            seed=args.seed,
            ticks_per_action=args.ticks_per_action,
            max_ticks=args.max_ticks,
            trace_decisions=args.trace_decisions,
        )
        print(json.dumps(record, allow_nan=False), flush=True)
        records.append(record)
        if not record["won"] and not args.allow_incomplete:
            raise RuntimeError(
                f"mode {record['name']} did not reach a legal win: {record}"
            )

    print(
        json.dumps(
            {
                "modes": len(records),
                "wins": sum(bool(record["won"]) for record in records),
                "terminal": sum(bool(record["terminal"]) for record in records),
                "renderer_state_identical": args.candidate_binary is not None,
                "incomplete": [
                    record["game_mode"] for record in records if not bool(record["won"])
                ],
            }
        )
    )


if __name__ == "__main__":
    main()

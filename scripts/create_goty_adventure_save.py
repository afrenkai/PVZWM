#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from pvzwm.goty_playthrough import NativeBoardDriver, recommended_seed_types
from pvzwm.portable import PortableBridgeClient, PortableProcessConfig
from pvzwm.teacher_demo import read_checkpoint_profiles


def parse_adventure_level(value: str) -> int:
    """Parse either a campaign number or the familiar ``world-stage`` form."""

    if "-" not in value:
        level = int(value)
    else:
        parts = value.split("-")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"invalid Adventure level: {value!r}")
        world, stage = (int(part) for part in parts)
        if not 1 <= world <= 5 or not 1 <= stage <= 10:
            raise argparse.ArgumentTypeError(
                f"Adventure world-stage must be in 1-1..5-10, got {value!r}"
            )
        level = (world - 1) * 10 + stage
    if not 2 <= level <= 50:
        raise argparse.ArgumentTypeError(
            "target must have a preceding Adventure level in 1-1..5-9"
        )
    return level


def default_save_root() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library/Application Support/PVZWM/save-slots"
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return data_home / "pvzwm/save-slots"


def complete_previous_level(
    client: PortableBridgeClient,
    *,
    target_level: int,
    seed: int,
    ticks_per_action: int,
    max_ticks: int,
) -> dict[str, Any]:
    previous_level = target_level - 1
    reset_command: dict[str, Any] = {
        "cmd": "reset",
        "level": previous_level,
        "seed": seed,
        "frame_width": 0,
        "frame_height": 0,
    }
    payload = client.request(reset_command)
    selected_seed_types: tuple[int, ...] | None = None
    if bool(payload.get("choose_seeds", False)):
        selected_seed_types = recommended_seed_types(payload)
        if len(selected_seed_types) != int(payload["seed_bank_capacity"]):
            raise RuntimeError("could not construct a complete legal seed loadout")
        payload = client.request(
            {**reset_command, "seed_types": list(selected_seed_types)}
        )

    driver = NativeBoardDriver()
    ticks = 0
    actions = 0
    while not bool(payload.get("done", False)) and ticks < max_ticks:
        action = driver.choose_action(payload)
        if action not in {int(item) for item in payload["legal_actions"]}:
            raise RuntimeError(f"driver selected illegal action {action}")
        step_ticks = min(ticks_per_action, max_ticks - ticks)
        payload = client.request(
            {
                "cmd": "step",
                "action": action,
                "ticks": step_ticks,
                "frame_width": 0,
                "frame_height": 0,
            }
        )
        ticks += step_ticks
        actions += 1

    if not bool(payload.get("won", False)):
        raise RuntimeError(
            f"native engine did not win level {previous_level} within {max_ticks} ticks"
        )

    # Adventure ``WON`` first means the final zombie dropped the native level
    # award. A human still collects that award and waits for the fade before
    # the outer app commits profile progression. Drive those ordinary actions
    # instead of treating the earlier combat result as a save boundary.
    reward_ticks = 0
    reward_actions = 0
    while not bool(payload.get("level_complete", False)) and reward_ticks < 10_000:
        legal = {int(item) for item in payload["legal_actions"]}
        action = (
            driver.codec.collect_action
            if driver.codec.collect_action in legal
            else 0
        )
        payload = client.request(
            {
                "cmd": "step",
                "action": action,
                "ticks": 1,
                "frame_width": 0,
                "frame_height": 0,
            }
        )
        reward_ticks += 1
        reward_actions += 1
    if not bool(payload.get("level_complete", False)):
        raise RuntimeError("native Adventure reward did not reach level completion")

    finalized = client.request({"cmd": "finalize_win"})
    if int(finalized.get("adventure_level", -1)) != target_level:
        raise RuntimeError(
            "native win did not advance the profile to the target level: "
            f"{finalized}"
        )
    return {
        "completed_level": previous_level,
        "finalized_adventure_level": int(finalized["adventure_level"]),
        "ticks": ticks,
        "actions": actions,
        "reward_ticks": reward_ticks,
        "reward_actions": reward_actions,
        "seed_types": selected_seed_types,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an isolated human save immediately before an Adventure level "
            "by legally completing its preceding level with the native Metal engine."
        )
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--save-slot", required=True)
    parser.add_argument("--save-root", type=Path, default=default_save_root())
    parser.add_argument("--target-level", type=parse_adventure_level, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ticks-per-action", type=int, default=10)
    parser.add_argument("--max-ticks", type=int, default=180_000)
    parser.add_argument(
        "--direct-target",
        action="store_true",
        help=(
            "initialize the target with the native engine instead of completing "
            "the preceding stage; intended for isolated human fidelity testing"
        ),
    )
    args = parser.parse_args()

    if not args.save_slot or not all(
        character.isalnum() or character in "._-" for character in args.save_slot
    ):
        raise ValueError("save slot must contain only letters, digits, '.', '_' or '-'")
    if len(args.save_slot) > 64:
        raise ValueError("save slot must be at most 64 characters")
    if args.ticks_per_action < 1 or args.max_ticks < 1:
        raise ValueError("tick counts must be positive")

    save_root = args.save_root.expanduser().resolve()
    destination = save_root / args.save_slot
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing save slot: {destination}")
    save_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{args.save_slot}.", dir=save_root
    ) as temporary_name:
        temporary = Path(temporary_name)
        config = PortableProcessConfig(
            binary=args.binary.resolve(),
            resources=args.resources.resolve(),
            save_directory=temporary,
            xvfb_run=None,
            frame_width=0,
            frame_height=0,
        )
        client = PortableBridgeClient(config)
        try:
            if client.ready.get("renderer") != "metal":
                raise RuntimeError("checkpoint creation requires the Metal bridge")
            if args.direct_target:
                payload = client.request(
                    {
                        "cmd": "reset",
                        "level": args.target_level,
                        "seed": args.seed,
                        "frame_width": 0,
                        "frame_height": 0,
                    }
                )
                play_record = {
                    "method": "native-direct-target",
                    "initialized_level": int(payload["level"]),
                    "initialized_stage": (
                        f"{int(payload['area'])}-{int(payload['stage'])}"
                    ),
                }
            else:
                play_record = {
                    "method": "native-preceding-level-win",
                    **complete_previous_level(
                        client,
                        target_level=args.target_level,
                        seed=args.seed,
                        ticks_per_action=args.ticks_per_action,
                        max_ticks=args.max_ticks,
                    ),
                }
            client.request({"cmd": "flush_profile"})
        finally:
            client.close()

        profiles = read_checkpoint_profiles(temporary)
        matching_profiles = [
            profile
            for profile in profiles
            if profile.adventure_level == args.target_level
        ]
        if len(matching_profiles) != 1:
            raise RuntimeError(
                "native profile did not persist the requested target level: "
                f"{[profile.as_dict() for profile in profiles]}"
            )

        verification = PortableBridgeClient(config)
        try:
            loaded_profile = verification.ready.get("profile")
            if verification.ready.get("renderer") != "metal":
                raise RuntimeError("saved profile did not reopen with Metal")
            if not isinstance(loaded_profile, dict) or int(
                loaded_profile.get("adventure_level", -1)
            ) != args.target_level:
                raise RuntimeError(
                    f"reopened profile is not at level {args.target_level}: "
                    f"{loaded_profile}"
                )
        finally:
            verification.close()

        temporary.rename(destination)

    print(
        json.dumps(
            {
                "valid": True,
                "renderer": "metal",
                "save_slot": args.save_slot,
                "save_directory": str(destination),
                "target_level": args.target_level,
                "target_stage": (
                    f"{(args.target_level - 1) // 10 + 1}-"
                    f"{(args.target_level - 1) % 10 + 1}"
                ),
                "profile": matching_profiles[0].as_dict(),
                "bootstrap": play_record,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

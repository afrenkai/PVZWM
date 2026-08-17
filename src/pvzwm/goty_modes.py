from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pvzwm.portable import PortableActionCodec


@dataclass(frozen=True, slots=True)
class GameModeSpec:
    game_mode: int
    name: str
    family: str
    background: int
    seed_bank_capacity: int
    update_multiplier: int = 1

    @property
    def row_count(self) -> int:
        return 6 if self.background in {2, 3} else 5


SECONDARY_GAME_MODES = (
    GameModeSpec(1, "survival-day", "survival", 0, 6),
    GameModeSpec(2, "survival-night", "survival", 1, 6),
    GameModeSpec(3, "survival-pool", "survival", 2, 6),
    GameModeSpec(4, "survival-fog", "survival", 3, 6),
    GameModeSpec(5, "survival-roof", "survival", 4, 6),
    GameModeSpec(6, "survival-day-hard", "survival", 0, 6),
    GameModeSpec(7, "survival-night-hard", "survival", 1, 6),
    GameModeSpec(8, "survival-pool-hard", "survival", 2, 6),
    GameModeSpec(9, "survival-fog-hard", "survival", 3, 6),
    GameModeSpec(10, "survival-roof-hard", "survival", 4, 6),
    GameModeSpec(11, "survival-day-endless", "survival", 0, 6),
    GameModeSpec(12, "survival-night-endless", "survival", 1, 6),
    GameModeSpec(13, "survival-pool-endless", "survival", 2, 6),
    GameModeSpec(14, "survival-fog-endless", "survival", 3, 6),
    GameModeSpec(15, "survival-roof-endless", "survival", 4, 6),
    GameModeSpec(16, "war-and-peas", "mini-game", 0, 6),
    GameModeSpec(17, "wall-nut-bowling", "mini-game", 0, 10),
    GameModeSpec(18, "slot-machine", "mini-game", 0, 3),
    GameModeSpec(19, "its-raining-seeds", "mini-game", 3, 0),
    GameModeSpec(20, "beghouled", "mini-game", 1, 0),
    GameModeSpec(21, "invisighoul", "mini-game", 3, 10),
    GameModeSpec(22, "seeing-stars", "mini-game", 0, 6),
    GameModeSpec(23, "zombiquarium", "mini-game", 8, 2),
    GameModeSpec(24, "beghouled-twist", "mini-game", 1, 0),
    GameModeSpec(25, "little-trouble", "mini-game", 2, 10),
    GameModeSpec(26, "portal-combat", "mini-game", 1, 10),
    GameModeSpec(27, "column-like-you-see-em", "mini-game", 4, 10),
    GameModeSpec(28, "bobsled-bonanza", "mini-game", 2, 6),
    GameModeSpec(29, "zombie-nimble-zombie-quick", "mini-game", 2, 6, 2),
    GameModeSpec(30, "whack-a-zombie", "mini-game", 1, 3),
    GameModeSpec(31, "last-stand", "mini-game", 2, 6),
    GameModeSpec(32, "war-and-peas-2", "mini-game", 2, 6),
    GameModeSpec(33, "wall-nut-bowling-2", "mini-game", 0, 10),
    GameModeSpec(34, "pogo-party", "mini-game", 4, 6),
    GameModeSpec(35, "dr-zomboss-revenge", "mini-game", 5, 10),
    GameModeSpec(36, "art-challenge-wall-nut", "limbo", 0, 6),
    GameModeSpec(37, "sunny-day", "limbo", 0, 6),
    GameModeSpec(38, "unsodded", "limbo", 0, 6),
    GameModeSpec(39, "big-time", "limbo", 0, 6),
    GameModeSpec(40, "art-challenge-sunflower", "limbo", 0, 6),
    GameModeSpec(41, "air-raid", "limbo", 3, 6),
    GameModeSpec(42, "ice-level", "limbo", 0, 6),
    GameModeSpec(43, "zen-garden", "persistent", 7, 0),
    GameModeSpec(44, "high-gravity", "limbo", 4, 6),
    GameModeSpec(45, "grave-danger", "limbo", 1, 6),
    GameModeSpec(46, "can-you-dig-it", "limbo", 0, 10),
    GameModeSpec(47, "dark-stormy-night", "limbo", 3, 10),
    GameModeSpec(48, "bungee-blitz", "limbo", 4, 10),
    GameModeSpec(49, "squirrel", "limbo", 0, 0),
    GameModeSpec(50, "tree-of-wisdom", "persistent", 9, 0),
    GameModeSpec(51, "vasebreaker-1", "vasebreaker", 1, 1),
    GameModeSpec(52, "vasebreaker-2", "vasebreaker", 1, 1),
    GameModeSpec(53, "vasebreaker-3", "vasebreaker", 1, 1),
    GameModeSpec(54, "vasebreaker-4", "vasebreaker", 1, 1),
    GameModeSpec(55, "vasebreaker-5", "vasebreaker", 1, 1),
    GameModeSpec(56, "vasebreaker-6", "vasebreaker", 1, 1),
    GameModeSpec(57, "vasebreaker-7", "vasebreaker", 1, 1),
    GameModeSpec(58, "vasebreaker-8", "vasebreaker", 1, 1),
    GameModeSpec(59, "vasebreaker-9", "vasebreaker", 1, 1),
    GameModeSpec(60, "vasebreaker-endless", "vasebreaker", 1, 1),
    GameModeSpec(61, "i-zombie-1", "i-zombie", 1, 3),
    GameModeSpec(62, "i-zombie-2", "i-zombie", 1, 3),
    GameModeSpec(63, "i-zombie-3", "i-zombie", 1, 3),
    GameModeSpec(64, "i-zombie-4", "i-zombie", 1, 3),
    GameModeSpec(65, "i-zombie-5", "i-zombie", 1, 4),
    GameModeSpec(66, "i-zombie-6", "i-zombie", 1, 4),
    GameModeSpec(67, "i-zombie-7", "i-zombie", 1, 4),
    GameModeSpec(68, "i-zombie-8", "i-zombie", 1, 6),
    GameModeSpec(69, "i-zombie-9", "i-zombie", 1, 8),
    GameModeSpec(70, "i-zombie-endless", "i-zombie", 1, 9),
)
GAME_MODE_BY_ID = {spec.game_mode: spec for spec in SECONDARY_GAME_MODES}

_ZEN_MUSIC_MODES = {43, 50}
_FINAL_BOSS_MUSIC_MODES = {35}
_MINIGAME_MUSIC_MODES = {17, 25, 29, 30, 33, 48}
_CONVEYOR_MUSIC_MODES = {27}
_SILENT_MUSIC_MODES = {47}
_MUSIC_TUNE_BY_BACKGROUND = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 2,
    7: 1,
    8: 2,
    9: 1,
}


def expected_game_music_tune(spec: GameModeSpec) -> int:
    if spec.game_mode in _ZEN_MUSIC_MODES:
        return 8
    if spec.game_mode in _FINAL_BOSS_MUSIC_MODES:
        return 12
    if spec.game_mode in _MINIGAME_MUSIC_MODES:
        return 10
    if spec.game_mode in _CONVEYOR_MUSIC_MODES:
        return 11
    if spec.game_mode in _SILENT_MUSIC_MODES:
        return -1
    if spec.family in {"vasebreaker", "i-zombie"}:
        return 9
    return _MUSIC_TUNE_BY_BACKGROUND[spec.background]


def validate_game_mode_snapshot(
    spec: GameModeSpec,
    payload: Mapping[str, Any],
    *,
    checkpoint: int,
) -> list[str]:
    """Validate stable GOTY invariants for a secondary-mode bridge snapshot."""

    errors: list[str] = []

    def expect(field: str, expected: object) -> None:
        actual = payload.get(field)
        if actual != expected:
            errors.append(f"{field}: expected {expected!r}, got {actual!r}")

    expect("protocol_version", 4)
    expect("action_count", PortableActionCodec.size)
    expect("logical_width", 800)
    expect("logical_height", 600)
    expect("game_mode", spec.game_mode)
    expect("level", 0)
    expect("background", spec.background)
    expect("row_count", spec.row_count)
    expect("seed_bank_capacity", spec.seed_bank_capacity)
    expect("game_music_tune", expected_game_music_tune(spec))

    if checkpoint == 0:
        expect("scene", 3)
        expect("main_counter", spec.update_multiplier)
        expect("board_result", 0)
        expect("done", False)
        expect("won", False)
        expect("level_complete", False)
    else:
        if not payload.get("done", False):
            expect("scene", 3)
        main_counter = payload.get("main_counter")
        if not isinstance(main_counter, int):
            errors.append(f"main_counter: expected an integer, got {main_counter!r}")
        elif (
            not payload.get("done", False)
            and main_counter != (checkpoint + 1) * spec.update_multiplier
        ):
            errors.append(
                "main_counter: active mode expected "
                f"{(checkpoint + 1) * spec.update_multiplier}, "
                f"got {main_counter}"
            )

    legal = payload.get("legal_actions")
    if not isinstance(legal, list):
        errors.append("legal_actions: expected a list")
    else:
        if not legal or legal[0] != 0:
            errors.append("legal_actions: WAIT action 0 must be first")
        if len(legal) != len(set(legal)):
            errors.append("legal_actions: duplicate action ids")
        if any(
            not isinstance(action, int)
            or action < 0
            or action >= PortableActionCodec.size
            for action in legal
        ):
            errors.append("legal_actions: action id outside protocol vocabulary")

    interactions = payload.get("available_interactions")
    if not isinstance(interactions, list):
        errors.append("available_interactions: expected a list")
    else:
        if len(interactions) != len(set(interactions)):
            errors.append("available_interactions: duplicate interaction names")
        if any(not isinstance(interaction, str) for interaction in interactions):
            errors.append("available_interactions: names must be strings")
        expected_interactions = (
            ["last_stand_start"] if spec.game_mode == 31 else []
        )
        if interactions != expected_interactions:
            errors.append(
                "available_interactions: expected "
                f"{expected_interactions!r}, got {interactions!r}"
            )

    seed_packets = payload.get("seed_packets")
    if not isinstance(seed_packets, list):
        errors.append("seed_packets: expected a list")
    elif len(seed_packets) != spec.seed_bank_capacity:
        errors.append(
            f"seed_packets: expected {spec.seed_bank_capacity}, "
            f"got {len(seed_packets)}"
        )

    for count_field, collection_field in (
        ("plants_count", "plants"),
        ("zombies_count", "zombies"),
        ("coins_count", "coins"),
    ):
        collection = payload.get(collection_field)
        if not isinstance(collection, list):
            errors.append(f"{collection_field}: expected a list")
        elif payload.get(count_field) != len(collection):
            errors.append(
                f"{count_field}: expected {len(collection)}, "
                f"got {payload.get(count_field)!r}"
            )

    grid_items = payload.get("grid_items", [])
    if spec.family == "vasebreaker" and not any(
        item.get("type") == 7 for item in grid_items
    ):
        errors.append("grid_items: Vasebreaker must initialize scary pots")
    if spec.family == "i-zombie":
        brain_count = sum(item.get("type") == 12 for item in grid_items)
        if brain_count != 5:
            errors.append(f"grid_items: I, Zombie expected 5 brains, got {brain_count}")
    if spec.game_mode == 23 and payload.get("zombies_count", 0) == 0:
        errors.append("zombies: Zombiquarium must initialize aquarium zombies")
    if spec.game_mode == 35 and payload.get("zombies_count", 0) == 0:
        errors.append("zombies: Dr. Zomboss's Revenge must initialize Zomboss")

    return errors

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pvzwm.campaign import CampaignStage
from pvzwm.portable import PortableActionCodec

_BACKGROUND_BY_AREA = {
    "day": 0,
    "night": 1,
    "pool": 2,
    "fog": 3,
    "roof": 4,
    "night-roof": 5,
}
_CONVEYOR_MODES = {
    "wallnut-bowling",
    "little-trouble",
    "bungee-blitz",
    "conveyor",
    "boss",
}
_BOARD_CLICK_MODES = {"whack-a-zombie", "vasebreaker"}
_MUSIC_TUNE_BY_AREA = {
    "day": 1,
    "night": 2,
    "pool": 3,
    "fog": 4,
    "roof": 5,
}
_MINIGAME_MUSIC_MODES = {
    "wallnut-bowling",
    "whack-a-zombie",
    "little-trouble",
    "bungee-blitz",
}


def adventure_level_number(stage: CampaignStage) -> int:
    return (stage.world - 1) * 10 + stage.level


def expected_seed_bank_capacity(stage: CampaignStage) -> int:
    if stage.mode == "whack-a-zombie":
        return 3
    if stage.mode == "vasebreaker":
        return 1
    if stage.mode in _CONVEYOR_MODES:
        return 10

    seeds_available = (stage.world - 1) * 8 + stage.level
    if stage.level >= 10:
        seeds_available -= 2
    elif stage.level >= 5:
        seeds_available -= 1
    return min(6, seeds_available)


def expected_game_music_tune(stage: CampaignStage) -> int:
    if stage.mode == "boss":
        return 12
    if stage.mode in _MINIGAME_MUSIC_MODES:
        return 10
    if adventure_level_number(stage) in {10, 20, 30}:
        return 11
    if stage.stage_id == "4-10":
        return -1
    if stage.mode == "vasebreaker":
        return 9
    return _MUSIC_TUNE_BY_AREA[stage.area]


def validate_adventure_snapshot(
    stage: CampaignStage,
    payload: Mapping[str, Any],
    *,
    checkpoint: int,
) -> list[str]:
    """Validate GOTY Adventure invariants visible through the bridge."""

    errors: list[str] = []

    def expect(field: str, expected: object) -> None:
        actual = payload.get(field)
        if actual != expected:
            errors.append(f"{field}: expected {expected!r}, got {actual!r}")

    level = adventure_level_number(stage)
    background = _BACKGROUND_BY_AREA[stage.area]
    expect("protocol_version", 4)
    expect("action_count", PortableActionCodec.size)
    expect("logical_width", 800)
    expect("logical_height", 600)
    expect("game_mode", 0)
    expect("level", level)
    expect("area", stage.world)
    expect("stage", stage.level)
    expect("background", background)
    expect("row_count", stage.rows)
    expect("stage_is_night", background in {1, 3, 5})
    expect("stage_has_pool", background in {2, 3})
    expect("stage_has_fog", background == 3 and level != 40)
    expect("stage_has_roof", background in {4, 5})
    expect("game_music_tune", expected_game_music_tune(stage))
    expect("has_conveyor", stage.mode in _CONVEYOR_MODES)
    expect(
        "choose_seeds",
        level > 7
        and stage.mode not in _CONVEYOR_MODES
        and stage.mode not in _BOARD_CLICK_MODES,
    )
    expect("seed_bank_capacity", expected_seed_bank_capacity(stage))

    if checkpoint == 0:
        expect("main_counter", 1)
        expect("scene", 3)
        expect("board_result", 0)
        expect("done", False)
        expect("won", False)
        expect("level_complete", False)
    else:
        main_counter = payload.get("main_counter")
        if not isinstance(main_counter, int):
            errors.append(f"main_counter: expected an integer, got {main_counter!r}")
        elif main_counter > checkpoint + 1:
            errors.append(
                f"main_counter: advanced past checkpoint ({main_counter} > "
                f"{checkpoint + 1})"
            )
        elif not payload.get("done", False) and main_counter != checkpoint + 1:
            errors.append(
                f"main_counter: active stage expected {checkpoint + 1}, "
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
        board_clicks = [
            action
            for action in legal
            if PortableActionCodec.board_click_begin
            <= action
            < PortableActionCodec.collect_action
        ]
        expected_clicks = stage.rows * PortableActionCodec.cols
        if stage.mode in _BOARD_CLICK_MODES:
            if len(board_clicks) != expected_clicks:
                errors.append(
                    "legal_actions: expected "
                    f"{expected_clicks} board clicks, got {len(board_clicks)}"
                )
        elif board_clicks:
            errors.append(
                f"legal_actions: unexpected board clicks for {stage.mode} stage"
            )

    seed_packets = payload.get("seed_packets")
    if not isinstance(seed_packets, list):
        errors.append("seed_packets: expected a list")
    else:
        expected_capacity = expected_seed_bank_capacity(stage)
        if len(seed_packets) != expected_capacity:
            errors.append(
                "seed_packets: expected "
                f"{expected_capacity} packets, got {len(seed_packets)}"
            )
        slots = [packet.get("slot") for packet in seed_packets]
        if slots != list(range(len(seed_packets))):
            errors.append(f"seed_packets: non-canonical slots {slots!r}")

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

    if checkpoint == 0 and stage.mode == "whack-a-zombie":
        grave_count = sum(
            item.get("type") == 1 for item in payload.get("grid_items", [])
        )
        if grave_count == 0:
            errors.append("grid_items: Whack-a-Zombie must start with graves")
    if checkpoint == 0 and stage.mode == "vasebreaker":
        vase_count = sum(
            item.get("type") == 7 for item in payload.get("grid_items", [])
        )
        if vase_count == 0:
            errors.append("grid_items: Vasebreaker must start with scary pots")
    if checkpoint == 0 and stage.mode == "boss" and not payload.get("zombies"):
        errors.append("zombies: Zomboss stage must start with the boss")

    return errors

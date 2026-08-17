from __future__ import annotations

from copy import deepcopy

from pvzwm.goty_modes import (
    GAME_MODE_BY_ID,
    SECONDARY_GAME_MODES,
    expected_game_music_tune,
    validate_game_mode_snapshot,
)


def _snapshot(game_mode: int) -> dict[str, object]:
    spec = GAME_MODE_BY_ID[game_mode]
    grid_items: list[dict[str, int]] = []
    plants: list[dict[str, int]] = []
    zombies: list[dict[str, int]] = []
    if spec.family == "vasebreaker":
        grid_items.append({"type": 7})
    if spec.family == "i-zombie":
        grid_items.extend({"type": 12} for _ in range(5))
    if game_mode in {23, 35}:
        zombies.append({"type": 0})
    return {
        "protocol_version": 4,
        "action_count": 650,
        "logical_width": 800,
        "logical_height": 600,
        "game_mode": game_mode,
        "level": 0,
        "background": spec.background,
        "row_count": spec.row_count,
        "seed_bank_capacity": spec.seed_bank_capacity,
        "game_music_tune": expected_game_music_tune(spec),
        "scene": 3,
        "main_counter": spec.update_multiplier,
        "board_result": 0,
        "done": False,
        "won": False,
        "level_complete": False,
        "legal_actions": [0],
        "available_interactions": (
            ["last_stand_start"] if game_mode == 31 else []
        ),
        "seed_packets": [
            {"slot": slot} for slot in range(spec.seed_bank_capacity)
        ],
        "plants_count": len(plants),
        "plants": plants,
        "zombies_count": len(zombies),
        "zombies": zombies,
        "coins_count": 0,
        "coins": [],
        "grid_items": grid_items,
    }


def test_secondary_game_mode_manifest_is_complete_and_ordered() -> None:
    assert [spec.game_mode for spec in SECONDARY_GAME_MODES] == list(range(1, 71))
    assert len({spec.name for spec in SECONDARY_GAME_MODES}) == 70
    assert {spec.family for spec in SECONDARY_GAME_MODES} == {
        "survival",
        "mini-game",
        "limbo",
        "persistent",
        "vasebreaker",
        "i-zombie",
    }


def test_secondary_game_mode_invariants_cover_special_modes() -> None:
    for game_mode in range(1, 71):
        assert validate_game_mode_snapshot(
            GAME_MODE_BY_ID[game_mode],
            _snapshot(game_mode),
            checkpoint=0,
        ) == []


def test_secondary_music_tunes_cover_all_70_modes() -> None:
    expected_special = {
        17: 10,
        25: 10,
        27: 11,
        29: 10,
        30: 10,
        33: 10,
        35: 12,
        43: 8,
        47: -1,
        48: 10,
        50: 8,
    }
    for game_mode in range(51, 71):
        expected_special[game_mode] = 9

    assert len(SECONDARY_GAME_MODES) == 70
    for spec in SECONDARY_GAME_MODES:
        if spec.game_mode in expected_special:
            expected = expected_special[spec.game_mode]
        else:
            expected = {
                0: 1,
                1: 2,
                2: 3,
                3: 4,
                4: 5,
                8: 2,
            }[spec.background]
        assert expected_game_music_tune(spec) == expected


def test_secondary_game_mode_invariants_report_structural_differences() -> None:
    payload = deepcopy(_snapshot(51))
    payload["grid_items"] = []
    payload["seed_bank_capacity"] = 2

    errors = validate_game_mode_snapshot(
        GAME_MODE_BY_ID[51],
        payload,
        checkpoint=0,
    )

    assert "seed_bank_capacity: expected 1, got 2" in errors
    assert "grid_items: Vasebreaker must initialize scary pots" in errors

from __future__ import annotations

from pvzwm.campaign import campaign_stages
from pvzwm.goty_campaign import (
    adventure_level_number,
    expected_game_music_tune,
    expected_seed_bank_capacity,
)


def _stage(stage_id: str):
    return next(stage for stage in campaign_stages() if stage.stage_id == stage_id)


def test_adventure_stage_contracts_cover_special_modes() -> None:
    assert adventure_level_number(_stage("1-1")) == 1
    assert adventure_level_number(_stage("5-10")) == 50
    assert expected_seed_bank_capacity(_stage("1-1")) == 1
    assert expected_seed_bank_capacity(_stage("1-7")) == 6
    assert expected_seed_bank_capacity(_stage("1-5")) == 10
    assert expected_seed_bank_capacity(_stage("2-5")) == 3
    assert expected_seed_bank_capacity(_stage("4-5")) == 1
    assert expected_seed_bank_capacity(_stage("5-10")) == 10


def test_adventure_music_tunes_cover_every_area_and_set_piece() -> None:
    expected_by_stage = {
        "1-1": 1,
        "1-5": 10,
        "1-10": 11,
        "2-1": 2,
        "2-5": 10,
        "2-10": 11,
        "3-1": 3,
        "3-5": 10,
        "3-10": 11,
        "4-1": 4,
        "4-5": 9,
        "4-10": -1,
        "5-1": 5,
        "5-5": 10,
        "5-10": 12,
    }

    stages = campaign_stages()
    assert len(stages) == 50
    for stage in stages:
        if stage.stage_id in expected_by_stage:
            expected = expected_by_stage[stage.stage_id]
        else:
            expected = {
                "day": 1,
                "night": 2,
                "pool": 3,
                "fog": 4,
                "roof": 5,
            }[stage.area]
        assert expected_game_music_tune(stage) == expected

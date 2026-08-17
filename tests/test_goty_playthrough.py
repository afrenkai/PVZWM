from __future__ import annotations

from pvzwm.goty_playthrough import (
    NativeBoardDriver,
    SeedType,
    _shooter_counts,
    legal_plants,
    recommended_mode_seed_types,
    recommended_seed_types,
)
from pvzwm.portable import PortableActionCodec


def payload_with(*, seed_type: SeedType, legal: list[int], **extra: object) -> dict:
    return {
        "legal_actions": legal,
        "seed_packets": [{"slot": 0, "type": int(seed_type)}],
        "plants": [],
        "zombies": [],
        "grid_items": [],
        "row_count": 5,
        "stage_is_night": False,
        "stage_has_pool": False,
        "stage_has_roof": False,
        "has_conveyor": False,
        **extra,
    }


def test_legal_plants_joins_action_to_packet_type() -> None:
    codec = PortableActionCodec()
    action = codec.plant(0, 2, 3)
    plants = legal_plants(
        payload_with(seed_type=SeedType.REPEATER, legal=[0, action]),
        codec,
    )
    assert plants[0].action == action
    assert plants[0].seed_type == SeedType.REPEATER
    assert (plants[0].row, plants[0].col) == (2, 3)


def test_bobsled_mode_loadout_includes_ice_counters() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=28,
        seed_bank_capacity=6,
        available_seed_types=list(range(40)),
    )
    assert recommended_mode_seed_types(payload) == (
        SeedType.LILY_PAD,
        SeedType.SUNFLOWER,
        SeedType.JALAPENO,
        SeedType.SPIKEWEED,
        SeedType.SQUASH,
        SeedType.TALL_NUT,
    )


def test_air_raid_mode_loadout_includes_two_anti_air_plants() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=41,
        seed_bank_capacity=6,
        available_seed_types=list(range(40)),
    )
    assert recommended_mode_seed_types(payload) == (
        SeedType.LILY_PAD,
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.CACTUS,
        SeedType.BLOVER,
        SeedType.FUME_SHROOM,
    )


def test_secondary_mode_loadout_uses_the_bridge_challenge_unlock_set() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=16,
        seed_bank_capacity=6,
        available_seed_types=[int(SeedType.PEASHOOTER)],
    )
    assert recommended_mode_seed_types(payload) == (
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.REPEATER,
        SeedType.SNOW_PEA,
        SeedType.WALL_NUT,
        SeedType.CHERRY_BOMB,
    )


def test_survival_day_uses_an_affordable_initial_loadout() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=1,
        seed_bank_capacity=6,
        available_seed_types=[int(SeedType.PEASHOOTER)],
    )
    assert recommended_mode_seed_types(payload) == (
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.PEASHOOTER,
        SeedType.SNOW_PEA,
        SeedType.WALL_NUT,
        SeedType.CHERRY_BOMB,
    )


def test_survival_day_repick_uses_late_stage_reinforcement() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=1,
        seed_bank_capacity=6,
        available_seed_types=[int(SeedType.PEASHOOTER)],
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.MELON_PULT,
        SeedType.SNOW_PEA,
        SeedType.TALL_NUT,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
        SeedType.SQUASH,
    )


def test_survival_night_repick_adds_splash_damage_and_lane_control() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=2,
        seed_bank_capacity=7,
        available_seed_types=[
            int(SeedType.PEASHOOTER),
            int(SeedType.GLOOM_SHROOM),
        ],
        stage_is_night=True,
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.FUME_SHROOM,
        SeedType.GLOOM_SHROOM,
        SeedType.PUMPKIN,
        SeedType.MELON_PULT,
        SeedType.CHERRY_BOMB,
    )


def test_survival_night_final_repick_uses_immediate_defense() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=2,
        survival_stage=4,
        seed_bank_capacity=7,
        available_seed_types=[
            int(SeedType.PEASHOOTER),
            int(SeedType.GLOOM_SHROOM),
        ],
        stage_is_night=True,
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.PUFF_SHROOM,
        SeedType.FUME_SHROOM,
        SeedType.GLOOM_SHROOM,
        SeedType.PUMPKIN,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
        SeedType.ICE_SHROOM,
    )


def test_survival_night_fourth_stage_keeps_the_melon_reinforcement_deck() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=2,
        survival_stage=3,
        seed_bank_capacity=7,
        available_seed_types=[
            int(SeedType.PEASHOOTER),
            int(SeedType.GLOOM_SHROOM),
        ],
        stage_is_night=True,
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.FUME_SHROOM,
        SeedType.GLOOM_SHROOM,
        SeedType.PUMPKIN,
        SeedType.MELON_PULT,
        SeedType.CHERRY_BOMB,
    )


def test_survival_pool_uses_an_affordable_six_lane_opening() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=3,
        seed_bank_capacity=6,
        available_seed_types=[int(SeedType.PEASHOOTER)],
        stage_has_pool=True,
    )
    assert recommended_mode_seed_types(payload) == (
        SeedType.LILY_PAD,
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.PEASHOOTER,
        SeedType.SNOW_PEA,
        SeedType.WALL_NUT,
    )


def test_survival_pool_repick_adds_pool_and_flag_wave_counters() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=3,
        seed_bank_capacity=6,
        available_seed_types=[int(SeedType.PEASHOOTER)],
        stage_has_pool=True,
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.LILY_PAD,
        SeedType.SNOW_PEA,
        SeedType.TALL_NUT,
        SeedType.TANGLE_KELP,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
    )


def test_survival_fog_repick_builds_a_protected_cattail_core() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=4,
        seed_bank_capacity=7,
        available_seed_types=[
            int(SeedType.PEASHOOTER),
            int(SeedType.CATTAIL),
        ],
        stage_has_pool=True,
        stage_is_night=True,
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.LILY_PAD,
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.CATTAIL,
        SeedType.PUMPKIN,
        SeedType.WALL_NUT,
        SeedType.JALAPENO,
    )


def test_survival_fog_final_repick_adds_a_second_wave_clear() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=4,
        survival_stage=3,
        seed_bank_capacity=7,
        available_seed_types=[
            int(SeedType.PEASHOOTER),
            int(SeedType.CATTAIL),
        ],
        stage_has_pool=True,
        stage_is_night=True,
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.LILY_PAD,
        SeedType.PUFF_SHROOM,
        SeedType.CATTAIL,
        SeedType.PUMPKIN,
        SeedType.WALL_NUT,
        SeedType.JALAPENO,
        SeedType.CHERRY_BOMB,
    )


def test_survival_roof_repick_keeps_pots_and_adds_heavy_counters() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=5,
        seed_bank_capacity=6,
        available_seed_types=[int(SeedType.PEASHOOTER)],
        stage_has_roof=True,
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.FLOWER_POT,
        SeedType.KERNEL_PULT,
        SeedType.TALL_NUT,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
        SeedType.SQUASH,
    )


def test_survival_roof_completed_profile_preserves_the_proven_opening() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=5,
        seed_bank_capacity=10,
        available_seed_types=[int(SeedType.PEASHOOTER)],
        stage_has_roof=True,
    )
    assert recommended_mode_seed_types(payload) == (
        SeedType.FLOWER_POT,
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.MELON_PULT,
        SeedType.KERNEL_PULT,
        SeedType.CABBAGE_PULT,
        SeedType.UMBRELLA_LEAF,
        SeedType.GARLIC,
        SeedType.MAGNET_SHROOM,
        SeedType.MARIGOLD,
    )


def test_survival_roof_purchased_slots_use_defensive_reinforcements() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=5,
        seed_bank_capacity=10,
        available_seed_types=[int(SeedType.PEASHOOTER)],
        stage_has_roof=True,
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.FLOWER_POT,
        SeedType.KERNEL_PULT,
        SeedType.TALL_NUT,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
        SeedType.SQUASH,
        SeedType.SUNFLOWER,
        SeedType.MELON_PULT,
        SeedType.PUMPKIN,
        SeedType.UMBRELLA_LEAF,
    )


def test_survival_roof_seventh_packet_rebuilds_economy() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=5,
        seed_bank_capacity=7,
        available_seed_types=[int(SeedType.PEASHOOTER)],
        stage_has_roof=True,
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.FLOWER_POT,
        SeedType.KERNEL_PULT,
        SeedType.TALL_NUT,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
        SeedType.SQUASH,
        SeedType.SUNFLOWER,
    )


def test_survival_roof_final_repick_switches_to_melon_splash() -> None:
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        game_mode=5,
        survival_stage=3,
        seed_bank_capacity=6,
        available_seed_types=[int(SeedType.PEASHOOTER)],
        stage_has_roof=True,
    )
    assert recommended_mode_seed_types(payload, survival_repick=True) == (
        SeedType.FLOWER_POT,
        SeedType.MELON_PULT,
        SeedType.TALL_NUT,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
        SeedType.SQUASH,
    )


def test_driver_places_a_potato_mine_in_an_arming_column() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    near_spawn = codec.plant(0, 0, 7)
    arming_column = codec.plant(0, 0, 3)
    payload = payload_with(
        seed_type=SeedType.POTATO_MINE,
        legal=[0, near_spawn, arming_column],
        level=1,
        zombies=[
            {
                "type": 0,
                "row": 0,
                "x": 700.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == arming_column


def test_driver_uses_an_instant_on_an_imminent_breach_before_a_heavy_zombie() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    urgent = codec.plant(0, 1, 2)
    heavy = codec.plant(0, 3, 4)
    payload = payload_with(
        seed_type=SeedType.SQUASH,
        legal=[0, urgent, heavy],
        level=45,
        stage_has_roof=True,
        zombies=[
            {
                "type": 2,
                "row": 1,
                "x": 200.0,
                "mind_controlled": False,
            },
            {
                "type": 3,
                "row": 3,
                "x": 320.0,
                "mind_controlled": False,
            },
        ],
    )
    assert driver.choose_action(payload) == urgent


def test_later_survival_roof_orders_close_instant_targets_by_distance() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    closest = codec.plant(0, 1, 3)
    heavier = codec.plant(0, 3, 4)
    payload = payload_with(
        seed_type=SeedType.SQUASH,
        legal=[0, closest, heavier],
        game_mode=5,
        survival_stage=4,
        level=0,
        stage_has_roof=True,
        zombies=[
            {
                "type": 0,
                "row": 1,
                "x": 300.0,
                "mind_controlled": False,
            },
            {
                "type": 7,
                "row": 3,
                "x": 420.0,
                "mind_controlled": False,
            },
        ],
    )
    assert driver.choose_action(payload) == closest


def test_later_survival_roof_prioritizes_a_lane_without_a_cleaner() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    unprotected = codec.plant(0, 1, 3)
    protected = codec.plant(0, 3, 3)
    payload = payload_with(
        seed_type=SeedType.SQUASH,
        legal=[0, unprotected, protected],
        game_mode=5,
        survival_stage=4,
        level=0,
        stage_has_roof=True,
        mowers=[{"row": 3, "state": 1}],
        zombies=[
            {
                "type": 0,
                "row": 1,
                "x": 350.0,
                "mind_controlled": False,
            },
            {
                "type": 0,
                "row": 3,
                "x": 300.0,
                "mind_controlled": False,
            },
        ],
    )
    assert driver.choose_action(payload) == unprotected


def test_later_survival_roof_prepares_support_in_the_closest_emergency_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    emergency_pot = codec.plant(0, 2, 3)
    distant_squash = codec.plant(1, 4, 3)
    payload = payload_with(
        seed_type=SeedType.FLOWER_POT,
        legal=[0, emergency_pot, distant_squash],
        game_mode=5,
        survival_stage=4,
        level=0,
        stage_has_roof=True,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.FLOWER_POT)},
            {"slot": 1, "type": int(SeedType.SQUASH)},
        ],
        zombies=[
            {
                "type": 0,
                "row": 2,
                "x": 300.0,
                "mind_controlled": False,
            },
            {
                "type": 7,
                "row": 4,
                "x": 600.0,
                "mind_controlled": False,
            },
        ],
    )
    assert driver.choose_action(payload) == emergency_pot


def test_survival_roof_builds_tall_nut_defense_at_level_zero() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    tall_nut = codec.plant(0, 1, 6)
    payload = payload_with(
        seed_type=SeedType.TALL_NUT,
        legal=[0, tall_nut],
        game_mode=5,
        level=0,
        stage_has_roof=True,
        plants=[
            {"type": int(SeedType.KERNEL_PULT), "row": row, "col": 2}
            for row in range(5)
        ],
        zombies=[
            {
                "type": 2,
                "row": 1,
                "x": 550.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == tall_nut


def test_later_survival_stage_uses_an_affordable_selected_pult() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    kernel_pult = codec.plant(0, 1, 2)
    payload = payload_with(
        seed_type=SeedType.KERNEL_PULT,
        legal=[0, kernel_pult],
        game_mode=5,
        survival_stage=1,
        level=0,
        stage_has_roof=True,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.KERNEL_PULT)},
            {"slot": 1, "type": int(SeedType.MELON_PULT)},
        ],
        plants=[
            {"type": int(SeedType.KERNEL_PULT), "row": row, "col": 2}
            for row in (0, 2, 3, 4)
        ],
        zombies=[
            {
                "type": 0,
                "row": 1,
                "x": 600.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == kernel_pult


def test_later_survival_roof_reserves_instants_for_close_threats() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    squash = codec.plant(0, 1, 4)
    payload = payload_with(
        seed_type=SeedType.SQUASH,
        legal=[0, squash],
        game_mode=5,
        survival_stage=1,
        level=0,
        stage_has_roof=True,
        zombies=[
            {
                "type": 0,
                "row": 1,
                "x": 550.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == 0


def test_final_survival_roof_spends_an_instant_before_a_lane_collapses() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    squash = codec.plant(0, 1, 4)
    payload = payload_with(
        seed_type=SeedType.SQUASH,
        legal=[0, squash],
        game_mode=5,
        survival_stage=4,
        level=0,
        stage_has_roof=True,
        zombies=[
            {
                "type": 2,
                "row": 1,
                "x": 550.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == squash


def test_art_challenge_driver_fills_an_outlined_target() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    target = codec.plant(0, 0, 4)
    ordinary = codec.plant(0, 0, 3)
    payload = payload_with(
        seed_type=SeedType.WALL_NUT,
        legal=[0, ordinary, target],
        game_mode=36,
        plants=[
            {"type": int(SeedType.SUNFLOWER), "row": 0, "col": 0},
            {"type": int(SeedType.SUNFLOWER), "row": 1, "col": 0},
            {"type": int(SeedType.SUNFLOWER), "row": 2, "col": 0},
            {"type": int(SeedType.REPEATER), "row": 0, "col": 1},
            {"type": int(SeedType.REPEATER), "row": 1, "col": 1},
            {"type": int(SeedType.REPEATER), "row": 2, "col": 1},
            {"type": int(SeedType.REPEATER), "row": 3, "col": 1},
            {"type": int(SeedType.REPEATER), "row": 4, "col": 1},
        ],
    )
    assert driver.choose_action(payload) == target


def test_art_challenge_driver_builds_economy_before_the_picture() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    wall_nut = codec.plant(0, 0, 4)
    sunflower = codec.plant(1, 0, 0)
    payload = payload_with(
        seed_type=SeedType.WALL_NUT,
        legal=[0, wall_nut, sunflower],
        game_mode=36,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.WALL_NUT)},
            {"slot": 1, "type": int(SeedType.SUNFLOWER)},
        ],
    )
    assert driver.choose_action(payload) == sunflower


def test_last_stand_driver_starts_only_after_its_setup_policy_is_exhausted() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    plant = codec.plant(0, 0, 2)
    setup_payload = payload_with(
        seed_type=SeedType.REPEATER,
        legal=[0, plant],
        available_interactions=["last_stand_start"],
    )
    assert driver.choose_interaction(setup_payload) is None

    ready_payload = payload_with(
        seed_type=SeedType.REPEATER,
        legal=[0],
        available_interactions=["last_stand_start"],
    )
    assert driver.choose_interaction(ready_payload) == "last_stand_start"


def test_survival_driver_selects_repick_interaction_immediately() -> None:
    driver = NativeBoardDriver()
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0],
        available_interactions=["survival_repick"],
    )
    assert driver.choose_interaction(payload) == "survival_repick"


def test_driver_collects_before_planting() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, codec.plant(0, 0, 2), codec.collect_action],
    )
    assert driver.choose_action(payload) == codec.collect_action


def test_driver_builds_day_economy_before_shooters() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    sunflower = codec.plant(0, 0, 0)
    peashooter = codec.plant(1, 0, 2)
    payload = payload_with(
        seed_type=SeedType.SUNFLOWER,
        legal=[0, sunflower, peashooter],
        seed_packets=[
            {"slot": 0, "type": int(SeedType.SUNFLOWER)},
            {"slot": 1, "type": int(SeedType.PEASHOOTER)},
        ],
    )
    assert driver.choose_action(payload) == sunflower


def test_driver_builds_two_sun_shrooms_per_survival_night_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    sun_shroom = codec.plant(0, 0, 1)
    fume_shroom = codec.plant(1, 0, 2)
    payload = payload_with(
        seed_type=SeedType.SUN_SHROOM,
        legal=[0, sun_shroom, fume_shroom],
        game_mode=2,
        stage_is_night=True,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.SUN_SHROOM)},
            {"slot": 1, "type": int(SeedType.FUME_SHROOM)},
        ],
        plants=[
            {"type": int(SeedType.SUN_SHROOM), "row": row, "col": 0} for row in range(5)
        ],
    )
    assert driver.choose_action(payload) == sun_shroom


def test_driver_reserves_final_survival_night_sun_for_a_lane_clear() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    fume_shroom = codec.plant(0, 0, 3)
    payload = payload_with(
        seed_type=SeedType.FUME_SHROOM,
        legal=[0, fume_shroom],
        game_mode=2,
        survival_stage=4,
        stage_is_night=True,
        sun=75,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.FUME_SHROOM), "refresh": 0},
            {"slot": 1, "type": int(SeedType.JALAPENO), "refresh": 0},
        ],
        plants=[{"type": int(SeedType.PUFF_SHROOM), "row": 0, "col": 2}],
        zombies=[{"type": 0, "row": 0, "x": 400.0}],
    )
    assert driver.choose_action(payload) == 0


def test_driver_upgrades_and_protects_a_survival_night_gloom_core() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    gloom = codec.plant(0, 2, 4)
    pumpkin = codec.plant(1, 2, 4)
    base_plants = [
        {"type": int(SeedType.SUN_SHROOM), "row": row, "col": col}
        for row in range(5)
        for col in (0, 1)
    ]
    base_plants.append({"type": int(SeedType.FUME_SHROOM), "row": 2, "col": 4})
    payload = payload_with(
        seed_type=SeedType.GLOOM_SHROOM,
        legal=[0, gloom],
        game_mode=2,
        stage_is_night=True,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.GLOOM_SHROOM), "refresh": 0},
            {"slot": 1, "type": int(SeedType.PUMPKIN), "refresh": 0},
        ],
        plants=base_plants,
    )
    assert driver.choose_action(payload) == gloom

    payload["legal_actions"] = [0, pumpkin]
    payload["plants"] = [
        *base_plants,
        {"type": int(SeedType.GLOOM_SHROOM), "row": 2, "col": 4},
    ]
    assert driver.choose_action(payload) == pumpkin


def test_driver_clicks_a_vase_at_its_native_cell() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[{"type": 7, "row": 3, "col": 6}],
    )
    assert driver.choose_action(payload) == codec.board_click(3, 6)


def test_driver_deploys_an_affordable_i_zombie_in_the_weakest_brain_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    normal = codec.plant(0, 2, 4)
    bucket = codec.plant(1, 2, 4)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_NORMAL,
        legal=[0, normal, bucket],
        game_mode=61,
        sun=150,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_NORMAL)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_BUCKET)},
            {"slot": 2, "type": int(SeedType.ZOMBIE_FOOTBALL)},
        ],
        plants=[
            {"type": int(SeedType.SNOW_PEA), "row": 0, "col": 3},
            {"type": int(SeedType.PEASHOOTER), "row": 1, "col": 3},
            {"type": int(SeedType.SUNFLOWER), "row": 2, "col": 2},
            {"type": int(SeedType.PEASHOOTER), "row": 2, "col": 3},
        ],
        grid_items=[{"type": 12, "row": row, "col": 0} for row in range(3)],
    )
    assert driver.choose_action(payload) == bucket


def test_driver_uses_a_bucket_instead_of_overpaying_for_i_zombie_football() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    bucket = codec.plant(0, 2, 4)
    football = codec.plant(1, 2, 4)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_BUCKET,
        legal=[0, bucket, football],
        game_mode=61,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_BUCKET)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_FOOTBALL)},
        ],
        plants=[
            {"type": int(SeedType.SUNFLOWER), "row": 2, "col": 2},
            {"type": int(SeedType.PEASHOOTER), "row": 2, "col": 3},
        ],
        grid_items=[{"type": 12, "row": 2, "col": 0}],
    )
    assert driver.choose_action(payload) == bucket


def test_driver_uses_i_zombie_football_on_level_nines_bottom_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    bucket = codec.plant(0, 4, 6)
    football = codec.plant(1, 4, 6)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_BUCKET,
        legal=[0, bucket, football],
        game_mode=69,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_BUCKET)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_FOOTBALL)},
        ],
        plants=[
            {"type": int(SeedType.STARFRUIT), "row": 4, "col": 4},
            {"type": int(SeedType.SUNFLOWER), "row": 4, "col": 2},
        ],
        grid_items=[{"type": 12, "row": 4, "col": 0}],
    )
    assert driver.choose_action(payload) == football


def test_driver_uses_one_digger_against_rear_i_zombie_defenders() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    direct = codec.plant(0, 3, 6)
    digger = codec.plant(1, 3, 6)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_IMP,
        legal=[0, direct, digger],
        game_mode=68,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_IMP)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_DIGGER)},
        ],
        plants=[
            {"type": int(SeedType.PEASHOOTER), "row": 3, "col": 0},
            {"type": int(SeedType.PEASHOOTER), "row": 3, "col": 1},
        ],
        grid_items=[{"type": 12, "row": 3, "col": 0}],
    )
    assert driver.choose_action(payload) == digger

    payload["zombies"] = [{"type": 5, "row": 3, "x": 240.0}]
    payload["plants"] = payload["plants"][:-1]
    assert driver.choose_action(payload) == 0


def test_driver_opens_i_zombie_nine_by_bungeeing_a_top_sunflower() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    bungee = codec.plant(0, 0, 5)
    digger = codec.plant(1, 1, 6)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_BUNGEE,
        legal=[0, bungee, digger],
        game_mode=69,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_BUNGEE)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_DIGGER)},
        ],
        plants=[
            {"type": int(SeedType.SUNFLOWER), "row": 0, "col": 5},
            {"type": int(SeedType.THREEPEATER), "row": 1, "col": 2},
        ],
        grid_items=[
            {"type": 12, "row": 0, "col": 0},
            {"type": 12, "row": 1, "col": 0},
        ],
    )
    assert driver.choose_action(payload) == bungee


def test_driver_uses_an_imp_to_eat_an_exposed_i_zombie_brain() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    imp = codec.plant(0, 3, 6)
    pole_vaulter = codec.plant(1, 3, 6)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_IMP,
        legal=[0, imp, pole_vaulter],
        game_mode=69,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_IMP)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_POLE_VAULTER)},
        ],
        grid_items=[{"type": 12, "row": 3, "col": 0}],
    )
    assert driver.choose_action(payload) == imp


def test_driver_uses_a_cheap_direct_i_zombie_on_an_unarmed_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    normal = codec.plant(0, 3, 4)
    bucket = codec.plant(1, 3, 4)
    digger = codec.plant(2, 3, 4)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_NORMAL,
        legal=[0, normal, bucket, digger],
        game_mode=63,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_NORMAL)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_BUCKET)},
            {"slot": 2, "type": int(SeedType.ZOMBIE_DIGGER)},
        ],
        plants=[
            {"type": int(SeedType.SUNFLOWER), "row": 3, "col": 0},
            {"type": int(SeedType.TORCHWOOD), "row": 3, "col": 2},
        ],
        grid_items=[{"type": 12, "row": 3, "col": 0}],
    )
    assert driver.choose_action(payload) == normal


def test_driver_reserves_an_i_zombie_balloon_until_after_sun_harvesting() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    bucket = codec.plant(0, 2, 4)
    balloon = codec.plant(1, 2, 4)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_BUCKET,
        legal=[0, bucket, balloon],
        game_mode=65,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_BUCKET)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_BALLOON)},
        ],
        plants=[
            {"type": int(SeedType.SUNFLOWER), "row": 2, "col": 2},
            {"type": int(SeedType.PEASHOOTER), "row": 2, "col": 3},
        ],
        grid_items=[{"type": 12, "row": 2, "col": 0}],
    )
    assert driver.choose_action(payload) == bucket


def test_driver_waits_when_two_i_zombies_are_already_attacking_a_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    action = codec.plant(0, 0, 4)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_NORMAL,
        legal=[0, action],
        game_mode=61,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_NORMAL)},
        ],
        zombies=[
            {"type": 0, "row": 0, "x": 400.0},
            {"type": 0, "row": 0, "x": 500.0},
        ],
        grid_items=[{"type": 12, "row": 0, "col": 0}],
    )
    assert driver.choose_action(payload) == 0


def test_driver_reinforces_a_threatened_i_zombie_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    action = codec.plant(0, 0, 4)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_BUCKET,
        legal=[0, action],
        game_mode=65,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_BUCKET)},
        ],
        plants=[{"type": int(SeedType.PEASHOOTER), "row": 0, "col": 2}],
        zombies=[{"type": 4, "row": 0, "x": 350.0}],
        grid_items=[{"type": 12, "row": 0, "col": 0}],
    )
    assert driver.choose_action(payload) == action


def test_driver_uses_a_pole_vaulter_to_clear_i_zombie_garlic() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    pole_vaulter = codec.plant(0, 3, 5)
    bucket = codec.plant(1, 3, 5)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_POLE_VAULTER,
        legal=[0, pole_vaulter, bucket],
        game_mode=66,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_POLE_VAULTER)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_BUCKET)},
        ],
        plants=[
            {"type": int(SeedType.GARLIC), "row": 3, "col": 4},
            {"type": int(SeedType.SUNFLOWER), "row": 3, "col": 2},
            {"type": int(SeedType.PEASHOOTER), "row": 3, "col": 1},
        ],
        grid_items=[{"type": 12, "row": 3, "col": 0}],
    )
    assert driver.choose_action(payload) == pole_vaulter


def test_driver_uses_a_pole_vaulter_against_a_front_i_zombie_obstacle() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    pole_vaulter = codec.plant(0, 3, 6)
    bucket = codec.plant(1, 3, 6)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_POLE_VAULTER,
        legal=[0, pole_vaulter, bucket],
        game_mode=68,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_POLE_VAULTER)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_BUCKET)},
        ],
        plants=[
            {"type": int(SeedType.TORCHWOOD), "row": 3, "col": 5},
            {"type": int(SeedType.PEASHOOTER), "row": 3, "col": 4},
            {"type": int(SeedType.SUNFLOWER), "row": 3, "col": 3},
        ],
        grid_items=[{"type": 12, "row": 3, "col": 0}],
    )
    assert driver.choose_action(payload) == pole_vaulter


def test_driver_uses_a_normal_i_zombie_to_trigger_a_potato_mine() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    normal = codec.plant(0, 2, 5)
    bucket = codec.plant(1, 2, 5)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_NORMAL,
        legal=[0, normal, bucket],
        game_mode=67,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_NORMAL)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_BUCKET)},
        ],
        plants=[
            {"type": int(SeedType.POTATO_MINE), "row": 2, "col": 3},
            {"type": int(SeedType.SUNFLOWER), "row": 2, "col": 2},
        ],
        grid_items=[{"type": 12, "row": 2, "col": 0}],
    )
    assert driver.choose_action(payload) == normal


def test_driver_does_not_double_commit_normals_to_one_i_zombie_mine() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    normal = codec.plant(0, 2, 5)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_NORMAL,
        legal=[0, normal],
        game_mode=67,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_NORMAL)},
        ],
        plants=[
            {"type": int(SeedType.POTATO_MINE), "row": 2, "col": 3},
        ],
        zombies=[{"type": 0, "row": 2, "x": 350.0}],
        grid_items=[{"type": 12, "row": 2, "col": 0}],
    )
    assert driver.choose_action(payload) == 0


def test_driver_uses_an_affordable_i_zombie_dancer_against_chomper_rows() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    normal = codec.plant(0, 2, 5)
    dancer = codec.plant(1, 2, 5)
    payload = payload_with(
        seed_type=SeedType.ZOMBIE_NORMAL,
        legal=[0, normal, dancer],
        game_mode=67,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.ZOMBIE_NORMAL)},
            {"slot": 1, "type": int(SeedType.ZOMBIE_DANCER)},
        ],
        plants=[
            {"type": int(SeedType.POTATO_MINE), "row": 2, "col": 3},
            {"type": int(SeedType.CHOMPER), "row": 1, "col": 2},
            {"type": int(SeedType.CHOMPER), "row": 2, "col": 1},
        ],
        grid_items=[{"type": 12, "row": 2, "col": 0}],
    )
    assert driver.choose_action(payload) == dancer


def test_driver_places_a_collected_vase_seed_before_opening_another_vase() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    first_vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[{"type": 7, "row": 3, "col": 6, "seed_type": 0}],
    )
    assert driver.choose_action(first_vase) == codec.board_click(3, 6)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        grid_items=[{"type": 7, "row": 2, "col": 6, "seed_type": 17}],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(3, 4)


def test_driver_does_not_place_a_vase_seed_on_an_occupied_cell() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[{"type": 0, "row": 0, "col": 4}],
        grid_items=[{"type": 7, "row": 0, "col": 3, "seed_type": 17}],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(0, 2)


def test_driver_places_a_recovered_wall_nut_near_the_house() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 2,
                "col": 3,
                "seed_type": int(SeedType.WALL_NUT),
            },
            {
                "type": 7,
                "row": 4,
                "col": 8,
                "seed_type": int(SeedType.SPIKEWEED),
            },
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(2, 3)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        grid_items=[
            {
                "type": 7,
                "row": 4,
                "col": 8,
                "seed_type": int(SeedType.SPIKEWEED),
            }
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(2, 0)


def test_driver_places_a_recovered_tall_nut_in_front_of_shooters() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 2,
                "col": 8,
                "seed_type": int(SeedType.TALL_NUT),
            },
            {"type": 7, "row": 2, "col": 5, "seed_type": -1},
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(2, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        grid_items=[
            {"type": 7, "row": 2, "col": 5, "seed_type": -1},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(2, 4)


def test_driver_spreads_recovered_peashooters_across_vasebreaker_lanes() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[{"type": 7, "row": 1, "col": 8, "seed_type": 0}],
    )
    assert driver.choose_action(vase) == codec.board_click(1, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[{"type": 0, "row": 1, "col": 4}],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(0, 4)


def test_driver_spreads_other_recovered_shooters_across_vasebreaker_lanes() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 3,
                "col": 8,
                "seed_type": int(SeedType.PUFF_SHROOM),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(3, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {"type": int(SeedType.PUFF_SHROOM), "row": 3, "col": 4},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(0, 4)


def test_driver_places_a_recovered_threepeater_for_three_lane_coverage() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 4,
                "col": 8,
                "seed_type": int(SeedType.THREEPEATER),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(4, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {"type": int(SeedType.TALL_NUT), "row": 1, "col": 0},
            {"type": int(SeedType.SQUASH), "row": 1, "col": 1},
        ],
        grid_items=[
            *(
                {"type": 7, "row": 0, "col": col, "seed_type": -1}
                for col in (2, 3, 5, 7, 8)
            ),
            *({"type": 7, "row": 1, "col": col, "seed_type": -1} for col in (2, 5, 6)),
            *(
                {"type": 7, "row": 2, "col": col, "seed_type": -1}
                for col in (2, 3, 5, 6)
            ),
            {"type": 7, "row": 3, "col": 8, "seed_type": -1},
            {"type": 7, "row": 4, "col": 8, "seed_type": -1},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(0, 1)


def test_driver_spreads_recovered_spikeweeds_across_vasebreaker_lanes() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 4,
                "col": 8,
                "seed_type": int(SeedType.SPIKEWEED),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(4, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {"type": int(SeedType.SPIKEWEED), "row": 4, "col": 4},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(0, 4)


def test_driver_places_a_recovered_peashooter_in_a_lane_without_squash() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 8,
                "seed_type": int(SeedType.PEASHOOTER),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(0, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[{"type": int(SeedType.SQUASH), "row": 0, "col": 5}],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(1, 4)


def test_driver_balances_recovered_snow_peas_across_vasebreaker_lanes() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 1,
                "col": 8,
                "seed_type": int(SeedType.SNOW_PEA),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(1, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {"type": int(SeedType.PEASHOOTER), "row": 0, "col": 4},
            {"type": int(SeedType.PEASHOOTER), "row": 1, "col": 4},
            {"type": int(SeedType.SQUASH), "row": 1, "col": 5},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(2, 4)


def test_driver_weights_recovered_shooters_by_unopened_zombie_vases() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 8,
                "seed_type": int(SeedType.SNOW_PEA),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(0, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        grid_items=[
            {"type": 7, "row": 1, "col": 6, "seed_type": -1},
            {"type": 7, "row": 1, "col": 7, "seed_type": -1},
            {"type": 7, "row": 2, "col": 6, "seed_type": -1},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(1, 4)


def test_driver_gives_each_threatened_vasebreaker_lane_two_defenders() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 8,
                "seed_type": int(SeedType.SNOW_PEA),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(0, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {"type": int(SeedType.PEASHOOTER), "row": 1, "col": 4},
            {"type": int(SeedType.PEASHOOTER), "row": 3, "col": 4},
            {"type": int(SeedType.SNOW_PEA), "row": 3, "col": 3},
        ],
        grid_items=[
            {"type": 7, "row": 1, "col": 6, "seed_type": -1},
            {"type": 7, "row": 3, "col": 5, "seed_type": -1},
            {"type": 7, "row": 3, "col": 6, "seed_type": -1},
            {"type": 7, "row": 3, "col": 7, "seed_type": -1},
            {"type": 7, "row": 3, "col": 8, "seed_type": -1},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(1, 3)


def test_driver_balances_recovered_leftpeaters_on_the_right_side() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 1,
                "col": 8,
                "seed_type": int(SeedType.LEFTPEATER),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(1, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {"type": int(SeedType.LEFTPEATER), "row": 0, "col": 7},
            {"type": int(SeedType.SNOW_PEA), "row": 1, "col": 4},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(2, 8)


def test_driver_moves_a_leftpeater_out_of_a_blocked_source_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 1,
                "col": 6,
                "seed_type": int(SeedType.LEFTPEATER),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(1, 6)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        grid_items=[{"type": 7, "row": 1, "col": col} for col in (2, 3, 4, 5, 6, 7, 8)]
        + [{"type": 7, "row": 0, "col": 7}],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(0, 8)


def test_driver_does_not_count_a_leftpeater_left_of_a_zombie_vase() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 1,
                "col": 8,
                "seed_type": int(SeedType.LEFTPEATER),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(1, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {"type": int(SeedType.LEFTPEATER), "row": 0, "col": 6},
            {"type": int(SeedType.SNOW_PEA), "row": 1, "col": 4},
        ],
        grid_items=[
            {"type": 7, "row": 0, "col": 7, "seed_type": -1},
            {"type": 7, "row": 1, "col": 6, "seed_type": -1},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(0, 8)


def test_driver_spreads_recovered_squashes_across_vasebreaker_lanes() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 8,
                "seed_type": int(SeedType.SQUASH),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(0, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {
                "type": int(SeedType.SQUASH),
                "row": 0,
                "col": 5,
            }
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(1, 5)


def test_driver_places_a_recovered_squash_in_the_weakest_mixed_shooter_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 8,
                "seed_type": int(SeedType.SQUASH),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(0, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {"type": int(SeedType.SNOW_PEA), "row": 0, "col": 2},
            {"type": int(SeedType.LEFTPEATER), "row": 1, "col": 7},
            {"type": int(SeedType.SNOW_PEA), "row": 2, "col": 2},
            {"type": int(SeedType.LEFTPEATER), "row": 3, "col": 7},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(4, 5)


def test_driver_spreads_squashes_before_weighting_vasebreaker_lanes() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 8,
                "seed_type": int(SeedType.SQUASH),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(0, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {"type": int(SeedType.SQUASH), "row": 0, "col": 5},
            {"type": int(SeedType.PEASHOOTER), "row": 1, "col": 4},
        ],
        grid_items=[
            {"type": 7, "row": 0, "col": 6, "seed_type": -1},
            {"type": 7, "row": 0, "col": 7, "seed_type": -1},
            {"type": 7, "row": 0, "col": 8, "seed_type": -1},
            {"type": 7, "row": 1, "col": 6, "seed_type": -1},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(1, 5)


def test_driver_reserves_a_recovered_squash_for_a_dancer_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 8,
                "seed_type": int(SeedType.SQUASH),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(0, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        plants=[
            {"type": int(SeedType.SNOW_PEA), "row": 2, "col": 2},
        ],
        grid_items=[
            {
                "type": 7,
                "row": 1,
                "col": 7,
                "seed_type": -1,
                "zombie_type": 0,
            },
            {
                "type": 7,
                "row": 2,
                "col": 4,
                "seed_type": -1,
                "zombie_type": 8,
            },
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(2, 5)


def test_driver_moves_a_recovered_blocker_out_of_an_empty_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 8,
                "seed_type": int(SeedType.WALL_NUT),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(0, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        grid_items=[
            {"type": 7, "row": 2, "col": 7, "seed_type": -1},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(2, 4)


def test_driver_reserves_a_lane_with_an_unopened_matching_defense_vase() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    vase = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 8,
                "seed_type": int(SeedType.HYPNO_SHROOM),
            }
        ],
    )
    assert driver.choose_action(vase) == codec.board_click(0, 8)

    carrying_seed = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        coins=[{"type": 16, "being_collected": True}],
        grid_items=[
            {
                "type": 7,
                "row": 2,
                "col": 3,
                "seed_type": int(SeedType.HYPNO_SHROOM),
            },
            {"type": 7, "row": 2, "col": 7, "seed_type": -1},
            {"type": 7, "row": 3, "col": 7, "seed_type": -1},
        ],
    )
    assert driver.choose_action(carrying_seed) == codec.board_click(3, 4)


def test_driver_opens_seed_vases_before_zombie_vases() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {"type": 7, "row": 0, "col": 6, "seed_type": 0},
            {"type": 7, "row": 4, "col": 8, "seed_type": -1},
        ],
    )
    assert driver.choose_action(payload) == codec.board_click(0, 6)


def test_driver_opens_blocker_and_squash_vases_before_offense_vases() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 1,
                "col": 8,
                "seed_type": int(SeedType.SPIKEWEED),
            },
            {
                "type": 7,
                "row": 2,
                "col": 6,
                "seed_type": int(SeedType.SQUASH),
            },
            {
                "type": 7,
                "row": 2,
                "col": 3,
                "seed_type": int(SeedType.WALL_NUT),
            },
        ],
    )
    assert driver.choose_action(payload) == codec.board_click(2, 3)


def test_driver_opens_the_closest_zombie_vase_first() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 8,
                "seed_type": -1,
                "zombie_type": 0,
            },
            {
                "type": 7,
                "row": 3,
                "col": 6,
                "seed_type": -1,
                "zombie_type": 0,
            },
        ],
    )
    assert driver.choose_action(payload) == codec.board_click(3, 6)


def test_driver_opens_a_heavy_zombie_vase_before_a_closer_normal_vase() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 4,
                "seed_type": -1,
                "zombie_type": 0,
            },
            {
                "type": 7,
                "row": 3,
                "col": 8,
                "seed_type": -1,
                "zombie_type": 4,
            },
        ],
    )
    assert driver.choose_action(payload) == codec.board_click(3, 8)


def test_driver_opens_a_jack_in_the_box_vase_last() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 3,
                "seed_type": -1,
                "zombie_type": 15,
            },
            {
                "type": 7,
                "row": 1,
                "col": 7,
                "seed_type": -1,
                "zombie_type": 0,
            },
        ],
    )
    assert driver.choose_action(payload) == codec.board_click(1, 7)


def test_driver_opens_a_bucket_vase_before_a_dancer_vase() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 3,
                "seed_type": -1,
                "zombie_type": 8,
            },
            {
                "type": 7,
                "row": 1,
                "col": 7,
                "seed_type": -1,
                "zombie_type": 4,
            },
        ],
    )
    assert driver.choose_action(payload) == codec.board_click(1, 7)


def test_driver_opens_a_dancer_vase_before_a_normal_vase() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[
            {
                "type": 7,
                "row": 0,
                "col": 3,
                "seed_type": -1,
                "zombie_type": 8,
            },
            {
                "type": 7,
                "row": 1,
                "col": 7,
                "seed_type": -1,
                "zombie_type": 0,
            },
        ],
    )
    assert driver.choose_action(payload) == codec.board_click(0, 3)


def test_driver_waits_for_a_revealed_vase_zombie_to_clear() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        grid_items=[{"type": 7, "row": 4, "col": 8, "seed_type": -1}],
        zombies=[{"row": 2, "x": 500.0, "mind_controlled": False}],
    )
    assert driver.choose_action(payload) == 0


def test_driver_waits_after_the_last_vase_instead_of_clicking_a_zombie() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    clicks = [
        codec.board_click(row, col) for row in range(5) for col in range(codec.cols)
    ]
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, *clicks],
        level=35,
        zombies=[{"row": 2, "x": 500.0, "mind_controlled": False}],
    )
    assert driver.choose_action(payload) == 0


def test_driver_uses_an_instant_for_a_close_lane_threat() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    squash = codec.plant(0, 2, 4)
    payload = payload_with(
        seed_type=SeedType.SQUASH,
        legal=[0, squash],
        zombies=[{"row": 2, "x": 350.0, "mind_controlled": False}],
    )
    assert driver.choose_action(payload) == squash


def test_driver_prioritizes_a_pogo_zombie_for_instant_defense() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    ordinary_lane = codec.plant(0, 0, 2)
    pogo_lane = codec.plant(0, 4, 4)
    payload = payload_with(
        seed_type=SeedType.SQUASH,
        legal=[0, ordinary_lane, pogo_lane],
        level=44,
        stage_has_roof=True,
        zombies=[
            {
                "type": 0,
                "row": 0,
                "x": 250.0,
                "mind_controlled": False,
            },
            {
                "type": 18,
                "row": 4,
                "x": 500.0,
                "mind_controlled": False,
            },
        ],
    )
    assert driver.choose_action(payload) == pogo_lane


def test_driver_uses_jalapeno_against_a_single_football_zombie() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    jalapeno = codec.plant(0, 0, 4)
    payload = payload_with(
        seed_type=SeedType.JALAPENO,
        legal=[0, jalapeno],
        game_mode=4,
        survival_stage=3,
        stage_is_night=True,
        stage_has_pool=True,
        zombies=[
            {
                "type": 7,
                "row": 0,
                "x": 500.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == jalapeno


def test_roof_driver_uses_jalapeno_against_a_single_catapult() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    jalapeno = codec.plant(0, 3, 4)
    payload = payload_with(
        seed_type=SeedType.JALAPENO,
        legal=[0, jalapeno],
        level=48,
        stage_has_roof=True,
        zombies=[
            {
                "type": 23,
                "row": 3,
                "x": 700.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == jalapeno


def test_roof_driver_reserves_ready_jalapeno_for_a_catapult_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    melon_pult = codec.plant(1, 0, 3)
    payload = payload_with(
        seed_type=SeedType.JALAPENO,
        legal=[0, melon_pult],
        level=49,
        sun=100,
        stage_has_roof=True,
        seed_packets=[
            {
                "slot": 0,
                "type": int(SeedType.JALAPENO),
                "active": True,
                "refresh": 0,
            },
            {
                "slot": 1,
                "type": int(SeedType.MELON_PULT),
                "active": True,
                "refresh": 0,
            },
        ],
        plants=[{"type": int(SeedType.FLOWER_POT), "row": 3, "col": 4}],
        zombies=[
            {
                "type": 23,
                "row": 3,
                "x": 700.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == 0


def test_roof_driver_uses_same_lane_tall_nut_for_a_pogo_zombie() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    tall_nut = codec.plant(0, 4, 6)
    squash = codec.plant(1, 4, 5)
    payload = payload_with(
        seed_type=SeedType.TALL_NUT,
        legal=[0, squash, tall_nut],
        level=44,
        stage_has_roof=True,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.TALL_NUT)},
            {"slot": 1, "type": int(SeedType.SQUASH)},
        ],
        zombies=[
            {
                "type": 18,
                "row": 4,
                "x": 700.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == tall_nut


def test_roof_driver_reserves_sun_for_ready_tall_nut_on_a_pure_pot() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    potato_mine = codec.plant(1, 4, 4)
    payload = payload_with(
        seed_type=SeedType.TALL_NUT,
        legal=[0, potato_mine],
        level=44,
        sun=75,
        stage_has_roof=True,
        seed_packets=[
            {
                "slot": 0,
                "type": int(SeedType.TALL_NUT),
                "active": True,
            },
            {
                "slot": 1,
                "type": int(SeedType.POTATO_MINE),
                "active": True,
            },
        ],
        plants=[{"type": int(SeedType.FLOWER_POT), "row": 4, "col": 6}],
        zombies=[
            {
                "type": 18,
                "row": 4,
                "x": 700.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == 0


def test_roof_driver_does_not_wait_for_a_tall_nut_on_cooldown() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    potato_mine = codec.plant(1, 4, 4)
    payload = payload_with(
        seed_type=SeedType.TALL_NUT,
        legal=[0, potato_mine],
        level=44,
        sun=75,
        stage_has_roof=True,
        seed_packets=[
            {
                "slot": 0,
                "type": int(SeedType.TALL_NUT),
                "active": False,
                "refresh": 100,
            },
            {
                "slot": 1,
                "type": int(SeedType.POTATO_MINE),
                "active": True,
            },
        ],
        plants=[{"type": int(SeedType.FLOWER_POT), "row": 4, "col": 6}],
        zombies=[
            {
                "type": 18,
                "row": 4,
                "x": 500.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == potato_mine


def test_late_roof_driver_builds_middle_tall_nut_after_lane_coverage() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    tall_nut = codec.plant(1, 2, 6)
    payload = payload_with(
        seed_type=SeedType.KERNEL_PULT,
        legal=[0, tall_nut],
        level=44,
        sun=125,
        stage_has_roof=True,
        seed_packets=[
            {
                "slot": 0,
                "type": int(SeedType.KERNEL_PULT),
                "active": True,
                "refresh": 0,
            },
            {
                "slot": 1,
                "type": int(SeedType.TALL_NUT),
                "active": True,
                "refresh": 0,
            },
        ],
        plants=[
            *[
                {"type": int(SeedType.SUNFLOWER), "row": row, "col": 0}
                for row in range(5)
            ],
            *[
                {"type": int(SeedType.KERNEL_PULT), "row": row, "col": 2}
                for row in range(5)
            ],
            {
                "type": int(SeedType.FLOWER_POT),
                "row": 2,
                "col": 6,
            },
        ],
    )
    assert driver.choose_action(payload) == tall_nut


def test_driver_uses_an_affordable_fallback_shooter_for_a_close_threat() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    snow_pea = codec.plant(1, 5, 2)
    payload = payload_with(
        seed_type=SeedType.REPEATER,
        legal=[0, snow_pea],
        level=21,
        row_count=6,
        stage_has_pool=True,
        sun=175,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.REPEATER)},
            {"slot": 1, "type": int(SeedType.SNOW_PEA)},
        ],
        zombies=[
            {
                "type": 0,
                "row": 5,
                "x": 400.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == snow_pea


def test_early_adventure_does_not_apply_late_instant_conservation() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    cherry = codec.plant(0, 2, 4)
    payload = payload_with(
        seed_type=SeedType.CHERRY_BOMB,
        legal=[0, cherry],
        level=9,
        zombies=[
            {
                "type": 0,
                "row": 2,
                "x": 300.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == cherry


def test_early_pool_defense_precedes_a_second_shooter_layer() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    wall = codec.plant(0, 0, 6)
    repeater = codec.plant(1, 0, 3)
    payload = payload_with(
        seed_type=SeedType.WALL_NUT,
        legal=[0, wall, repeater],
        level=21,
        row_count=6,
        stage_has_pool=True,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.WALL_NUT)},
            {"slot": 1, "type": int(SeedType.REPEATER)},
        ],
        plants=[
            {"type": int(SeedType.PEASHOOTER), "row": row, "col": 2} for row in range(6)
        ],
    )
    assert driver.choose_action(payload) == wall


def test_pool_conveyor_reserves_lily_pads_for_shooters() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    land_wall = codec.plant(0, 0, 6)
    water_wall = codec.plant(0, 3, 2)
    payload = payload_with(
        seed_type=SeedType.WALL_NUT,
        legal=[0, water_wall, land_wall],
        level=25,
        row_count=6,
        stage_has_pool=True,
        has_conveyor=True,
        plants=[{"type": int(SeedType.LILY_PAD), "row": 3, "col": 2}],
        zombies=[
            {
                "type": 0,
                "row": 3,
                "x": 300.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == land_wall


def test_pool_conveyor_establishes_water_coverage_before_surplus_land() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    land_pea = codec.plant(0, 0, 3)
    water_pea = codec.plant(0, 3, 2)
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, land_pea, water_pea],
        level=25,
        row_count=6,
        stage_has_pool=True,
        has_conveyor=True,
        plants=[{"type": int(SeedType.LILY_PAD), "row": 3, "col": 2}],
        zombies=[
            {
                "type": 0,
                "row": 0,
                "x": 300.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == water_pea


def test_pool_conveyor_builds_water_defense_after_lane_coverage() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    lily = codec.plant(0, 3, 6)
    land_wall = codec.plant(1, 0, 6)
    common = {
        "level": 25,
        "row_count": 6,
        "stage_has_pool": True,
        "has_conveyor": True,
        "seed_packets": [
            {"slot": 0, "type": int(SeedType.LILY_PAD)},
            {"slot": 1, "type": int(SeedType.WALL_NUT)},
        ],
        "plants": [
            {"type": int(SeedType.LILY_PAD), "row": 3, "col": 2},
            {"type": int(SeedType.PEASHOOTER), "row": 3, "col": 2},
        ],
        "zombies": [],
    }
    preparing_payload = payload_with(
        seed_type=SeedType.LILY_PAD,
        legal=[0, lily, land_wall],
        **common,
    )
    assert driver.choose_action(preparing_payload) == lily

    water_wall = codec.plant(1, 3, 6)
    finishing_payload = {
        **preparing_payload,
        "legal_actions": [0, water_wall, land_wall],
        "plants": [
            {"type": int(SeedType.LILY_PAD), "row": 3, "col": 2},
            {"type": int(SeedType.LILY_PAD), "row": 3, "col": 6},
        ],
    }
    assert driver.choose_action(finishing_payload) == water_wall


def test_pool_conveyor_balances_a_second_water_shooter_layer() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    weaker_water_lane = codec.plant(0, 2, 3)
    stronger_water_lane = codec.plant(0, 3, 4)
    payload = payload_with(
        seed_type=SeedType.PEASHOOTER,
        legal=[0, weaker_water_lane, stronger_water_lane],
        level=25,
        row_count=6,
        stage_has_pool=True,
        has_conveyor=True,
        plants=[
            {"type": int(SeedType.LILY_PAD), "row": 2, "col": 2},
            {"type": int(SeedType.PEASHOOTER), "row": 2, "col": 2},
            {"type": int(SeedType.LILY_PAD), "row": 2, "col": 3},
            {"type": int(SeedType.LILY_PAD), "row": 3, "col": 2},
            {"type": int(SeedType.PEASHOOTER), "row": 3, "col": 2},
            {"type": int(SeedType.LILY_PAD), "row": 3, "col": 3},
            {"type": int(SeedType.PEASHOOTER), "row": 3, "col": 3},
            {"type": int(SeedType.LILY_PAD), "row": 3, "col": 4},
        ],
    )
    assert driver.choose_action(payload) == weaker_water_lane


def test_bungee_blitz_conveyor_spreads_chompers_across_roof_lanes() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    crowded_lane = codec.plant(0, 0, 3)
    uncovered_lane = codec.plant(0, 4, 3)
    payload = payload_with(
        seed_type=SeedType.CHOMPER,
        legal=[0, crowded_lane, uncovered_lane],
        level=45,
        stage_has_roof=True,
        has_conveyor=True,
        plants=[
            {"type": int(SeedType.FLOWER_POT), "row": 0, "col": 3},
            {"type": int(SeedType.CHOMPER), "row": 0, "col": 2},
            {"type": int(SeedType.FLOWER_POT), "row": 4, "col": 3},
        ],
    )
    assert driver.choose_action(payload) == uncovered_lane


def test_zomboss_conveyor_spreads_pults_without_treating_boss_as_lane_zero() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    crowded_lane = codec.plant(0, 0, 3)
    uncovered_lane = codec.plant(0, 4, 3)
    payload = payload_with(
        seed_type=SeedType.KERNEL_PULT,
        legal=[0, crowded_lane, uncovered_lane],
        level=50,
        stage_has_roof=True,
        has_conveyor=True,
        plants=[
            {"type": int(SeedType.FLOWER_POT), "row": 0, "col": 3},
            {"type": int(SeedType.KERNEL_PULT), "row": 0, "col": 2},
            {"type": int(SeedType.FLOWER_POT), "row": 4, "col": 3},
        ],
        zombies=[
            {
                "type": 25,
                "row": 0,
                "x": 0.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == uncovered_lane


def test_zomboss_conveyor_uses_surplus_jalapeno_before_pults() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    jalapeno = codec.plant(0, 1, 4)
    second_jalapeno = codec.plant(1, 1, 4)
    kernel_pult = codec.plant(2, 4, 3)
    payload = payload_with(
        seed_type=SeedType.JALAPENO,
        legal=[0, kernel_pult, second_jalapeno, jalapeno],
        level=50,
        stage_has_roof=True,
        has_conveyor=True,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.JALAPENO)},
            {"slot": 1, "type": int(SeedType.JALAPENO)},
            {"slot": 2, "type": int(SeedType.KERNEL_PULT)},
        ],
        zombies=[
            {
                "type": 25,
                "row": 0,
                "x": 0.0,
                "mind_controlled": False,
            },
            {
                "type": 18,
                "row": 1,
                "x": 250.0,
                "mind_controlled": False,
            },
        ],
    )
    assert driver.choose_action(payload) == jalapeno


def test_zomboss_conveyor_uses_ice_shroom_against_active_fireball() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    ice_shroom = codec.plant(0, 2, 6)
    payload = payload_with(
        seed_type=SeedType.ICE_SHROOM,
        legal=[0, ice_shroom],
        level=50,
        stage_has_roof=True,
        has_conveyor=True,
        zombies=[
            {
                "type": 25,
                "row": 0,
                "x": 0.0,
                "boss_ball_active": True,
                "fireball_row": 2,
                "is_fireball": True,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == ice_shroom


def test_zomboss_conveyor_uses_jalapeno_against_active_iceball() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    jalapeno = codec.plant(0, 3, 6)
    payload = payload_with(
        seed_type=SeedType.JALAPENO,
        legal=[0, jalapeno],
        level=50,
        stage_has_roof=True,
        has_conveyor=True,
        zombies=[
            {
                "type": 25,
                "row": 0,
                "x": 0.0,
                "boss_ball_active": True,
                "fireball_row": 3,
                "is_fireball": False,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == jalapeno


def test_zomboss_conveyor_holds_jalapeno_without_a_lane_threat() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    jalapeno = codec.plant(0, 0, 4)
    payload = payload_with(
        seed_type=SeedType.JALAPENO,
        legal=[0, jalapeno],
        level=50,
        stage_has_roof=True,
        has_conveyor=True,
        zombies=[
            {
                "type": 25,
                "row": 0,
                "x": 0.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == 0


def test_fog_conveyor_uses_blover_for_an_active_balloon() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    blover = codec.plant(0, 0, 0)
    cactus = codec.plant(1, 4, 2)
    payload = payload_with(
        seed_type=SeedType.BLOVER,
        legal=[0, cactus, blover],
        level=40,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        has_conveyor=True,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.BLOVER)},
            {"slot": 1, "type": int(SeedType.CACTUS)},
        ],
        zombies=[
            {
                "type": 16,
                "row": 4,
                "x": 350.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == blover


def test_fog_conveyor_holds_blover_until_a_balloon_appears() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    blover = codec.plant(0, 0, 0)
    payload = payload_with(
        seed_type=SeedType.BLOVER,
        legal=[0, blover],
        level=40,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        has_conveyor=True,
    )
    assert driver.choose_action(payload) == 0


def test_fog_conveyor_waits_for_one_blover_to_clear_the_sky() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    first_blover = codec.plant(0, 0, 0)
    second_blover = codec.plant(1, 0, 1)
    balloon = {
        "type": 16,
        "row": 4,
        "x": 700.0,
        "mind_controlled": False,
    }
    payload = payload_with(
        seed_type=SeedType.BLOVER,
        legal=[0, first_blover, second_blover],
        level=40,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        has_conveyor=True,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.BLOVER)},
            {"slot": 1, "type": int(SeedType.BLOVER)},
        ],
        zombies=[balloon],
    )
    assert driver.choose_action(payload) == first_blover
    assert driver.choose_action(payload) == 0

    cleared_payload = {**payload, "zombies": []}
    assert driver.choose_action(cleared_payload) == 0
    assert driver.choose_action(payload) == first_blover


def test_pool_loadout_prioritizes_support_economy_and_instant_defense() -> None:
    payload = {
        "seed_bank_capacity": 6,
        "available_seed_types": list(range(17)),
        "stage_has_roof": False,
        "stage_is_night": False,
        "stage_has_pool": True,
    }
    assert recommended_seed_types(payload) == (
        SeedType.LILY_PAD,
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.REPEATER,
        SeedType.SNOW_PEA,
        SeedType.WALL_NUT,
    )


def test_pool_loadouts_follow_world_three_progression() -> None:
    expected = {
        23: (
            SeedType.LILY_PAD,
            SeedType.SUNFLOWER,
            SeedType.SQUASH,
            SeedType.PEASHOOTER,
            SeedType.WALL_NUT,
            SeedType.CHERRY_BOMB,
        ),
        24: (
            SeedType.LILY_PAD,
            SeedType.SUNFLOWER,
            SeedType.TANGLE_KELP,
            SeedType.SNOW_PEA,
            SeedType.REPEATER,
            SeedType.WALL_NUT,
        ),
        26: (
            SeedType.LILY_PAD,
            SeedType.SUNFLOWER,
            SeedType.JALAPENO,
            SeedType.SNOW_PEA,
            SeedType.REPEATER,
            SeedType.WALL_NUT,
        ),
        28: (
            SeedType.LILY_PAD,
            SeedType.SUNFLOWER,
            SeedType.SQUASH,
            SeedType.PEASHOOTER,
            SeedType.REPEATER,
            SeedType.WALL_NUT,
        ),
        29: (
            SeedType.LILY_PAD,
            SeedType.SUNFLOWER,
            SeedType.TANGLE_KELP,
            SeedType.PEASHOOTER,
            SeedType.SNOW_PEA,
            SeedType.SQUASH,
        ),
    }
    for level, loadout in expected.items():
        payload = {
            "level": level,
            "seed_bank_capacity": 6,
            "available_seed_types": list(range(level - 4)),
            "stage_has_roof": False,
            "stage_is_night": False,
            "stage_has_pool": True,
        }
        assert recommended_seed_types(payload) == loadout


def test_night_pool_loadout_uses_free_water_coverage_and_defense() -> None:
    payload = {
        "level": 31,
        "seed_bank_capacity": 6,
        "available_seed_types": list(range(25)),
        "stage_has_roof": False,
        "stage_is_night": True,
        "stage_has_pool": True,
    }
    assert recommended_seed_types(payload) == (
        SeedType.LILY_PAD,
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.SEA_SHROOM,
        SeedType.FUME_SHROOM,
        SeedType.WALL_NUT,
    )


def test_night_pool_loadout_adds_cactus_when_balloon_zombies_unlock() -> None:
    payload = {
        "level": 33,
        "seed_bank_capacity": 6,
        "available_seed_types": list(range(27)),
        "stage_has_roof": False,
        "stage_is_night": True,
        "stage_has_pool": True,
    }
    assert recommended_seed_types(payload) == (
        SeedType.LILY_PAD,
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.SEA_SHROOM,
        SeedType.CACTUS,
        SeedType.WALL_NUT,
    )


def test_night_pool_loadout_switches_to_blover_when_unlocked() -> None:
    payload = {
        "level": 34,
        "seed_bank_capacity": 6,
        "available_seed_types": list(range(28)),
        "stage_has_roof": False,
        "stage_is_night": True,
        "stage_has_pool": True,
    }
    assert recommended_seed_types(payload) == (
        SeedType.LILY_PAD,
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.SEA_SHROOM,
        SeedType.BLOVER,
        SeedType.WALL_NUT,
    )


def test_late_night_pool_loadout_adds_hypno_shroom_defense() -> None:
    payload = {
        "level": 36,
        "seed_bank_capacity": 6,
        "available_seed_types": list(range(28)),
        "stage_has_roof": False,
        "stage_is_night": True,
        "stage_has_pool": True,
    }
    assert recommended_seed_types(payload) == (
        SeedType.LILY_PAD,
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.TANGLE_KELP,
        SeedType.BLOVER,
        SeedType.HYPNO_SHROOM,
    )


def test_driver_places_cactus_in_the_visible_balloon_lane() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    wrong_lane = codec.plant(0, 3, 2)
    balloon_lane = codec.plant(0, 5, 2)
    payload = payload_with(
        seed_type=SeedType.CACTUS,
        legal=[0, wrong_lane, balloon_lane],
        level=33,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        zombies=[
            {
                "type": 16,
                "row": 5,
                "x": 300.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == balloon_lane


def test_driver_holds_paid_actions_until_blover_is_affordable() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    wall = codec.plant(1, 0, 6)
    payload = payload_with(
        seed_type=SeedType.BLOVER,
        legal=[0, wall],
        level=34,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        sun=50,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.BLOVER)},
            {"slot": 1, "type": int(SeedType.WALL_NUT)},
        ],
        zombies=[
            {
                "type": 16,
                "row": 4,
                "x": 300.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == 0


def test_driver_does_not_spend_cactus_without_a_balloon() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    covered_lane = codec.plant(0, 0, 3)
    uncovered_lane = codec.plant(0, 1, 3)
    payload = payload_with(
        seed_type=SeedType.CACTUS,
        legal=[0, covered_lane, uncovered_lane],
        level=33,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        plants=[{"type": int(SeedType.CACTUS), "row": 0, "col": 2}],
    )
    assert driver.choose_action(payload) == 0


def test_driver_uses_only_free_coverage_while_reserving_for_cactus() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    free_puff = codec.plant(1, 0, 3)
    payload = payload_with(
        seed_type=SeedType.CACTUS,
        legal=[0, free_puff],
        level=33,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.CACTUS)},
            {"slot": 1, "type": int(SeedType.PUFF_SHROOM)},
        ],
        plants=[
            {
                "type": int(SeedType.SUN_SHROOM),
                "row": row,
                "col": 0,
            }
            for row in (0, 1, 4, 5)
        ]
        + [
            {
                "type": int(SeedType.PUFF_SHROOM),
                "row": row,
                "col": 2,
            }
            for row in (0, 1, 4, 5)
        ]
        + [{"type": int(SeedType.LILY_PAD), "row": row, "col": 2} for row in (2, 3)]
        + [{"type": int(SeedType.SEA_SHROOM), "row": row, "col": 2} for row in (2, 3)],
    )
    assert driver.choose_action(payload) == free_puff

    wall = codec.plant(1, 0, 6)
    saving_payload = {
        **payload,
        "sun": 100,
        "legal_actions": [0, wall],
        "seed_packets": [
            {"slot": 0, "type": int(SeedType.CACTUS)},
            {"slot": 1, "type": int(SeedType.WALL_NUT)},
        ],
    }
    assert driver.choose_action(saving_payload) == 0

    damaged_payload = {
        **saving_payload,
        "plants": [],
    }
    assert driver.choose_action(damaged_payload) == 0


def test_driver_spends_only_sun_above_blover_reserve_on_fume_shroom() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    fume = codec.plant(1, 4, 3)
    common = {
        "level": 36,
        "row_count": 6,
        "stage_is_night": True,
        "stage_has_pool": True,
        "seed_packets": [
            {"slot": 0, "type": int(SeedType.BLOVER)},
            {"slot": 1, "type": int(SeedType.FUME_SHROOM)},
        ],
        "plants": [
            {
                "type": int(SeedType.PUFF_SHROOM),
                "row": row,
                "col": 2,
            }
            for row in range(6)
        ],
        "zombies": [
            {
                "type": 2,
                "row": 4,
                "x": 500.0,
                "mind_controlled": False,
            }
        ],
    }
    driver._air_reserve_active = True
    assert (
        driver.choose_action(
            payload_with(
                seed_type=SeedType.BLOVER,
                legal=[0, fume],
                sun=150,
                **common,
            )
        )
        == 0
    )
    assert (
        driver.choose_action(
            payload_with(
                seed_type=SeedType.BLOVER,
                legal=[0, fume],
                sun=175,
                **common,
            )
        )
        == fume
    )


def test_driver_uses_fume_shroom_for_an_immediate_ground_breach() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    fume = codec.plant(1, 4, 4)
    driver._air_reserve_active = True
    payload = payload_with(
        seed_type=SeedType.BLOVER,
        legal=[0, fume],
        level=36,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        sun=75,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.BLOVER)},
            {"slot": 1, "type": int(SeedType.FUME_SHROOM)},
        ],
        plants=[
            {
                "type": int(SeedType.PUFF_SHROOM),
                "row": row,
                "col": 2,
            }
            for row in range(6)
        ],
        zombies=[
            {
                "type": 2,
                "row": 4,
                "x": 350.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == fume


def test_driver_uses_hypno_shroom_for_an_immediate_ground_breach() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    hypno = codec.plant(1, 4, 4)
    driver._air_reserve_active = True
    payload = payload_with(
        seed_type=SeedType.BLOVER,
        legal=[0, hypno],
        level=36,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        sun=75,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.BLOVER)},
            {"slot": 1, "type": int(SeedType.HYPNO_SHROOM)},
        ],
        plants=[
            {
                "type": int(SeedType.PUFF_SHROOM),
                "row": row,
                "col": 2,
            }
            for row in range(6)
        ],
        zombies=[
            {
                "type": 2,
                "row": 4,
                "x": 350.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == hypno


def test_driver_rebuilds_pool_support_for_an_immediate_water_breach() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    lily = codec.plant(1, 2, 3)
    driver._air_reserve_active = True
    payload = payload_with(
        seed_type=SeedType.BLOVER,
        legal=[0, lily],
        level=37,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        sun=25,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.BLOVER)},
            {"slot": 1, "type": int(SeedType.LILY_PAD)},
        ],
        plants=[
            {
                "type": int(SeedType.PUFF_SHROOM),
                "row": row,
                "col": 2,
            }
            for row in (0, 1, 4, 5)
        ],
        zombies=[
            {
                "type": 2,
                "row": 2,
                "x": 350.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == lily


def test_driver_uses_tangle_kelp_for_an_immediate_water_breach() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    tangle = codec.plant(1, 3, 4)
    driver._air_reserve_active = True
    payload = payload_with(
        seed_type=SeedType.BLOVER,
        legal=[0, tangle],
        level=36,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        sun=25,
        seed_packets=[
            {"slot": 0, "type": int(SeedType.BLOVER)},
            {"slot": 1, "type": int(SeedType.TANGLE_KELP)},
        ],
        plants=[
            {
                "type": int(SeedType.PUFF_SHROOM),
                "row": row,
                "col": 2,
            }
            for row in (0, 1, 4, 5)
        ],
        zombies=[
            {
                "type": 2,
                "row": 3,
                "x": 350.0,
                "mind_controlled": False,
            }
        ],
    )
    assert driver.choose_action(payload) == tangle


def test_roof_loadout_uses_pults_instead_of_flat_projectiles() -> None:
    payload = {
        "seed_bank_capacity": 6,
        "available_seed_types": list(range(40)),
        "stage_has_roof": True,
        "stage_is_night": False,
        "stage_has_pool": False,
    }
    assert recommended_seed_types(payload) == (
        SeedType.FLOWER_POT,
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.MELON_PULT,
        SeedType.KERNEL_PULT,
        SeedType.CABBAGE_PULT,
    )


def test_early_roof_loadout_uses_squash_against_pole_vaulters() -> None:
    payload = {
        "level": 42,
        "seed_bank_capacity": 6,
        "available_seed_types": list(range(34)),
        "stage_has_roof": True,
        "stage_is_night": False,
        "stage_has_pool": False,
    }
    assert recommended_seed_types(payload) == (
        SeedType.FLOWER_POT,
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.CABBAGE_PULT,
        SeedType.SQUASH,
        SeedType.TALL_NUT,
    )


def test_kernel_pult_roof_loadout_adds_durable_defense() -> None:
    payload = {
        "level": 43,
        "seed_bank_capacity": 6,
        "available_seed_types": list(range(35)),
        "stage_has_roof": True,
        "stage_is_night": False,
        "stage_has_pool": False,
    }
    assert recommended_seed_types(payload) == (
        SeedType.FLOWER_POT,
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.KERNEL_PULT,
        SeedType.SQUASH,
        SeedType.TALL_NUT,
    )


def test_pogo_roof_loadout_uses_jalapeno_for_late_lane_clears() -> None:
    payload = {
        "level": 44,
        "seed_bank_capacity": 6,
        "available_seed_types": list(range(35)),
        "stage_has_roof": True,
        "stage_is_night": False,
        "stage_has_pool": False,
    }
    assert recommended_seed_types(payload) == (
        SeedType.FLOWER_POT,
        SeedType.SUNFLOWER,
        SeedType.JALAPENO,
        SeedType.KERNEL_PULT,
        SeedType.SQUASH,
        SeedType.TALL_NUT,
    )


def test_threepeater_counts_as_adjacent_lane_coverage() -> None:
    assert _shooter_counts({(2, 3): [SeedType.THREEPEATER]}, tuple(range(6))) == {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 0,
        5: 0,
    }


def test_cattail_counts_as_all_lane_coverage() -> None:
    assert _shooter_counts({(2, 3): [SeedType.CATTAIL]}, tuple(range(6))) == {
        row: 1 for row in range(6)
    }


def test_driver_reserves_sun_for_a_selected_cattail_upgrade() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    puff = codec.plant(0, 0, 2)
    fume = codec.plant(1, 0, 2)
    payload = payload_with(
        seed_type=SeedType.PUFF_SHROOM,
        legal=[0, puff, fume],
        game_mode=4,
        survival_stage=1,
        sun=200,
        stage_is_night=True,
        stage_has_pool=True,
        plants=[{"type": SeedType.LILY_PAD, "row": 2, "col": 3}],
        seed_packets=[
            {"slot": 0, "type": int(SeedType.PUFF_SHROOM)},
            {"slot": 1, "type": int(SeedType.FUME_SHROOM)},
            {"slot": 2, "type": int(SeedType.CATTAIL)},
        ],
    )
    assert driver.choose_action(payload) == puff


def test_driver_uses_a_ready_cattail_upgrade() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    cattail = codec.plant(0, 2, 3)
    payload = payload_with(
        seed_type=SeedType.CATTAIL,
        legal=[0, cattail],
        game_mode=4,
        survival_stage=1,
        sun=225,
        stage_is_night=True,
        stage_has_pool=True,
        plants=[{"type": SeedType.LILY_PAD, "row": 2, "col": 3}],
    )
    assert driver.choose_action(payload) == cattail


def test_driver_reserves_sun_for_a_second_cattail() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    puff = codec.plant(0, 0, 2)
    payload = payload_with(
        seed_type=SeedType.PUFF_SHROOM,
        legal=[0, puff],
        game_mode=4,
        survival_stage=2,
        sun=200,
        stage_is_night=True,
        stage_has_pool=True,
        plants=[
            {"type": SeedType.LILY_PAD, "row": 2, "col": 3},
            {"type": SeedType.CATTAIL, "row": 3, "col": 3},
        ],
        seed_packets=[
            {"slot": 0, "type": int(SeedType.PUFF_SHROOM)},
            {"slot": 1, "type": int(SeedType.CATTAIL)},
        ],
    )
    assert driver.choose_action(payload) == puff


def test_driver_protects_a_cattail_with_pumpkin() -> None:
    codec = PortableActionCodec()
    driver = NativeBoardDriver()
    pumpkin = codec.plant(0, 2, 3)
    payload = payload_with(
        seed_type=SeedType.PUMPKIN,
        legal=[0, pumpkin],
        game_mode=4,
        survival_stage=2,
        row_count=6,
        stage_is_night=True,
        stage_has_pool=True,
        plants=[
            {"type": SeedType.SUN_SHROOM, "row": row, "col": 0} for row in (0, 1, 4, 5)
        ]
        + [
            {"type": SeedType.LILY_PAD, "row": 2, "col": 3},
            {"type": SeedType.CATTAIL, "row": 2, "col": 3},
            {"type": SeedType.LILY_PAD, "row": 3, "col": 3},
            {"type": SeedType.CATTAIL, "row": 3, "col": 3},
        ],
    )
    assert driver.choose_action(payload) == pumpkin

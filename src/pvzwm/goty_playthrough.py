from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from pvzwm.portable import PortableActionCodec


class SeedType(IntEnum):
    PEASHOOTER = 0
    SUNFLOWER = 1
    CHERRY_BOMB = 2
    WALL_NUT = 3
    POTATO_MINE = 4
    SNOW_PEA = 5
    CHOMPER = 6
    REPEATER = 7
    PUFF_SHROOM = 8
    SUN_SHROOM = 9
    FUME_SHROOM = 10
    GRAVE_BUSTER = 11
    HYPNO_SHROOM = 12
    SCAREDY_SHROOM = 13
    ICE_SHROOM = 14
    DOOM_SHROOM = 15
    LILY_PAD = 16
    SQUASH = 17
    THREEPEATER = 18
    TANGLE_KELP = 19
    JALAPENO = 20
    SPIKEWEED = 21
    TORCHWOOD = 22
    TALL_NUT = 23
    SEA_SHROOM = 24
    PLANTERN = 25
    CACTUS = 26
    BLOVER = 27
    SPLIT_PEA = 28
    STARFRUIT = 29
    PUMPKIN = 30
    MAGNET_SHROOM = 31
    CABBAGE_PULT = 32
    FLOWER_POT = 33
    KERNEL_PULT = 34
    COFFEE_BEAN = 35
    GARLIC = 36
    UMBRELLA_LEAF = 37
    MARIGOLD = 38
    MELON_PULT = 39
    GATLING_PEA = 40
    TWIN_SUNFLOWER = 41
    GLOOM_SHROOM = 42
    CATTAIL = 43
    WINTER_MELON = 44
    GOLD_MAGNET = 45
    SPIKEROCK = 46
    COB_CANNON = 47
    EXPLODE_O_NUT = 49
    GIANT_WALL_NUT = 50
    LEFTPEATER = 52
    ZOMBIE_NORMAL = 60
    ZOMBIE_TRAFFIC_CONE = 61
    ZOMBIE_POLE_VAULTER = 62
    ZOMBIE_BUCKET = 63
    ZOMBIE_LADDER = 64
    ZOMBIE_DIGGER = 65
    ZOMBIE_BUNGEE = 66
    ZOMBIE_FOOTBALL = 67
    ZOMBIE_BALLOON = 68
    ZOMBIE_SCREEN_DOOR = 69
    ZOMBONI = 70
    ZOMBIE_POGO = 71
    ZOMBIE_DANCER = 72
    ZOMBIE_GARGANTUAR = 73
    ZOMBIE_IMP = 74


@dataclass(frozen=True, slots=True)
class LegalPlant:
    action: int
    slot: int
    row: int
    col: int
    seed_type: int


ECONOMY_TYPES = {SeedType.SUNFLOWER, SeedType.SUN_SHROOM}
SUPPORT_TYPES = {SeedType.LILY_PAD, SeedType.FLOWER_POT}
SHOOTER_TYPES = {
    SeedType.PEASHOOTER,
    SeedType.SNOW_PEA,
    SeedType.REPEATER,
    SeedType.PUFF_SHROOM,
    SeedType.FUME_SHROOM,
    SeedType.SCAREDY_SHROOM,
    SeedType.THREEPEATER,
    SeedType.SEA_SHROOM,
    SeedType.CACTUS,
    SeedType.SPLIT_PEA,
    SeedType.STARFRUIT,
    SeedType.CABBAGE_PULT,
    SeedType.KERNEL_PULT,
    SeedType.MELON_PULT,
    SeedType.GLOOM_SHROOM,
    SeedType.CATTAIL,
    SeedType.WINTER_MELON,
    SeedType.LEFTPEATER,
}
VASE_OFFENSE_TYPES = SHOOTER_TYPES | {SeedType.SPIKEWEED}
VASE_PATH_CONTROL_TYPES = {
    SeedType.SPIKEWEED,
    SeedType.TALL_NUT,
    SeedType.PUMPKIN,
}
VASE_SEED_OPEN_PRIORITY = {
    SeedType.WALL_NUT: 0,
    SeedType.TALL_NUT: 0,
    SeedType.PUMPKIN: 0,
    SeedType.SQUASH: 1,
}
PEA_PROJECTILE_TYPES = {
    SeedType.PEASHOOTER,
    SeedType.SNOW_PEA,
    SeedType.REPEATER,
    SeedType.THREEPEATER,
    SeedType.SPLIT_PEA,
}
DEFENSE_TYPES = {SeedType.WALL_NUT, SeedType.TALL_NUT, SeedType.PUMPKIN}
INSTANT_TYPES = {
    SeedType.CHERRY_BOMB,
    SeedType.POTATO_MINE,
    SeedType.SQUASH,
    SeedType.TANGLE_KELP,
    SeedType.JALAPENO,
}
INSTANT_PRIORITY = {
    SeedType.CHERRY_BOMB: 0,
    SeedType.JALAPENO: 1,
    SeedType.SQUASH: 2,
    SeedType.TANGLE_KELP: 3,
    SeedType.POTATO_MINE: 4,
}

I_ZOMBIE_STRENGTH = {
    SeedType.ZOMBIE_IMP: 10,
    SeedType.ZOMBIE_NORMAL: 20,
    SeedType.ZOMBIE_TRAFFIC_CONE: 35,
    SeedType.ZOMBIE_POLE_VAULTER: 40,
    SeedType.ZOMBIE_SCREEN_DOOR: 55,
    SeedType.ZOMBIE_BUCKET: 65,
    SeedType.ZOMBIE_DIGGER: 70,
    SeedType.ZOMBIE_LADDER: 75,
    SeedType.ZOMBIE_BALLOON: 80,
    SeedType.ZOMBIE_POGO: 85,
    SeedType.ZOMBONI: 90,
    SeedType.ZOMBIE_FOOTBALL: 95,
    SeedType.ZOMBIE_DANCER: 100,
    SeedType.ZOMBIE_GARGANTUAR: 110,
    SeedType.ZOMBIE_BUNGEE: 0,
}
I_ZOMBIE_COST = {
    SeedType.ZOMBIE_NORMAL: 50,
    SeedType.ZOMBIE_TRAFFIC_CONE: 75,
    SeedType.ZOMBIE_POLE_VAULTER: 75,
    SeedType.ZOMBIE_BUCKET: 125,
    SeedType.ZOMBIE_LADDER: 150,
    SeedType.ZOMBIE_DIGGER: 125,
    SeedType.ZOMBIE_BUNGEE: 125,
    SeedType.ZOMBIE_FOOTBALL: 175,
    SeedType.ZOMBIE_BALLOON: 150,
    SeedType.ZOMBIE_SCREEN_DOOR: 100,
    SeedType.ZOMBONI: 175,
    SeedType.ZOMBIE_POGO: 200,
    SeedType.ZOMBIE_DANCER: 350,
    SeedType.ZOMBIE_GARGANTUAR: 300,
    SeedType.ZOMBIE_IMP: 50,
}
I_ZOMBIE_PLANT_THREAT = {
    SeedType.PEASHOOTER: 2,
    SeedType.POTATO_MINE: 4,
    SeedType.SNOW_PEA: 4,
    SeedType.CHOMPER: 5,
    SeedType.REPEATER: 4,
    SeedType.FUME_SHROOM: 3,
    SeedType.SCAREDY_SHROOM: 2,
    SeedType.SQUASH: 5,
    SeedType.THREEPEATER: 3,
    SeedType.SPIKEWEED: 2,
    SeedType.CACTUS: 2,
    SeedType.SPLIT_PEA: 3,
    SeedType.STARFRUIT: 3,
    SeedType.KERNEL_PULT: 3,
}

SHOOTER_PRIORITY = {
    SeedType.WINTER_MELON: 14,
    SeedType.THREEPEATER: 13,
    SeedType.MELON_PULT: 12,
    SeedType.GLOOM_SHROOM: 12,
    SeedType.CATTAIL: 15,
    SeedType.REPEATER: 11,
    SeedType.SNOW_PEA: 10,
    SeedType.CACTUS: 9,
    SeedType.KERNEL_PULT: 7,
    SeedType.CABBAGE_PULT: 6,
    SeedType.FUME_SHROOM: 5,
    SeedType.SCAREDY_SHROOM: 4,
    SeedType.PEASHOOTER: 3,
    SeedType.SEA_SHROOM: 2,
    SeedType.PUFF_SHROOM: 1,
}


def recommended_seed_types(payload: dict[str, Any]) -> tuple[int, ...]:
    """Choose a legal terrain-aware loadout from the profile's unlocked seeds."""

    capacity = int(payload.get("seed_bank_capacity", 0))
    available = {
        int(seed_type) for seed_type in payload.get("available_seed_types", [])
    }
    if capacity < 1:
        return ()

    if bool(payload.get("stage_has_roof", False)):
        level = int(payload.get("level", 0))
        priority = (
            SeedType.FLOWER_POT,
            SeedType.SUNFLOWER,
            *((SeedType.JALAPENO,) if level >= 44 else (SeedType.POTATO_MINE,)),
            SeedType.MELON_PULT,
            SeedType.KERNEL_PULT,
            *(
                (
                    SeedType.SQUASH,
                    SeedType.TALL_NUT,
                    SeedType.CABBAGE_PULT,
                )
                if level >= 43
                else (
                    SeedType.CABBAGE_PULT,
                    SeedType.SQUASH,
                    SeedType.TALL_NUT,
                )
            ),
            SeedType.POTATO_MINE,
            SeedType.WALL_NUT,
            SeedType.CHERRY_BOMB,
            SeedType.JALAPENO,
        )
    elif bool(payload.get("stage_is_night", False)):
        if bool(payload.get("stage_has_pool", False)):
            level = int(payload.get("level", 0))
            priority = (
                SeedType.LILY_PAD,
                SeedType.SUN_SHROOM,
                SeedType.PUFF_SHROOM,
                *((SeedType.TANGLE_KELP,) if level == 36 else (SeedType.SEA_SHROOM,)),
                *(
                    (SeedType.BLOVER,)
                    if level >= 34 and int(SeedType.BLOVER) in available
                    else (
                        (SeedType.CACTUS,)
                        if level >= 33 and int(SeedType.CACTUS) in available
                        else (SeedType.FUME_SHROOM,)
                    )
                ),
                *(
                    (
                        SeedType.HYPNO_SHROOM,
                        SeedType.FUME_SHROOM,
                        SeedType.WALL_NUT,
                    )
                    if level >= 36
                    else (SeedType.WALL_NUT, SeedType.FUME_SHROOM)
                ),
                SeedType.SCAREDY_SHROOM,
                SeedType.DOOM_SHROOM,
                SeedType.ICE_SHROOM,
                SeedType.PLANTERN,
                SeedType.BLOVER,
            )
        else:
            priority = (
                SeedType.SUN_SHROOM,
                SeedType.PUFF_SHROOM,
                SeedType.FUME_SHROOM,
                SeedType.SCAREDY_SHROOM,
                SeedType.HYPNO_SHROOM,
                SeedType.DOOM_SHROOM,
                SeedType.ICE_SHROOM,
                SeedType.GRAVE_BUSTER,
                SeedType.PLANTERN,
                SeedType.BLOVER,
            )
    elif bool(payload.get("stage_has_pool", False)):
        level = int(payload.get("level", 0))
        if level == 23 and int(SeedType.SQUASH) in available:
            priority = (
                SeedType.LILY_PAD,
                SeedType.SUNFLOWER,
                SeedType.SQUASH,
                SeedType.PEASHOOTER,
                SeedType.WALL_NUT,
                SeedType.CHERRY_BOMB,
                SeedType.POTATO_MINE,
                SeedType.SNOW_PEA,
                SeedType.REPEATER,
                SeedType.TANGLE_KELP,
                SeedType.JALAPENO,
            )
        elif level == 28 and int(SeedType.SQUASH) in available:
            priority = (
                SeedType.LILY_PAD,
                SeedType.SUNFLOWER,
                SeedType.SQUASH,
                SeedType.PEASHOOTER,
                SeedType.REPEATER,
                SeedType.WALL_NUT,
                SeedType.TANGLE_KELP,
                SeedType.SNOW_PEA,
                SeedType.POTATO_MINE,
                SeedType.CHERRY_BOMB,
                SeedType.JALAPENO,
                SeedType.THREEPEATER,
            )
        elif level == 29 and int(SeedType.TANGLE_KELP) in available:
            priority = (
                SeedType.LILY_PAD,
                SeedType.SUNFLOWER,
                SeedType.TANGLE_KELP,
                SeedType.PEASHOOTER,
                SeedType.SNOW_PEA,
                SeedType.SQUASH,
                SeedType.REPEATER,
                SeedType.WALL_NUT,
                SeedType.POTATO_MINE,
                SeedType.CHERRY_BOMB,
                SeedType.JALAPENO,
                SeedType.THREEPEATER,
            )
        elif level >= 30 and int(SeedType.TANGLE_KELP) in available:
            priority = (
                SeedType.LILY_PAD,
                SeedType.SUNFLOWER,
                SeedType.TANGLE_KELP,
                SeedType.PEASHOOTER,
                SeedType.REPEATER,
                SeedType.WALL_NUT,
                SeedType.SQUASH,
                SeedType.SNOW_PEA,
                SeedType.POTATO_MINE,
                SeedType.CHERRY_BOMB,
                SeedType.JALAPENO,
                SeedType.THREEPEATER,
            )
        elif level >= 26 and int(SeedType.JALAPENO) in available:
            priority = (
                SeedType.LILY_PAD,
                SeedType.SUNFLOWER,
                SeedType.JALAPENO,
                SeedType.SNOW_PEA,
                SeedType.REPEATER,
                SeedType.WALL_NUT,
                SeedType.TANGLE_KELP,
                SeedType.SQUASH,
                SeedType.POTATO_MINE,
                SeedType.CHERRY_BOMB,
                SeedType.THREEPEATER,
                SeedType.PEASHOOTER,
            )
        elif level >= 24 and int(SeedType.TANGLE_KELP) in available:
            priority = (
                SeedType.LILY_PAD,
                SeedType.SUNFLOWER,
                SeedType.TANGLE_KELP,
                SeedType.SNOW_PEA,
                SeedType.REPEATER,
                SeedType.WALL_NUT,
                SeedType.SQUASH,
                SeedType.POTATO_MINE,
                SeedType.CHERRY_BOMB,
                SeedType.THREEPEATER,
                SeedType.PEASHOOTER,
                SeedType.JALAPENO,
            )
        else:
            primary_pool_instant = (
                SeedType.SQUASH
                if level >= 23 and int(SeedType.SQUASH) in available
                else SeedType.POTATO_MINE
            )
            priority = (
                SeedType.LILY_PAD,
                SeedType.SUNFLOWER,
                primary_pool_instant,
                SeedType.REPEATER,
                SeedType.SNOW_PEA,
                SeedType.WALL_NUT,
                SeedType.POTATO_MINE,
                SeedType.PEASHOOTER,
                SeedType.CHERRY_BOMB,
                SeedType.TANGLE_KELP,
                SeedType.JALAPENO,
            )
    else:
        priority = (
            SeedType.SUNFLOWER,
            SeedType.POTATO_MINE,
            SeedType.REPEATER,
            SeedType.SNOW_PEA,
            SeedType.WALL_NUT,
            SeedType.CHERRY_BOMB,
            SeedType.SQUASH,
            SeedType.JALAPENO,
            SeedType.PEASHOOTER,
        )

    selected = list(
        dict.fromkeys(
            int(seed_type) for seed_type in priority if int(seed_type) in available
        )
    )
    if len(selected) < capacity:
        selected.extend(sorted(available - set(selected)))
    return tuple(selected[:capacity])


MODE_SEED_PRIORITY: dict[int, tuple[SeedType, ...]] = {
    # Survival: Day needs an affordable opening. Repeater-first chooser
    # recommendations leave the verifier unable to establish five lanes
    # before the early waves, whereas Peashooter plus Potato Mine matches the
    # ordinary low-cost human opening and can be reinforced across stages.
    1: (
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.PEASHOOTER,
        SeedType.SNOW_PEA,
        SeedType.WALL_NUT,
        SeedType.CHERRY_BOMB,
    ),
    # Survival: Night carries the lawn between five ten-wave stages. Prefer
    # packets the verifier can use continuously: free Puff-shrooms preserve
    # sun, Fume/Scaredy-shrooms provide sustained damage, Wall-nuts preserve
    # the accumulated formation, and Cherry Bomb handles dense flag waves.
    2: (
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.FUME_SHROOM,
        SeedType.GLOOM_SHROOM,
        SeedType.PUMPKIN,
        SeedType.WALL_NUT,
        SeedType.CHERRY_BOMB,
    ),
    # Survival: Pool needs affordable six-lane coverage before its first flag.
    # Repeater is legal but the opening economy never affords one; Peashooter
    # supplies actual damage while preserving Lily Pad and pool counters.
    3: (
        SeedType.LILY_PAD,
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.PEASHOOTER,
        SeedType.SNOW_PEA,
        SeedType.WALL_NUT,
    ),
    # Preserve the proven six-packet Roof opening even on a purchased
    # ten-slot profile. The four extra packets are intentionally passive for
    # the verifier, avoiding premature counter/defense branches until the
    # first real Survival repick.
    5: (
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
    ),
    # Bobsled Bonanza: remove ice trails and stop Zombonis before they can
    # establish repeated bobsled lanes.
    28: (
        SeedType.LILY_PAD,
        SeedType.SUNFLOWER,
        SeedType.JALAPENO,
        SeedType.SPIKEWEED,
        SeedType.SQUASH,
        SeedType.TALL_NUT,
    ),
    # Last Stand has no generated sun and disallows economy mushrooms/plants.
    31: (
        SeedType.LILY_PAD,
        SeedType.THREEPEATER,
        SeedType.REPEATER,
        SeedType.TORCHWOOD,
        SeedType.TALL_NUT,
        SeedType.PUMPKIN,
    ),
    34: (
        SeedType.FLOWER_POT,
        SeedType.SUNFLOWER,
        SeedType.TALL_NUT,
        SeedType.MAGNET_SHROOM,
        SeedType.MELON_PULT,
        SeedType.SQUASH,
    ),
    36: (
        SeedType.SUNFLOWER,
        SeedType.WALL_NUT,
        SeedType.REPEATER,
        SeedType.SNOW_PEA,
        SeedType.POTATO_MINE,
        SeedType.CHERRY_BOMB,
    ),
    # These three challenge plants are the actual flower-picture objective.
    40: (
        SeedType.STARFRUIT,
        SeedType.UMBRELLA_LEAF,
        SeedType.WALL_NUT,
        SeedType.SUNFLOWER,
        SeedType.POTATO_MINE,
        SeedType.CHERRY_BOMB,
    ),
    41: (
        SeedType.LILY_PAD,
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.CACTUS,
        SeedType.BLOVER,
        SeedType.FUME_SHROOM,
    ),
    44: (
        SeedType.FLOWER_POT,
        SeedType.SUNFLOWER,
        SeedType.TALL_NUT,
        SeedType.MELON_PULT,
        SeedType.SQUASH,
        SeedType.JALAPENO,
    ),
    45: (
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.FUME_SHROOM,
        SeedType.GRAVE_BUSTER,
        SeedType.DOOM_SHROOM,
        SeedType.ICE_SHROOM,
    ),
}

SURVIVAL_REPICK_SEED_PRIORITY: dict[int, tuple[SeedType, ...]] = {
    # The first Day stage establishes economy and low-cost lane coverage.
    # Later chooser screens retain that lawn, so spend the six packets on
    # stronger reinforcement, crowd control, and renewed barriers.
    1: (
        SeedType.MELON_PULT,
        SeedType.SNOW_PEA,
        SeedType.TALL_NUT,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
        SeedType.SQUASH,
    ),
    # Retain the free mushroom screen and established Sun-shrooms from the
    # opening, then add splash damage and durable lane control for later flags.
    2: (
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.FUME_SHROOM,
        SeedType.GLOOM_SHROOM,
        SeedType.PUMPKIN,
        SeedType.MELON_PULT,
        SeedType.CHERRY_BOMB,
    ),
    # The affordable Pool opening carries Peashooters and Sunflowers forward.
    # Repicks can therefore spend all six packets on durable reinforcement and
    # immediate counters for Dolphin, Buckethead, and dense flag waves.
    3: (
        SeedType.LILY_PAD,
        SeedType.SNOW_PEA,
        SeedType.TALL_NUT,
        SeedType.TANGLE_KELP,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
    ),
    # Fog's retained economy is steadily eaten during the later flags. Keep
    # Sun-shroom available at every repick so the land lanes can recover,
    # while free Puff-shrooms preserve sun. On a completed seven-packet
    # profile, Cattail adds all-lane coverage, Pumpkin protects the costly
    # upgrade, Wall-nut holds land, and Jalapeno clears Football pressure.
    4: (
        SeedType.LILY_PAD,
        SeedType.SUN_SHROOM,
        SeedType.PUFF_SHROOM,
        SeedType.CATTAIL,
        SeedType.PUMPKIN,
        SeedType.WALL_NUT,
        SeedType.JALAPENO,
    ),
    # Roof stages retain the Flower Pots, Sunflowers, and pults established
    # by the opening chooser.  Keep one Pot packet for replacement tiles,
    # then use the other five slots for durable damage and counters to the
    # Buckethead/Pogo/Catapult pressure introduced by later stages.
    5: (
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
        SeedType.GARLIC,
    ),
}

SURVIVAL_FINAL_REPICK_SEED_PRIORITY: dict[int, tuple[SeedType, ...]] = {
    # The fourth Night stage can leave only a few retained plants and active
    # zombies at the chooser boundary. Reuse its surviving Sun-shrooms while
    # spending every packet on immediate coverage, delay, and independent
    # cooldowns instead of attempting a fresh economy grid.
    2: (
        SeedType.PUFF_SHROOM,
        SeedType.FUME_SHROOM,
        SeedType.GLOOM_SHROOM,
        SeedType.PUMPKIN,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
        SeedType.ICE_SHROOM,
    ),
    # Four stages have already established persistent Sun-shrooms. Spend the
    # final chooser slot on an independent Cherry cooldown for the stacked
    # Football Zombies in the last flag wave.
    4: (
        SeedType.LILY_PAD,
        SeedType.PUFF_SHROOM,
        SeedType.CATTAIL,
        SeedType.PUMPKIN,
        SeedType.WALL_NUT,
        SeedType.JALAPENO,
        SeedType.CHERRY_BOMB,
    ),
    # The first three Roof repicks build an affordable recovery formation.
    # Switch to Melon splash damage for the densest Buckethead, Screen Door,
    # and Football waves.
    5: (
        SeedType.FLOWER_POT,
        SeedType.MELON_PULT,
        SeedType.TALL_NUT,
        SeedType.CHERRY_BOMB,
        SeedType.JALAPENO,
        SeedType.SQUASH,
        SeedType.PUMPKIN,
        SeedType.SUNFLOWER,
        SeedType.KERNEL_PULT,
        SeedType.CABBAGE_PULT,
        SeedType.WINTER_MELON,
        SeedType.UMBRELLA_LEAF,
        SeedType.GARLIC,
    ),
}

ART_CHALLENGE_TARGETS: dict[int, dict[tuple[int, int], SeedType]] = {
    22: {
        (0, 3): SeedType.STARFRUIT,
        (1, 3): SeedType.STARFRUIT,
        (1, 4): SeedType.STARFRUIT,
        (2, 1): SeedType.STARFRUIT,
        (2, 2): SeedType.STARFRUIT,
        (2, 3): SeedType.STARFRUIT,
        (2, 4): SeedType.STARFRUIT,
        (2, 5): SeedType.STARFRUIT,
        (2, 6): SeedType.STARFRUIT,
        (3, 3): SeedType.STARFRUIT,
        (3, 4): SeedType.STARFRUIT,
        (3, 5): SeedType.STARFRUIT,
        (4, 3): SeedType.STARFRUIT,
        (4, 6): SeedType.STARFRUIT,
    },
    36: {
        (0, 4): SeedType.WALL_NUT,
        (0, 5): SeedType.WALL_NUT,
        (0, 6): SeedType.WALL_NUT,
        (1, 3): SeedType.WALL_NUT,
        (1, 7): SeedType.WALL_NUT,
        (2, 3): SeedType.WALL_NUT,
        (2, 7): SeedType.WALL_NUT,
        (3, 3): SeedType.WALL_NUT,
        (3, 7): SeedType.WALL_NUT,
        (4, 4): SeedType.WALL_NUT,
        (4, 5): SeedType.WALL_NUT,
        (4, 6): SeedType.WALL_NUT,
    },
    40: {
        (0, 2): SeedType.STARFRUIT,
        (0, 3): SeedType.STARFRUIT,
        (0, 4): SeedType.STARFRUIT,
        (1, 1): SeedType.STARFRUIT,
        (1, 2): SeedType.WALL_NUT,
        (1, 3): SeedType.WALL_NUT,
        (1, 4): SeedType.WALL_NUT,
        (1, 5): SeedType.STARFRUIT,
        (2, 2): SeedType.STARFRUIT,
        (2, 3): SeedType.STARFRUIT,
        (2, 4): SeedType.STARFRUIT,
        (3, 3): SeedType.UMBRELLA_LEAF,
        (4, 2): SeedType.UMBRELLA_LEAF,
        (4, 3): SeedType.UMBRELLA_LEAF,
        (4, 4): SeedType.UMBRELLA_LEAF,
    },
}


def recommended_mode_seed_types(
    payload: dict[str, Any],
    *,
    survival_repick: bool = False,
) -> tuple[int, ...]:
    """Choose a human-legal loadout specialized for a secondary GOTY mode."""

    capacity = int(payload.get("seed_bank_capacity", 0))
    available = {
        int(seed_type) for seed_type in payload.get("available_seed_types", [])
    }
    mode = int(payload.get("game_mode", 0))
    if mode != 0:
        # The isolated bridge temporarily unlocks the complete original
        # seed set while constructing a secondary challenge, then restores
        # its throwaway profile level before emitting this observation.
        # Consequently ``available_seed_types`` reflects that profile (often
        # only Peashooter), not the chooser that built the current board.
        available.update(range(int(SeedType.MELON_PULT) + 1))
    recommendation_payload = {
        **payload,
        "available_seed_types": sorted(available),
    }
    if survival_repick:
        final_repick_stage = 4 if mode == 2 else 3
        priority = (
            SURVIVAL_FINAL_REPICK_SEED_PRIORITY.get(mode, ())
            if int(payload.get("survival_stage", 0)) >= final_repick_stage
            else ()
        ) or SURVIVAL_REPICK_SEED_PRIORITY.get(mode, ())
    else:
        priority = MODE_SEED_PRIORITY.get(mode, ())
    selected = [int(seed_type) for seed_type in priority if int(seed_type) in available]
    selected.extend(
        seed_type
        for seed_type in recommended_seed_types(recommendation_payload)
        if seed_type not in selected
    )
    selected.extend(
        int(packet["type"])
        for packet in payload.get("seed_packets", [])
        if int(packet["type"]) in available and int(packet["type"]) not in selected
    )
    selected.extend(sorted(available - set(selected)))
    return tuple(selected[:capacity])


def legal_plants(
    payload: dict[str, Any], codec: PortableActionCodec
) -> tuple[LegalPlant, ...]:
    packet_types = {
        int(packet["slot"]): int(packet["type"])
        for packet in payload.get("seed_packets", [])
    }
    plants: list[LegalPlant] = []
    for action in payload.get("legal_actions", []):
        action_id = int(action)
        if not codec.plant_begin <= action_id < codec.shovel_begin:
            continue
        decoded = codec.decode(action_id)
        seed_type = packet_types.get(decoded.slot)
        if seed_type is None:
            continue
        plants.append(
            LegalPlant(
                action=action_id,
                slot=decoded.slot,
                row=decoded.row,
                col=decoded.col,
                seed_type=seed_type,
            )
        )
    return tuple(plants)


def _plants_by_cell(payload: dict[str, Any]) -> dict[tuple[int, int], list[int]]:
    cells: dict[tuple[int, int], list[int]] = {}
    for plant in payload.get("plants", []):
        cell = (int(plant["row"]), int(plant["col"]))
        cells.setdefault(cell, []).append(int(plant["type"]))
    return cells


def _lane_zombies(payload: dict[str, Any]) -> dict[int, list[float]]:
    lanes: dict[int, list[float]] = {}
    for zombie in payload.get("zombies", []):
        if bool(zombie.get("mind_controlled", False)):
            continue
        lanes.setdefault(int(zombie["row"]), []).append(float(zombie["x"]))
    return lanes


def _shooter_counts(
    cells: dict[tuple[int, int], list[int]], rows: tuple[int, ...]
) -> dict[int, int]:
    counts = dict.fromkeys(rows, 0)
    shooter_type_values = {int(seed_type) for seed_type in SHOOTER_TYPES}
    for (plant_row, _), seed_types in cells.items():
        for seed_type in seed_types:
            if seed_type not in shooter_type_values:
                continue
            if seed_type == int(SeedType.CATTAIL):
                affected_rows = rows
            elif seed_type == int(SeedType.THREEPEATER):
                affected_rows = (plant_row - 1, plant_row, plant_row + 1)
            else:
                affected_rows = (plant_row,)
            for row in affected_rows:
                if row in counts:
                    counts[row] += 1
    return counts


def _pick(
    plants: tuple[LegalPlant, ...],
    *,
    seed_types: set[SeedType],
    rows: tuple[int, ...] | None = None,
    columns: tuple[int, ...] | None = None,
    reverse_type_priority: bool = False,
) -> int | None:
    choices = [plant for plant in plants if plant.seed_type in seed_types]
    if rows is not None:
        row_rank = {row: index for index, row in enumerate(rows)}
        choices = [plant for plant in choices if plant.row in row_rank]
    else:
        row_rank = {}
    if columns is not None:
        col_rank = {col: index for index, col in enumerate(columns)}
        choices = [plant for plant in choices if plant.col in col_rank]
    else:
        col_rank = {}
    if not choices:
        return None

    def key(plant: LegalPlant) -> tuple[int, int, int, int]:
        type_score = SHOOTER_PRIORITY.get(SeedType(plant.seed_type), 0)
        if not reverse_type_priority:
            type_score = -type_score
        return (
            row_rank.get(plant.row, plant.row),
            col_rank.get(plant.col, plant.col),
            type_score,
            plant.action,
        )

    return min(choices, key=key).action


class NativeBoardDriver:
    """Deterministic verifier policy using only bridge-visible player state.

    This is not an alternate game mechanic or an RL policy. It drives the
    ordinary native mouse-action surface to prove that a board can reach its
    authentic terminal state.
    """

    def __init__(self) -> None:
        self.codec = PortableActionCodec()
        self._covered_pool_rows: set[int] = set()
        self._air_reserve_active = False
        self._vase_seed_row: int | None = None
        self._vase_seed_type: int | None = None
        self._vase_path_control_active = False
        self._vase_house_side_blockers = False
        self._blover_response_pending = False
        self._i_zombie_front_jump_counts: dict[int, int] = {}
        self._i_zombie_digger_rows: set[int] = set()
        self._i_zombie_nine_bungee_targets: set[tuple[int, int]] = set()
        self._i_zombie_nine_ladders_sent = 0
        self._i_zombie_nine_digger_sent = False

    def choose_action(self, payload: dict[str, Any]) -> int:
        legal = {int(action) for action in payload.get("legal_actions", [])}
        if self.codec.collect_action in legal:
            return self.codec.collect_action

        click = self._board_click(payload, legal)
        if click is not None:
            return click

        plants = legal_plants(payload, self.codec)
        if not plants:
            return 0
        if 61 <= int(payload.get("game_mode", 0)) <= 70:
            return self._i_zombie_action(payload, plants)
        art_action = self._art_challenge_action(payload, plants)
        if art_action is not None:
            return art_action
        if bool(payload.get("has_conveyor", False)):
            return self._conveyor_action(payload, plants)
        return self._chosen_seed_action(payload, plants)

    def choose_interaction(self, payload: dict[str, Any]) -> str | None:
        """Select a native non-board UI control when board setup is complete."""

        interactions = {
            str(interaction)
            for interaction in payload.get("available_interactions", [])
        }
        if "survival_repick" in interactions:
            return "survival_repick"
        if "last_stand_start" not in interactions:
            return None
        if self.choose_action(payload) != 0:
            return None
        return "last_stand_start"

    def _i_zombie_action(
        self, payload: dict[str, Any], plants: tuple[LegalPlant, ...]
    ) -> int:
        brain_rows = {
            int(item["row"])
            for item in payload.get("grid_items", [])
            if int(item.get("type", -1)) == 12
        }
        if not brain_rows:
            return 0

        lane_plants = {
            row: [
                plant for plant in payload.get("plants", []) if int(plant["row"]) == row
            ]
            for row in brain_rows
        }
        mode = int(payload.get("game_mode", 0))
        if mode == 69:
            return self._i_zombie_nine_action(
                payload,
                plants,
                brain_rows=brain_rows,
                lane_plants=lane_plants,
            )

        def lane_key(row: int) -> tuple[int, int, int, int]:
            threat = sum(
                I_ZOMBIE_PLANT_THREAT.get(
                    SeedType(int(plant["type"])),
                    0,
                )
                for plant in lane_plants[row]
            )
            sunflowers = sum(
                int(plant["type"]) == int(SeedType.SUNFLOWER)
                for plant in lane_plants[row]
            )
            return threat, -sunflowers, len(lane_plants[row]), row

        active_zombies = [
            zombie
            for zombie in payload.get("zombies", [])
            if not bool(zombie.get("mind_controlled", False))
            and int(zombie["row"]) in brain_rows
        ]
        if active_zombies:
            active_rows = sorted(
                {int(zombie["row"]) for zombie in active_zombies},
                key=lane_key,
            )
            target_row = active_rows[0]
            active_in_target = sum(
                int(zombie["row"]) == target_row for zombie in active_zombies
            )
            target_has_potato_mine = any(
                int(plant["type"]) == int(SeedType.POTATO_MINE)
                for plant in lane_plants[target_row]
            )
            active_limit = 2 if mode == 65 else 1
            if (
                active_in_target >= active_limit
                or lane_key(target_row)[0] == 0
                or target_has_potato_mine
                or target_row in self._i_zombie_digger_rows
            ):
                return 0
        else:
            target_row = min(brain_rows, key=lane_key)
        if (
            0 < len(lane_plants[target_row]) <= 2
            and max(int(plant["col"]) for plant in lane_plants[target_row]) <= 1
            and target_row not in self._i_zombie_digger_rows
        ):
            diggers = [
                plant
                for plant in plants
                if plant.row == target_row
                and plant.seed_type == int(SeedType.ZOMBIE_DIGGER)
            ]
            if diggers:
                self._i_zombie_digger_rows.add(target_row)
                return min(diggers, key=lambda plant: (plant.col, plant.slot)).action
        choices = [
            plant
            for plant in plants
            if plant.row == target_row
            and plant.seed_type
            not in {
                int(SeedType.ZOMBIE_DIGGER),
                int(SeedType.ZOMBIE_BUNGEE),
                int(SeedType.ZOMBIE_BALLOON),
            }
        ]
        if not choices:
            return 0
        front_plant = max(
            lane_plants[target_row],
            key=lambda plant: int(plant["col"]),
            default=None,
        )
        if (
            front_plant is not None
            and int(front_plant["type"])
            in {
                int(SeedType.TALL_NUT),
                int(SeedType.TORCHWOOD),
                int(SeedType.MAGNET_SHROOM),
            }
            and self._i_zombie_front_jump_counts.get(target_row, 0) < 2
        ):
            pole_vaulters = [
                plant
                for plant in choices
                if plant.seed_type == int(SeedType.ZOMBIE_POLE_VAULTER)
            ]
            if pole_vaulters:
                self._i_zombie_front_jump_counts[target_row] = (
                    self._i_zombie_front_jump_counts.get(target_row, 0) + 1
                )
                return min(
                    pole_vaulters, key=lambda plant: (plant.col, plant.slot)
                ).action
        dancer_zombies = [
            plant for plant in choices if plant.seed_type == int(SeedType.ZOMBIE_DANCER)
        ]
        nearby_chompers = sum(
            int(plant["type"]) == int(SeedType.CHOMPER)
            and abs(int(plant["row"]) - target_row) <= 1
            for plant in payload.get("plants", [])
        )
        if dancer_zombies and nearby_chompers >= 2:
            return min(dancer_zombies, key=lambda plant: (plant.col, plant.slot)).action
        if any(
            int(plant["type"]) == int(SeedType.POTATO_MINE)
            for plant in lane_plants[target_row]
        ):
            normal_zombies = [
                plant
                for plant in choices
                if plant.seed_type == int(SeedType.ZOMBIE_NORMAL)
            ]
            if normal_zombies:
                return min(
                    normal_zombies, key=lambda plant: (plant.col, plant.slot)
                ).action
        if any(
            int(plant["type"]) == int(SeedType.GARLIC)
            for plant in lane_plants[target_row]
        ):
            pole_vaulters = [
                plant
                for plant in choices
                if plant.seed_type == int(SeedType.ZOMBIE_POLE_VAULTER)
            ]
            if pole_vaulters:
                return min(
                    pole_vaulters, key=lambda plant: (plant.col, plant.slot)
                ).action
        target_threat = lane_key(target_row)[0]
        required_strength = (
            10 if target_threat == 0 else 60 if target_threat <= 10 else 90
        )
        sufficient_choices = [
            plant
            for plant in choices
            if I_ZOMBIE_STRENGTH.get(SeedType(plant.seed_type), -1) >= required_strength
        ]
        ranked_choices = sufficient_choices or choices
        return min(
            ranked_choices,
            key=lambda plant: (
                (
                    I_ZOMBIE_COST.get(SeedType(plant.seed_type), 10_000)
                    if sufficient_choices
                    else -I_ZOMBIE_STRENGTH.get(SeedType(plant.seed_type), -1)
                ),
                -I_ZOMBIE_STRENGTH.get(SeedType(plant.seed_type), -1),
                plant.col,
                plant.slot,
            ),
        ).action

    def _i_zombie_nine_action(
        self,
        payload: dict[str, Any],
        plants: tuple[LegalPlant, ...],
        *,
        brain_rows: set[int],
        lane_plants: dict[int, list[dict[str, Any]]],
    ) -> int:
        """Play the fixed GOTY level-nine puzzle through its visible phases."""

        def deploy(
            seed_type: SeedType,
            row: int,
            *,
            col: int | None = None,
        ) -> int | None:
            choices = [
                plant
                for plant in plants
                if plant.seed_type == int(seed_type)
                and plant.row == row
                and (col is None or plant.col == col)
            ]
            if not choices:
                return None
            return min(choices, key=lambda plant: (plant.col, plant.slot)).action

        active_zombies = [
            zombie
            for zombie in payload.get("zombies", [])
            if not bool(zombie.get("mind_controlled", False))
        ]

        # The guaranteed solution first steals both Sunflowers above the four
        # Potato Mines. Each target is committed once while its Bungee animates.
        top_sunflowers = [
            plant
            for plant in lane_plants.get(0, [])
            if int(plant["type"]) == int(SeedType.SUNFLOWER)
        ]
        if top_sunflowers:
            target = next(
                (
                    plant
                    for plant in sorted(
                        top_sunflowers, key=lambda item: int(item["col"])
                    )
                    if (0, int(plant["col"])) not in self._i_zombie_nine_bungee_targets
                ),
                None,
            )
            if target is None:
                return 0
            col = int(target["col"])
            action = deploy(SeedType.ZOMBIE_BUNGEE, 0, col=col)
            if action is not None:
                self._i_zombie_nine_bungee_targets.add((0, col))
                return action
            return 0

        # Two Ladder Zombies cooperate on the Tall-nut/Snow Pea lane. They are
        # intentionally launched together; one ladders the nut and the other
        # retains enough protection to finish the shooters behind it.
        if 1 in brain_rows and lane_plants.get(1):
            if self._i_zombie_nine_ladders_sent < 2:
                action = deploy(SeedType.ZOMBIE_LADDER, 1)
                if action is not None:
                    self._i_zombie_nine_ladders_sent += 1
                    return action
            return 0
        if active_zombies:
            return 0

        magnet = next(
            (
                plant
                for plant in payload.get("plants", [])
                if int(plant["type"]) == int(SeedType.MAGNET_SHROOM)
            ),
            None,
        )
        if magnet is not None:
            target = (int(magnet["row"]), int(magnet["col"]))
            if target in self._i_zombie_nine_bungee_targets:
                return 0
            action = deploy(
                SeedType.ZOMBIE_BUNGEE,
                target[0],
                col=target[1],
            )
            if action is not None:
                self._i_zombie_nine_bungee_targets.add(target)
                return action
            return 0

        # With the Magnet-shroom gone, one Football can harvest the bottom lane.
        if 4 in brain_rows and lane_plants.get(4):
            return deploy(SeedType.ZOMBIE_FOOTBALL, 4) or 0
        if 1 in brain_rows:
            return deploy(SeedType.ZOMBIE_IMP, 1) or 0

        # Removing row 2's Threepeater makes the Peashooter lane independent;
        # a Digger clears it from behind and an Imp claims the exposed brain.
        if 3 in brain_rows and lane_plants.get(3):
            if not self._i_zombie_nine_digger_sent:
                action = deploy(SeedType.ZOMBIE_DIGGER, 3)
                if action is not None:
                    self._i_zombie_nine_digger_sent = True
                    return action
            return 0
        if 3 in brain_rows:
            return deploy(SeedType.ZOMBIE_IMP, 3) or 0

        # Imps are deliberately single-file against Chompers, Squash, and
        # Potato Mines so one instant-kill plant cannot consume a stack.
        for row in (2, 0):
            if row in brain_rows:
                return deploy(SeedType.ZOMBIE_IMP, row) or 0
        return 0

    @staticmethod
    def _art_challenge_action(
        payload: dict[str, Any], plants: tuple[LegalPlant, ...]
    ) -> int | None:
        targets = ART_CHALLENGE_TARGETS.get(int(payload.get("game_mode", 0)))
        if targets is None:
            return None
        economy_count = sum(
            int(plant.get("type", -1)) in {int(seed) for seed in ECONOMY_TYPES}
            for plant in payload.get("plants", [])
        )
        if economy_count < 3:
            return None

        occupied = _plants_by_cell(payload)
        lanes = _lane_zombies(payload)
        if int(payload.get("game_mode", 0)) in {36, 40}:
            rows = tuple(range(int(payload.get("row_count", 5))))
            if min(_shooter_counts(occupied, rows).values(), default=0) < 1:
                return None
        target_actions = [
            plant
            for plant in plants
            if targets.get((plant.row, plant.col)) == plant.seed_type
            and plant.seed_type not in occupied.get((plant.row, plant.col), [])
        ]
        if not target_actions:
            return None
        seed_priority = {
            int(SeedType.STARFRUIT): 0,
            int(SeedType.WALL_NUT): 1,
            int(SeedType.UMBRELLA_LEAF): 2,
        }
        return min(
            target_actions,
            key=lambda plant: (
                min(lanes.get(plant.row, [float("inf")])),
                seed_priority.get(plant.seed_type, 3),
                -plant.col,
                plant.row,
            ),
        ).action

    def _board_click(self, payload: dict[str, Any], legal: set[int]) -> int | None:
        click_actions = {
            action
            for action in legal
            if self.codec.board_click_begin <= action < self.codec.collect_action
        }
        if not click_actions:
            return None
        self._vase_path_control_active |= any(
            int(item.get("type", -1)) == 7
            and int(item.get("seed_type", -1))
            in {int(seed_type) for seed_type in VASE_PATH_CONTROL_TYPES}
            for item in payload.get("grid_items", [])
        )
        self._vase_house_side_blockers |= any(
            int(item.get("type", -1)) == 7
            and int(item.get("seed_type", -1))
            in {int(SeedType.SPIKEWEED), int(SeedType.PUMPKIN)}
            for item in payload.get("grid_items", [])
        )
        usable_seed_coins = [
            coin
            for coin in payload.get("coins", [])
            if int(coin.get("type", -1)) == 16
            and bool(coin.get("being_collected", False))
        ]
        if usable_seed_coins:
            occupied = {
                (int(plant["row"]), int(plant["col"]))
                for plant in payload.get("plants", [])
            }
            occupied.update(
                (int(item["row"]), int(item["col"]))
                for item in payload.get("grid_items", [])
                if int(item.get("type", -1)) == 7
            )
            available_rows = sorted(
                {
                    (action - self.codec.board_click_begin) // self.codec.cols
                    for action in click_actions
                }
            )
            zombie_vase_counts = {
                row: sum(
                    int(item.get("type", -1)) == 7
                    and int(item.get("seed_type", -1)) < 0
                    and int(item.get("row", -1)) == row
                    for item in payload.get("grid_items", [])
                )
                for row in available_rows
            }
            leftmost_zombie_vase = {
                row: min(
                    (
                        int(item.get("col", self.codec.cols))
                        for item in payload.get("grid_items", [])
                        if int(item.get("type", -1)) == 7
                        and int(item.get("seed_type", -1)) < 0
                        and int(item.get("row", -1)) == row
                    ),
                    default=self.codec.cols,
                )
                for row in available_rows
            }
            preferred_rows = available_rows
            if self._vase_seed_type in {
                int(seed_type) for seed_type in VASE_OFFENSE_TYPES
            }:
                coverage_types = {
                    *(int(seed_type) for seed_type in VASE_OFFENSE_TYPES),
                    int(SeedType.SQUASH),
                }
                rightmost_zombie_vase = {
                    row: max(
                        (
                            int(item.get("col", -1))
                            for item in payload.get("grid_items", [])
                            if int(item.get("type", -1)) == 7
                            and int(item.get("seed_type", -1)) < 0
                            and int(item.get("row", -1)) == row
                        ),
                        default=-1,
                    )
                    for row in available_rows
                }
                coverage_counts = {
                    row: sum(
                        int(plant["type"]) in coverage_types
                        and int(plant["row"]) == row
                        for plant in payload.get("plants", [])
                    )
                    for row in available_rows
                }
                effective_coverage_counts = {
                    row: sum(
                        int(plant["type"]) in coverage_types
                        and int(plant["row"]) == row
                        and (
                            int(plant["type"]) != int(SeedType.LEFTPEATER)
                            or int(plant["col"]) > rightmost_zombie_vase[row]
                        )
                        for plant in payload.get("plants", [])
                    )
                    for row in available_rows
                }

                def leftpeater_columns(row: int) -> tuple[int, ...]:
                    rightmost_vase = rightmost_zombie_vase[row]
                    return tuple(
                        sorted(
                            range(self.codec.cols),
                            key=lambda col: (col <= rightmost_vase, -col),
                        )
                    )

                def threepeater_columns(row: int) -> tuple[int, ...]:
                    first_target_col = min(
                        (
                            leftmost_zombie_vase[lane]
                            for lane in available_rows
                            if abs(lane - row) <= 1 and zombie_vase_counts[lane] > 0
                        ),
                        default=self.codec.cols,
                    )
                    return tuple(
                        sorted(
                            range(self.codec.cols),
                            key=lambda col: (col >= first_target_col, -col),
                        )
                    )

                def has_effective_leftpeater_cell(row: int) -> bool:
                    return any(
                        col > rightmost_zombie_vase[row]
                        and (row, col) not in occupied
                        and self.codec.board_click(row, col) in click_actions
                        for col in range(self.codec.cols)
                    )

                def placement_rank(row: int) -> int:
                    if self._vase_seed_type != int(SeedType.LEFTPEATER):
                        return 0
                    return min(
                        (
                            rank
                            for rank, col in enumerate(leftpeater_columns(row))
                            if (row, col) not in occupied
                            and self.codec.board_click(row, col) in click_actions
                        ),
                        default=self.codec.cols,
                    )

                if self._vase_seed_type == int(SeedType.THREEPEATER):
                    lane_coverage_counts = {
                        lane: sum(
                            (
                                int(plant["type"]) == int(SeedType.THREEPEATER)
                                and abs(int(plant["row"]) - lane) <= 1
                                and int(plant["col"]) < leftmost_zombie_vase[lane]
                            )
                            or (
                                int(plant["type"]) in coverage_types
                                and int(plant["row"]) == lane
                                and (
                                    int(plant["type"]) != int(SeedType.LEFTPEATER)
                                    or int(plant["col"]) > rightmost_zombie_vase[lane]
                                )
                            )
                            for plant in payload.get("plants", [])
                        )
                        for lane in available_rows
                    }

                    def threepeater_value(row: int) -> float:
                        placement_col = next(
                            (
                                col
                                for col in threepeater_columns(row)
                                if (row, col) not in occupied
                                and self.codec.board_click(row, col) in click_actions
                            ),
                            self.codec.cols,
                        )
                        return sum(
                            zombie_vase_counts[lane] / (lane_coverage_counts[lane] + 1)
                            for lane in available_rows
                            if abs(lane - row) <= 1
                            and placement_col < leftmost_zombie_vase[lane]
                        )

                    preferred_rows = sorted(
                        available_rows,
                        key=lambda row: (
                            -threepeater_value(row),
                            row != self._vase_seed_row,
                            row,
                        ),
                    )
                else:
                    preferred_rows = sorted(
                        available_rows,
                        key=lambda row: (
                            zombie_vase_counts[row] == 0,
                            self._vase_seed_type == int(SeedType.LEFTPEATER)
                            and not has_effective_leftpeater_cell(row),
                            coverage_counts[row] >= 2,
                            coverage_counts[row] / max(zombie_vase_counts[row], 1),
                            effective_coverage_counts[row],
                            coverage_counts[row],
                            placement_rank(row),
                            row != self._vase_seed_row,
                            row,
                        ),
                    )
            elif self._vase_seed_type == int(SeedType.SQUASH):
                lanes = _lane_zombies(payload)
                defense_types = {
                    *(int(seed_type) for seed_type in SHOOTER_TYPES),
                    int(SeedType.SQUASH),
                }
                coverage_counts = {
                    row: sum(
                        int(plant["type"]) in defense_types and int(plant["row"]) == row
                        for plant in payload.get("plants", [])
                    )
                    for row in available_rows
                }
                squash_counts = {
                    row: sum(
                        int(plant["type"]) == int(SeedType.SQUASH)
                        and int(plant["row"]) == row
                        for plant in payload.get("plants", [])
                    )
                    for row in available_rows
                }
                dancer_vase_counts = {
                    row: sum(
                        int(item.get("type", -1)) == 7
                        and int(item.get("seed_type", -1)) < 0
                        and int(item.get("zombie_type", -1)) == 8
                        and int(item.get("row", -1)) == row
                        for item in payload.get("grid_items", [])
                    )
                    for row in available_rows
                }
                threatened_rows = [
                    row
                    for row, _ in sorted(
                        lanes.items(),
                        key=lambda item: (
                            squash_counts.get(item[0], 0) >= 1,
                            coverage_counts.get(item[0], 0),
                            min(item[1]),
                            item[0],
                        ),
                    )
                    if row in available_rows
                ]
                preferred_rows = threatened_rows + [
                    row
                    for row in sorted(
                        available_rows,
                        key=lambda row: (
                            zombie_vase_counts[row] == 0,
                            squash_counts[row] >= 1,
                            dancer_vase_counts[row] == 0,
                            coverage_counts[row] / max(zombie_vase_counts[row], 1),
                            coverage_counts[row],
                            row,
                        ),
                    )
                    if row not in threatened_rows
                ]
            elif self._vase_seed_type in {
                int(SeedType.WALL_NUT),
                int(SeedType.POTATO_MINE),
                int(SeedType.HYPNO_SHROOM),
            }:
                defense_types = {
                    *(int(seed_type) for seed_type in SHOOTER_TYPES),
                    int(SeedType.SQUASH),
                    int(SeedType.WALL_NUT),
                    int(SeedType.POTATO_MINE),
                    int(SeedType.HYPNO_SHROOM),
                }
                same_type_counts = {
                    row: sum(
                        int(plant["type"]) == self._vase_seed_type
                        and int(plant["row"]) == row
                        for plant in payload.get("plants", [])
                    )
                    + sum(
                        int(item.get("type", -1)) == 7
                        and int(item.get("seed_type", -1)) == self._vase_seed_type
                        and int(item.get("row", -1)) == row
                        for item in payload.get("grid_items", [])
                    )
                    for row in available_rows
                }
                defense_counts = {
                    row: sum(
                        int(plant["type"]) in defense_types and int(plant["row"]) == row
                        for plant in payload.get("plants", [])
                    )
                    for row in available_rows
                }
                preferred_rows = sorted(
                    available_rows,
                    key=lambda row: (
                        zombie_vase_counts[row] == 0,
                        same_type_counts[row],
                        row != self._vase_seed_row,
                        defense_counts[row] / max(zombie_vase_counts[row], 1),
                        defense_counts[row],
                        row,
                    ),
                )
            elif self._vase_seed_row in preferred_rows:
                preferred_rows.remove(self._vase_seed_row)
                preferred_rows.insert(0, self._vase_seed_row)
            for row in preferred_rows:
                if self._vase_seed_type == int(SeedType.SQUASH):
                    preferred_cols = (5, 4, 3, 2, 1, 0, 6, 7, 8)
                elif self._vase_seed_type == int(SeedType.LEFTPEATER):
                    preferred_cols = leftpeater_columns(row)
                elif self._vase_seed_type == int(SeedType.THREEPEATER):
                    preferred_cols = threepeater_columns(row)
                elif self._vase_seed_type == int(SeedType.TALL_NUT):
                    first_target_col = leftmost_zombie_vase[row]
                    preferred_cols = tuple(
                        sorted(
                            range(self.codec.cols),
                            key=lambda col: (col >= first_target_col, -col),
                        )
                    )
                elif (
                    self._vase_seed_type
                    in {int(seed_type) for seed_type in DEFENSE_TYPES}
                    and self._vase_house_side_blockers
                ):
                    preferred_cols = (0, 1, 2, 3, 4, 5, 6, 7, 8)
                else:
                    preferred_cols = (4, 3, 2, 1, 0, 5, 6, 7, 8)
                for col in preferred_cols:
                    if (row, col) in occupied:
                        continue
                    action = self.codec.board_click(row, col)
                    if action in click_actions:
                        self._vase_seed_row = None
                        self._vase_seed_type = None
                        return action
        pots = [
            item
            for item in payload.get("grid_items", [])
            if int(item.get("type", -1)) == 7
        ]
        active_zombies = [
            zombie
            for zombie in payload.get("zombies", [])
            if not bool(zombie.get("mind_controlled", False))
        ]
        if pots:
            if active_zombies:
                return None
            pot = min(
                pots,
                key=lambda item: (
                    int(item.get("seed_type", -1)) < 0,
                    (
                        False
                        if int(item.get("seed_type", -1)) >= 0
                        else int(item.get("zombie_type", 0)) == 15
                    ),
                    (
                        0
                        if int(item.get("seed_type", -1)) >= 0
                        else (
                            0
                            if int(item.get("zombie_type", 0)) == 4
                            else 1
                            if int(item.get("zombie_type", 0)) not in {0, 15}
                            else 2
                        )
                    ),
                    (
                        (
                            VASE_SEED_OPEN_PRIORITY.get(
                                int(item.get("seed_type", -1)),
                                2,
                            )
                            if self._vase_path_control_active
                            else 0
                        )
                        if int(item.get("seed_type", -1)) >= 0
                        else 0
                    ),
                    (
                        -int(item["col"])
                        if int(item.get("seed_type", -1)) >= 0
                        else int(item["col"])
                    ),
                    -int(item.get("zombie_type", -1)),
                    int(item["row"]),
                ),
            )
            action = self.codec.board_click(int(pot["row"]), int(pot["col"]))
            if action in click_actions:
                if int(pot.get("seed_type", -1)) >= 0:
                    self._vase_seed_row = int(pot["row"])
                    self._vase_seed_type = int(pot["seed_type"])
                return action
        zombies = [
            zombie
            for zombie in payload.get("zombies", [])
            if not bool(zombie.get("mind_controlled", False))
        ]
        is_vasebreaker = (
            int(payload.get("level", 0)) == 35
            or 51 <= int(payload.get("game_mode", 0)) <= 60
        )
        if zombies and is_vasebreaker:
            return None
        if zombies:
            zombie = min(zombies, key=lambda item: float(item["x"]))
            row = max(0, min(self.codec.rows - 1, int(zombie["row"])))
            approximate_col = max(
                0,
                min(
                    self.codec.cols - 1,
                    int((float(zombie["x"]) - 40.0) // 80.0),
                ),
            )
            by_distance = sorted(
                range(self.codec.cols),
                key=lambda col: (abs(col - approximate_col), col),
            )
            for col in by_distance:
                action = self.codec.board_click(row, col)
                if action in click_actions:
                    return action
        return min(click_actions)

    def _conveyor_action(
        self, payload: dict[str, Any], plants: tuple[LegalPlant, ...]
    ) -> int:
        lanes = _lane_zombies(payload)
        if int(payload.get("level", 0)) == 40 or int(payload.get("game_mode", 0)) == 47:
            balloon_rows = tuple(
                int(zombie["row"])
                for zombie in sorted(
                    payload.get("zombies", []),
                    key=lambda zombie: (
                        float(zombie["x"]),
                        int(zombie["row"]),
                    ),
                )
                if int(zombie.get("type", -1)) == 16
                and not bool(zombie.get("mind_controlled", False))
            )
            if balloon_rows:
                if self._blover_response_pending:
                    return 0
                blover = _pick(
                    plants,
                    seed_types={SeedType.BLOVER},
                    columns=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                )
                if blover is not None:
                    self._blover_response_pending = True
                    return blover
                cactus = _pick(
                    plants,
                    seed_types={SeedType.CACTUS},
                    rows=balloon_rows,
                    columns=(2, 3, 4, 1, 5, 0, 6, 7, 8),
                )
                if cactus is not None:
                    return cactus
            else:
                self._blover_response_pending = False
        threatened_rows = tuple(
            row
            for row, _ in sorted(
                lanes.items(),
                key=lambda item: (min(item[1]), item[0]),
            )
        )
        quiet_rows = tuple(
            row for row in range(int(payload.get("row_count", 5))) if row not in lanes
        )
        row_order = threatened_rows + quiet_rows
        if int(payload.get("level", 0)) == 5 or int(payload.get("game_mode", 0)) in {
            17,
            33,
        }:
            exploding_nut = _pick(
                plants,
                seed_types={SeedType.EXPLODE_O_NUT},
                rows=row_order,
                columns=(0, 1, 2, 3, 4, 5, 6, 7, 8),
            )
            if exploding_nut is not None:
                return exploding_nut
            giant_nut = _pick(
                plants,
                seed_types={SeedType.GIANT_WALL_NUT},
                rows=(2, 1, 3, 0, 4),
                columns=(0, 1, 2, 3, 4, 5, 6, 7, 8),
            )
            if giant_nut is not None:
                return giant_nut
            wall_nut = _pick(
                plants,
                seed_types={SeedType.WALL_NUT},
                rows=row_order,
                columns=(0, 1, 2, 3, 4, 5, 6, 7, 8),
            )
            if wall_nut is not None:
                return wall_nut
            return 0
        is_miniature_pool_conveyor = (
            int(payload.get("level", 0)) == 25 or int(payload.get("game_mode", 0)) == 25
        ) and bool(payload.get("stage_has_pool", False))
        if is_miniature_pool_conveyor:
            defense_row_order = tuple(row for row in row_order if row not in (2, 3))
            defense_row_order += tuple(row for row in row_order if row in (2, 3))
        else:
            defense_row_order = row_order

        if int(payload.get("level", 0)) == 50 or int(payload.get("game_mode", 0)) == 35:
            cells = _plants_by_cell(payload)
            rows = tuple(range(int(payload.get("row_count", 5))))
            boss_zombie = next(
                (
                    zombie
                    for zombie in payload.get("zombies", [])
                    if int(zombie.get("type", -1)) == 25
                    and not bool(zombie.get("mind_controlled", False))
                ),
                None,
            )
            boss_lanes: dict[int, list[float]] = {}
            for zombie in payload.get("zombies", []):
                if bool(zombie.get("mind_controlled", False)):
                    continue
                if int(zombie.get("type", -1)) == 25:
                    continue
                boss_lanes.setdefault(int(zombie["row"]), []).append(float(zombie["x"]))
            boss_threat_types: dict[int, list[int]] = {}
            for zombie in payload.get("zombies", []):
                if bool(zombie.get("mind_controlled", False)):
                    continue
                if int(zombie.get("type", -1)) == 25:
                    continue
                boss_threat_types.setdefault(int(zombie["row"]), []).append(
                    int(zombie.get("type", -1))
                )
            boss_threat_rows = tuple(
                row
                for row, _ in sorted(
                    boss_lanes.items(),
                    key=lambda item: (
                        not any(
                            zombie_type in {3, 4, 7, 18, 21, 22, 23, 24}
                            for zombie_type in boss_threat_types.get(item[0], [])
                        ),
                        min(item[1]),
                        item[0],
                    ),
                )
            )
            if boss_zombie is not None and bool(
                boss_zombie.get("boss_ball_active", False)
            ):
                ball_row = int(boss_zombie.get("fireball_row", -1))
                if bool(boss_zombie.get("is_fireball", False)):
                    ice_shroom = _pick(
                        plants,
                        seed_types={SeedType.ICE_SHROOM},
                        rows=rows,
                        columns=(6, 5, 7, 4, 3, 2, 1, 0),
                    )
                    if ice_shroom is not None:
                        return ice_shroom
                    ball_counter_rows = rows
                else:
                    jalapeno = _pick(
                        plants,
                        seed_types={SeedType.JALAPENO},
                        rows=(ball_row,) if ball_row in rows else rows,
                        columns=(6, 5, 7, 4, 3, 2, 1, 0),
                    )
                    if jalapeno is not None:
                        return jalapeno
                    ball_counter_rows = (ball_row,) if ball_row in rows else rows
                ball_counter_support = self._support_action(
                    plants,
                    cells,
                    support_type=SeedType.FLOWER_POT,
                    rows=ball_counter_rows,
                    columns=(6, 5, 7, 4, 3),
                )
                if ball_counter_support is not None:
                    return ball_counter_support
            jalapeno_count = len(
                {
                    plant.slot
                    for plant in plants
                    if plant.seed_type == int(SeedType.JALAPENO)
                }
            )
            closest_boss_threat = min(
                (min(xs) for xs in boss_lanes.values()),
                default=float("inf"),
            )
            if (
                jalapeno_count >= 2 and closest_boss_threat < 300.0
            ) or closest_boss_threat < 150.0:
                emergency_jalapeno = _pick(
                    plants,
                    seed_types={SeedType.JALAPENO},
                    rows=boss_threat_rows,
                    columns=(6, 5, 7, 4, 3, 2, 1, 0),
                )
                if emergency_jalapeno is not None:
                    return emergency_jalapeno
            ice_shroom_count = len(
                {
                    plant.slot
                    for plant in plants
                    if plant.seed_type == int(SeedType.ICE_SHROOM)
                }
            )
            if (
                ice_shroom_count >= 2 and closest_boss_threat < 300.0
            ) or closest_boss_threat < 100.0:
                emergency_ice_shroom = _pick(
                    plants,
                    seed_types={SeedType.ICE_SHROOM},
                    rows=rows,
                    columns=(6, 5, 7, 4, 3, 2, 1, 0),
                )
                if emergency_ice_shroom is not None:
                    return emergency_ice_shroom
            shooter_counts = _shooter_counts(cells, rows)
            jalapeno_waiting = any(
                plant.seed_type == int(SeedType.JALAPENO) for plant in plants
            )
            urgent_rows = tuple(
                row
                for row, _ in sorted(
                    boss_lanes.items(),
                    key=lambda item: (min(item[1]), item[0]),
                )
            )
            if jalapeno_waiting and urgent_rows:
                emergency_support = self._support_action(
                    plants,
                    cells,
                    support_type=SeedType.FLOWER_POT,
                    rows=urgent_rows,
                    columns=(6, 5, 7, 4, 3),
                )
                if emergency_support is not None:
                    return emergency_support
            coverage_rows = tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        shooter_counts[row],
                        min(boss_lanes.get(row, [float("inf")])),
                        row,
                    ),
                )
            )
            if min(shooter_counts.values(), default=0) >= 1:
                reserve_support = self._support_action(
                    plants,
                    cells,
                    support_type=SeedType.FLOWER_POT,
                    rows=coverage_rows,
                    columns=(6, 5, 7),
                )
                if reserve_support is not None:
                    return reserve_support
            roof_pult = _pick(
                plants,
                seed_types=SHOOTER_TYPES,
                rows=coverage_rows,
                columns=(3, 2, 1, 4, 0),
            )
            if roof_pult is not None:
                return roof_pult
            roof_pult_support = self._support_action(
                plants,
                cells,
                support_type=SeedType.FLOWER_POT,
                rows=coverage_rows,
                columns=(3, 2, 1, 4, 0),
            )
            if roof_pult_support is not None:
                return roof_pult_support
            plants = tuple(
                plant
                for plant in plants
                if plant.seed_type
                not in {
                    int(SeedType.JALAPENO),
                    int(SeedType.ICE_SHROOM),
                }
            )
            if not plants:
                return 0

        if int(payload.get("level", 0)) == 45 or int(payload.get("game_mode", 0)) == 48:
            cells = _plants_by_cell(payload)
            rows = tuple(range(int(payload.get("row_count", 5))))
            chomper_value = int(SeedType.CHOMPER)
            chomper_counts = {
                row: sum(
                    chomper_value in seed_types
                    for (plant_row, _), seed_types in cells.items()
                    if plant_row == row
                )
                for row in rows
            }
            coverage_rows = tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        chomper_counts[row],
                        min(lanes.get(row, [float("inf")])),
                        row,
                    ),
                )
            )
            chomper = _pick(
                plants,
                seed_types={SeedType.CHOMPER},
                rows=coverage_rows,
                columns=(3, 4, 2, 5, 1, 6, 0),
            )
            if chomper is not None:
                return chomper
            chomper_support = self._support_action(
                plants,
                cells,
                support_type=SeedType.FLOWER_POT,
                rows=coverage_rows,
                columns=(3, 4, 2, 5, 1, 6, 0),
            )
            if chomper_support is not None:
                return chomper_support
            pumpkin_value = int(SeedType.PUMPKIN)
            pumpkin_counts = {
                row: sum(
                    pumpkin_value in seed_types
                    for (plant_row, _), seed_types in cells.items()
                    if plant_row == row
                )
                for row in rows
            }
            pumpkin_rows = tuple(
                sorted(
                    (row for row in rows if chomper_counts[row] > 0),
                    key=lambda row: (
                        pumpkin_counts[row],
                        min(lanes.get(row, [float("inf")])),
                        row,
                    ),
                )
            )
            pumpkin = _pick(
                plants,
                seed_types={SeedType.PUMPKIN},
                rows=pumpkin_rows,
                columns=(3, 4, 2, 5, 1, 6, 0),
            )
            if pumpkin is not None:
                return pumpkin

        instant = self._instant_action(payload, plants, lanes)
        if instant is not None and any(min(xs) < 430.0 for xs in lanes.values()):
            return instant

        if is_miniature_pool_conveyor:
            cells = _plants_by_cell(payload)
            shooter_counts = _shooter_counts(cells, (2, 3))
            self._covered_pool_rows.update(
                row for row in (2, 3) if shooter_counts[row] >= 1
            )
            uncovered_water_rows = tuple(
                row for row in (2, 3) if shooter_counts[row] == 0
            )
            water_shooter = _pick(
                plants,
                seed_types=SHOOTER_TYPES,
                rows=uncovered_water_rows,
                columns=(2, 3, 1, 4, 0, 5),
            )
            if water_shooter is not None:
                return water_shooter
            water_shooter_support = _pick(
                plants,
                seed_types={SeedType.LILY_PAD},
                rows=uncovered_water_rows,
                columns=(2, 3, 1, 4, 0, 5),
            )
            if water_shooter_support is not None:
                return water_shooter_support
            defense_type_values = {int(seed_type) for seed_type in DEFENSE_TYPES}
            water_defense_rows = tuple(
                row
                for row in row_order
                if row in (2, 3)
                if row in self._covered_pool_rows
                and not any(
                    seed_type in defense_type_values
                    for (plant_row, _), seed_types in cells.items()
                    if plant_row == row
                    for seed_type in seed_types
                )
            )
            water_defense = _pick(
                plants,
                seed_types=DEFENSE_TYPES,
                rows=water_defense_rows,
                columns=(6, 7, 5, 8, 4),
            )
            if water_defense is not None:
                return water_defense
            water_support_rows = tuple(
                row
                for row in water_defense_rows
                if not any(
                    seed_types == [int(SeedType.LILY_PAD)]
                    for (plant_row, col), seed_types in cells.items()
                    if plant_row == row and col in (6, 7, 5, 8, 4)
                )
            )
            water_defense_support = _pick(
                plants,
                seed_types={SeedType.LILY_PAD},
                rows=water_support_rows,
                columns=(6, 7, 5, 8, 4),
            )
            if water_defense_support is not None:
                return water_defense_support
            reinforcement_rows = tuple(
                sorted(
                    (2, 3),
                    key=lambda row: (
                        shooter_counts[row],
                        min(lanes.get(row, [float("inf")])),
                        row,
                    ),
                )
            )
            if min(shooter_counts.values(), default=2) < 2:
                water_reinforcement = _pick(
                    plants,
                    seed_types=SHOOTER_TYPES,
                    rows=reinforcement_rows,
                    columns=(3, 2, 1, 4, 0, 5),
                )
                if water_reinforcement is not None:
                    return water_reinforcement
                water_reinforcement_support = self._support_action(
                    plants,
                    cells,
                    support_type=SeedType.LILY_PAD,
                    rows=tuple(
                        row for row in reinforcement_rows if shooter_counts[row] < 2
                    ),
                    columns=(3, 2, 1, 4, 0, 5),
                )
                if water_reinforcement_support is not None:
                    return water_reinforcement_support

        for seed_types, columns in (
            (SHOOTER_TYPES, (2, 3, 4, 1, 5, 0, 6, 7, 8)),
            (SUPPORT_TYPES, (2, 3, 1, 4, 0, 5, 6, 7, 8)),
            (DEFENSE_TYPES, (6, 7, 5, 8, 4, 3, 2, 1, 0)),
            (INSTANT_TYPES, (7, 6, 8, 5, 4, 3, 2, 1, 0)),
        ):
            placement_rows = (
                defense_row_order if seed_types is DEFENSE_TYPES else row_order
            )
            action = _pick(
                plants,
                seed_types=seed_types,
                rows=placement_rows,
                columns=columns,
            )
            if action is not None:
                return action
        fallback_plants = (
            tuple(plant for plant in plants if plant.seed_type != int(SeedType.BLOVER))
            if int(payload.get("level", 0)) == 40
            else plants
        )
        if fallback_plants:
            return min(plant.action for plant in fallback_plants)
        return 0

    def _chosen_seed_action(
        self, payload: dict[str, Any], plants: tuple[LegalPlant, ...]
    ) -> int:
        if int(payload.get("game_mode", 0)) == 0 and int(payload.get("level", 0)) <= 20:
            return self._early_adventure_action(payload, plants)

        row_count = int(payload.get("row_count", 5))
        rows = tuple(range(row_count))
        cells = _plants_by_cell(payload)
        lanes = _lane_zombies(payload)
        has_pool = bool(payload.get("stage_has_pool", False))
        has_roof = bool(payload.get("stage_has_roof", False))
        selected_types = {
            int(packet["type"]) for packet in payload.get("seed_packets", [])
        }
        shooter_columns = (2, 3, 4, 1, 5)

        if (
            int(payload.get("game_mode", 0)) == 2
            and int(payload.get("survival_stage", 0)) >= 4
            and lanes
        ):
            ready_clear_costs = [
                cost
                for packet in payload.get("seed_packets", [])
                for seed_type, cost in (
                    (SeedType.CHERRY_BOMB, 150),
                    (SeedType.JALAPENO, 125),
                )
                if int(packet.get("type", -1)) == int(seed_type)
                and int(packet.get("refresh", 0)) == 0
            ]
            if ready_clear_costs and int(payload.get("sun", 0)) < min(
                ready_clear_costs
            ):
                shooter_counts = _shooter_counts(cells, rows)
                uncovered_rows = tuple(
                    row
                    for row, _ in sorted(
                        lanes.items(),
                        key=lambda item: (min(item[1]), item[0]),
                    )
                    if shooter_counts.get(row, 0) == 0
                )
                free_coverage = _pick(
                    plants,
                    seed_types={SeedType.PUFF_SHROOM},
                    rows=uncovered_rows,
                    columns=shooter_columns,
                )
                if free_coverage is not None:
                    return free_coverage
                return 0

        cattail_value = int(SeedType.CATTAIL)
        cattail_count = sum(
            cattail_value in seed_types for seed_types in cells.values()
        )
        if cattail_value in selected_types and cattail_count < 2:
            cattail = _pick(
                plants,
                seed_types={SeedType.CATTAIL},
                rows=tuple(row for row in (2, 3) if row in rows),
                columns=(3, 4, 2, 5, 1, 6, 0, 7, 8),
            )
            if cattail is not None:
                return cattail
            pure_lily_exists = any(
                seed_types == [int(SeedType.LILY_PAD)]
                for (row, _), seed_types in cells.items()
                if row in (2, 3)
            )
            if not pure_lily_exists:
                cattail_support = self._support_action(
                    plants,
                    cells,
                    support_type=SeedType.LILY_PAD,
                    rows=tuple(row for row in (2, 3) if row in rows),
                    columns=(3, 4, 2, 5, 1, 6, 0, 7, 8),
                )
                if cattail_support is not None:
                    return cattail_support
            # Once a base exists, preserve the 225 sun needed by each of the
            # first two shop upgrades. Free mushrooms can still add pressure
            # while the retained Sun-shrooms produce and the packet refreshes.
            if pure_lily_exists and int(payload.get("sun", 0)) < 225:
                free_shooter = _pick(
                    plants,
                    seed_types={SeedType.PUFF_SHROOM, SeedType.SEA_SHROOM},
                    rows=rows,
                    columns=(2, 3, 4, 1, 5, 0, 6, 7, 8),
                )
                if free_shooter is not None:
                    return free_shooter
                return 0

        catapult_rows = tuple(
            dict.fromkeys(
                int(zombie["row"])
                for zombie in sorted(
                    payload.get("zombies", []),
                    key=lambda zombie: (
                        float(zombie["x"]),
                        int(zombie["row"]),
                    ),
                )
                if int(zombie.get("type", -1)) == 23
                and not bool(zombie.get("mind_controlled", False))
            )
        )
        if has_roof and catapult_rows:
            catapult_jalapeno = _pick(
                plants,
                seed_types={SeedType.JALAPENO},
                rows=catapult_rows,
                columns=(4, 3, 5, 2, 1, 6, 0),
            )
            if catapult_jalapeno is not None:
                return catapult_jalapeno
            jalapeno_ready = any(
                int(packet.get("type", -1)) == int(SeedType.JALAPENO)
                and int(packet.get("refresh", 0)) == 0
                for packet in payload.get("seed_packets", [])
            )
            pure_pot_available = any(
                seed_types == [int(SeedType.FLOWER_POT)]
                and row in catapult_rows
                and col in (4, 3, 5, 2, 1, 6, 0)
                for (row, col), seed_types in cells.items()
            )
            if jalapeno_ready and pure_pot_available:
                return 0

        pogo_zombies = tuple(
            zombie
            for zombie in sorted(
                payload.get("zombies", []),
                key=lambda zombie: (float(zombie["x"]), int(zombie["row"])),
            )
            if int(zombie.get("type", -1)) == 18
            and not bool(zombie.get("mind_controlled", False))
        )
        pogo_rows = tuple(dict.fromkeys(int(zombie["row"]) for zombie in pogo_zombies))
        if has_roof and pogo_rows:
            tall_nut_value = int(SeedType.TALL_NUT)
            pogo_counter_rows = tuple(
                row
                for row in pogo_rows
                if not any(
                    tall_nut_value in seed_types
                    for (plant_row, _), seed_types in cells.items()
                    if plant_row == row
                )
            )
            pogo_tall_nut = _pick(
                plants,
                seed_types={SeedType.TALL_NUT},
                rows=pogo_counter_rows,
                columns=(6, 7, 5, 4, 3, 2, 1, 0),
            )
            if pogo_tall_nut is not None:
                return pogo_tall_nut
            pogo_squash = _pick(
                plants,
                seed_types={SeedType.SQUASH},
                rows=pogo_counter_rows,
                columns=(5, 4, 6, 3, 2, 1, 0),
            )
            if pogo_squash is not None:
                return pogo_squash
            tall_nut_ready = any(
                int(packet.get("type", -1)) == tall_nut_value
                and int(packet.get("refresh", 0)) == 0
                for packet in payload.get("seed_packets", [])
            )
            pure_pot_available = any(
                seed_types == [int(SeedType.FLOWER_POT)]
                and row in pogo_counter_rows
                and col in (6, 7, 5, 4, 3, 2, 1, 0)
                for (row, col), seed_types in cells.items()
            )
            if (
                pogo_counter_rows
                and tall_nut_ready
                and pure_pot_available
                and int(payload.get("sun", 0)) < 125
            ):
                return 0

        balloon_rows = tuple(
            int(zombie["row"])
            for zombie in sorted(
                payload.get("zombies", []),
                key=lambda zombie: (float(zombie["x"]), int(zombie["row"])),
            )
            if int(zombie.get("type", -1)) == 16
            and not bool(zombie.get("mind_controlled", False))
        )
        if balloon_rows:
            air_defense = _pick(
                plants,
                seed_types={SeedType.CACTUS},
                rows=balloon_rows,
                columns=(2, 3, 4, 1, 5, 0, 6, 7, 8),
            )
            if air_defense is not None:
                return air_defense
            blover = _pick(
                plants,
                seed_types={SeedType.BLOVER},
                rows=rows,
                columns=(0, 1, 2, 3, 4, 5, 6, 7, 8),
            )
            if blover is not None:
                return blover
            if {
                int(SeedType.CACTUS),
                int(SeedType.BLOVER),
            } & selected_types:
                free_coverage = _pick(
                    plants,
                    seed_types={SeedType.PUFF_SHROOM, SeedType.SEA_SHROOM},
                    rows=rows,
                    columns=(1, 2, 3, 4, 5),
                )
                if free_coverage is not None:
                    return free_coverage
                return 0

        selected_air_counter_cost = (
            100
            if int(SeedType.BLOVER) in selected_types
            else 125
            if int(SeedType.CACTUS) in selected_types
            else 0
        )
        if selected_air_counter_cost and not balloon_rows:
            shooter_counts = _shooter_counts(cells, rows)
            if all(shooter_counts[row] >= 1 for row in rows):
                self._air_reserve_active = True
            if self._air_reserve_active:
                close_threat_rows = tuple(
                    row
                    for row, xs in sorted(
                        lanes.items(),
                        key=lambda item: (min(item[1]), item[0]),
                    )
                    if min(xs) < 420.0
                )
                close_water_rows = tuple(
                    row for row in close_threat_rows if row in (2, 3)
                )
                emergency_tangle = _pick(
                    plants,
                    seed_types={SeedType.TANGLE_KELP},
                    rows=close_water_rows,
                    columns=(4, 3, 2, 1, 5, 0),
                )
                if emergency_tangle is not None:
                    return emergency_tangle
                if has_pool and int(payload.get("level", 0)) >= 36 and close_water_rows:
                    emergency_water_support = self._support_action(
                        plants,
                        cells,
                        support_type=SeedType.LILY_PAD,
                        rows=close_water_rows,
                        columns=(3, 2, 1, 0, 4, 5),
                    )
                    if emergency_water_support is not None:
                        return emergency_water_support
                if close_threat_rows and int(payload.get("sun", 0)) >= 75:
                    emergency_hypno = _pick(
                        plants,
                        seed_types={SeedType.HYPNO_SHROOM},
                        rows=close_threat_rows,
                        columns=(4, 3, 2, 1, 5, 0),
                    )
                    if emergency_hypno is not None:
                        return emergency_hypno
                    emergency_fume = _pick(
                        plants,
                        seed_types={SeedType.FUME_SHROOM},
                        rows=close_threat_rows,
                        columns=(4, 3, 5, 2, 1),
                    )
                    if emergency_fume is not None:
                        return emergency_fume
                free_coverage = _pick(
                    plants,
                    seed_types={SeedType.PUFF_SHROOM, SeedType.SEA_SHROOM},
                    rows=rows,
                    columns=(1, 3, 4, 5),
                )
                if free_coverage is not None:
                    return free_coverage
                if int(payload.get("sun", 0)) >= selected_air_counter_cost + 75:
                    buffered_rows = tuple(
                        row
                        for row, _ in sorted(
                            lanes.items(),
                            key=lambda item: (
                                shooter_counts.get(item[0], 0),
                                min(item[1]),
                                item[0],
                            ),
                        )
                    )
                    buffered_rows += tuple(
                        row
                        for row in sorted(
                            rows,
                            key=lambda item: (shooter_counts[item], item),
                        )
                        if row not in lanes
                    )
                    buffered_fume = _pick(
                        plants,
                        seed_types={SeedType.FUME_SHROOM},
                        rows=buffered_rows,
                        columns=(3, 4, 2, 5, 1),
                    )
                    if buffered_fume is not None:
                        return buffered_fume
                if int(payload.get("sun", 0)) >= selected_air_counter_cost + 50:
                    buffered_defense = _pick(
                        plants,
                        seed_types=DEFENSE_TYPES,
                        rows=rows,
                        columns=(6, 7, 5),
                    )
                    if buffered_defense is not None:
                        return buffered_defense
                return 0

        instant = self._instant_action(payload, plants, lanes)
        reserve_roof_instants = (
            int(payload.get("game_mode", 0)) in {5, 10}
            and int(payload.get("survival_stage", 0)) >= 1
        )
        final_roof_stage = (
            reserve_roof_instants and int(payload.get("survival_stage", 0)) >= 4
        )
        instant_trigger_x = (
            620.0 if final_roof_stage or not reserve_roof_instants else 470.0
        )
        if instant is not None and any(
            min(xs) < instant_trigger_x for xs in lanes.values()
        ):
            if reserve_roof_instants:
                ready_mower_rows = {
                    int(mower["row"])
                    for mower in payload.get("mowers", [])
                    if int(mower.get("state", -1)) == 1
                }
                emergency_rows = tuple(
                    row
                    for row, xs in sorted(
                        lanes.items(),
                        key=lambda item: (
                            item[0] in ready_mower_rows,
                            min(item[1]),
                            item[0],
                        ),
                    )
                    if min(xs) < instant_trigger_x
                )
                instant_row = self.codec.decode(instant).row
                if emergency_rows and instant_row != emergency_rows[0]:
                    emergency_support = self._support_action(
                        plants,
                        cells,
                        support_type=SeedType.FLOWER_POT,
                        rows=(emergency_rows[0],),
                        columns=(3, 2, 4, 1, 5, 0, 6),
                    )
                    if emergency_support is not None:
                        return emergency_support
            return instant

        support_rows = (
            tuple(range(row_count)) if has_roof else ((2, 3) if has_pool else ())
        )
        support_type = SeedType.FLOWER_POT if has_roof else SeedType.LILY_PAD
        shooter_counts = _shooter_counts(cells, rows)
        uncovered_threats = tuple(
            row
            for row, _ in sorted(
                lanes.items(),
                key=lambda item: (min(item[1]), item[0]),
            )
            if shooter_counts.get(row, 0) == 0
        )
        if uncovered_threats:
            threat_support = self._support_action(
                plants,
                cells,
                support_type=support_type,
                rows=tuple(row for row in uncovered_threats if row in support_rows),
                columns=shooter_columns,
            )
            if threat_support is not None:
                return threat_support
            preferred_shooters = self._strategic_shooter_types(payload)
            threat_shooter = _pick(
                plants,
                seed_types=preferred_shooters,
                rows=uncovered_threats,
                columns=shooter_columns,
            )
            if threat_shooter is not None:
                return threat_shooter
            if any(min(lanes[row]) < 520.0 for row in uncovered_threats):
                emergency_shooter_types = (
                    SHOOTER_TYPES - {SeedType.CACTUS}
                    if bool(payload.get("stage_is_night", False))
                    else SHOOTER_TYPES
                )
                fallback_shooter = _pick(
                    plants,
                    seed_types=emergency_shooter_types,
                    rows=uncovered_threats,
                    columns=shooter_columns,
                )
                if fallback_shooter is not None:
                    return fallback_shooter
            if instant is not None and (
                not reserve_roof_instants
                or any(min(xs) < instant_trigger_x for xs in lanes.values())
            ):
                return instant
            return 0

        if int(SeedType.TORCHWOOD) in selected_types:
            pea_coverage = dict.fromkeys(rows, 0)
            pea_values = {int(seed_type) for seed_type in PEA_PROJECTILE_TYPES}
            for (plant_row, _), seed_types in cells.items():
                for seed_type in seed_types:
                    if seed_type not in pea_values:
                        continue
                    affected_rows = (
                        (plant_row - 1, plant_row, plant_row + 1)
                        if seed_type == int(SeedType.THREEPEATER)
                        else (plant_row,)
                    )
                    for row in affected_rows:
                        if row in pea_coverage:
                            pea_coverage[row] += 1
            if min(pea_coverage.values(), default=0) >= 1:
                torchwood_value = int(SeedType.TORCHWOOD)
                torchwood_rows = tuple(
                    row
                    for row in rows
                    if not any(
                        torchwood_value in seed_types
                        for (plant_row, _), seed_types in cells.items()
                        if plant_row == row
                    )
                )
                torchwood_support = self._support_action(
                    plants,
                    cells,
                    support_type=support_type,
                    rows=tuple(row for row in torchwood_rows if row in support_rows),
                    columns=(5, 4),
                )
                if torchwood_support is not None:
                    return torchwood_support
                torchwood = _pick(
                    plants,
                    seed_types={SeedType.TORCHWOOD},
                    rows=torchwood_rows,
                    columns=(5, 4),
                )
                if torchwood is not None:
                    return torchwood

        economy_type_values = {int(seed_type) for seed_type in ECONOMY_TYPES}
        economy_counts = {
            row: sum(
                seed_type in economy_type_values
                for (plant_row, _), seed_types in cells.items()
                if plant_row == row
                for seed_type in seed_types
            )
            for row in rows
        }
        # Establish one producer per lane until Squash is available. At that
        # point the faster pool waves are better served by keeping the opening
        # economy on land and reserving lily pads for defense.
        land_only_pool_economy = has_pool and (
            bool(payload.get("stage_is_night", False))
            or int(SeedType.SQUASH) in selected_types
        )
        economy_targets = {
            row: (
                0
                if land_only_pool_economy and row in (2, 3)
                else 2
                if int(payload.get("game_mode", 0)) == 2
                else 1
            )
            for row in rows
        }
        economy_rows = tuple(
            sorted(
                rows,
                key=lambda row: (
                    economy_counts[row] - economy_targets[row],
                    row,
                ),
            )
        )
        incomplete_economy = tuple(
            row for row in economy_rows if economy_counts[row] < economy_targets[row]
        )
        economy_support = self._support_action(
            plants,
            cells,
            support_type=support_type,
            rows=tuple(row for row in incomplete_economy if row in support_rows),
            columns=(0, 1),
        )
        if economy_support is not None:
            return economy_support
        if incomplete_economy:
            action = _pick(
                plants,
                seed_types=ECONOMY_TYPES,
                rows=incomplete_economy,
                columns=(0, 1, 2),
            )
            if action is not None:
                return action
            if selected_types & {int(seed_type) for seed_type in ECONOMY_TYPES}:
                return 0

        if (
            int(payload.get("game_mode", 0)) == 2
            and int(SeedType.GLOOM_SHROOM) in selected_types
        ):
            gloom_value = int(SeedType.GLOOM_SHROOM)
            pumpkin_value = int(SeedType.PUMPKIN)
            gloom_cells = {
                cell for cell, seed_types in cells.items() if gloom_value in seed_types
            }
            unprotected_glooms = {
                cell for cell in gloom_cells if pumpkin_value not in cells.get(cell, [])
            }
            pumpkin_choices = [
                plant
                for plant in plants
                if plant.seed_type == pumpkin_value
                and (plant.row, plant.col) in unprotected_glooms
            ]
            if pumpkin_choices:
                return min(
                    pumpkin_choices,
                    key=lambda plant: (
                        min(lanes.get(plant.row, [float("inf")])),
                        plant.row,
                        plant.col,
                    ),
                ).action

            gloom_rows = {row for row, _ in gloom_cells}
            missing_gloom_rows = tuple(
                row
                for row in sorted(
                    rows,
                    key=lambda row: (
                        min(lanes.get(row, [float("inf")])),
                        row,
                    ),
                )
                if row not in gloom_rows
            )
            if missing_gloom_rows:
                gloom = _pick(
                    plants,
                    seed_types={SeedType.GLOOM_SHROOM},
                    rows=missing_gloom_rows,
                    columns=(4, 3, 2, 5, 1),
                )
                if gloom is not None:
                    return gloom
                fume_base = _pick(
                    plants,
                    seed_types={SeedType.FUME_SHROOM},
                    rows=missing_gloom_rows,
                    columns=(4, 3, 2, 5, 1),
                )
                if fume_base is not None:
                    return fume_base
                gloom_ready = any(
                    int(packet.get("type", -1)) == gloom_value
                    and int(packet.get("refresh", 0)) == 0
                    for packet in payload.get("seed_packets", [])
                )
                has_upgrade_base = any(
                    int(SeedType.FUME_SHROOM) in seed_types
                    and row in missing_gloom_rows
                    for (row, _), seed_types in cells.items()
                )
                if gloom_ready and has_upgrade_base:
                    return 0

        if (
            has_roof
            and (
                int(payload.get("level", 0)) >= 44
                or int(payload.get("game_mode", 0)) in {5, 10}
            )
            and bool(selected_types & {int(SeedType.TALL_NUT), int(SeedType.SQUASH)})
            and min(shooter_counts.values(), default=0) >= 1
        ):
            tall_nut_value = int(SeedType.TALL_NUT)
            squash_value = int(SeedType.SQUASH)
            roof_counter_values = {tall_nut_value, squash_value}
            nearest_roof_threats = tuple(
                row
                for row, _ in sorted(
                    lanes.items(),
                    key=lambda item: (min(item[1]), item[0]),
                )
            )
            preferred_roof_rows = tuple(
                dict.fromkeys((*pogo_rows, *nearest_roof_threats, 2, 4, 1, 3, 0))
            )
            roof_defense_rows = tuple(
                row
                for row in preferred_roof_rows
                if row in rows
                and not any(
                    roof_counter_values.intersection(seed_types)
                    for (plant_row, _), seed_types in cells.items()
                    if plant_row == row
                )
            )
            if roof_defense_rows:
                target_row = roof_defense_rows[0]
                roof_tall_nut = _pick(
                    plants,
                    seed_types={SeedType.TALL_NUT},
                    rows=(target_row,),
                    columns=(6, 7, 5),
                )
                if roof_tall_nut is not None:
                    return roof_tall_nut
                roof_squash = _pick(
                    plants,
                    seed_types={SeedType.SQUASH},
                    rows=(target_row,),
                    columns=(6, 7, 5),
                )
                if roof_squash is not None:
                    return roof_squash
                roof_defense_support = self._support_action(
                    plants,
                    cells,
                    support_type=SeedType.FLOWER_POT,
                    rows=(target_row,),
                    columns=(6, 7, 5),
                )
                if roof_defense_support is not None:
                    return roof_defense_support
                roof_counter_ready = any(
                    int(packet.get("type", -1)) in roof_counter_values
                    and int(packet.get("refresh", 0)) == 0
                    for packet in payload.get("seed_packets", [])
                )
                pure_pot_available = any(
                    cells.get((target_row, col)) == [int(SeedType.FLOWER_POT)]
                    for col in (6, 7, 5)
                )
                if roof_counter_ready and pure_pot_available:
                    return 0

        covered_rows = sum(shooter_counts[row] > 0 for row in rows)
        if (
            not bool(payload.get("stage_is_night", False))
            and int(payload.get("wave", 0)) >= 4
            and covered_rows >= max(3, row_count // 2)
        ):
            expansion_rows = tuple(
                row
                for row in rows
                if (not has_pool or row not in (2, 3)) and economy_counts[row] < 2
            )
            expansion = _pick(
                plants,
                seed_types={SeedType.SUNFLOWER},
                rows=expansion_rows,
                columns=(1, 0),
            )
            if expansion is not None:
                return expansion

        defense_type_values = {int(seed_type) for seed_type in DEFENSE_TYPES}
        if has_pool:
            water_defense_rows = tuple(
                row
                for row in (2, 3)
                if shooter_counts[row] >= 1
                and not any(
                    seed_type in defense_type_values
                    for (plant_row, _), seed_types in cells.items()
                    if plant_row == row
                    for seed_type in seed_types
                )
            )
            water_defense_support = self._support_action(
                plants,
                cells,
                support_type=SeedType.LILY_PAD,
                rows=water_defense_rows,
                columns=(6, 7, 5),
            )
            if water_defense_support is not None:
                return water_defense_support
            water_defense = _pick(
                plants,
                seed_types=DEFENSE_TYPES,
                rows=water_defense_rows,
                columns=(6, 7, 5),
            )
            if water_defense is not None:
                return water_defense

        threatened_rows = tuple(
            row
            for row, _ in sorted(
                lanes.items(),
                key=lambda item: (
                    shooter_counts.get(item[0], 0),
                    min(item[1]),
                    item[0],
                ),
            )
        )
        shooter_rows = threatened_rows + tuple(
            row
            for row in sorted(rows, key=lambda item: (shooter_counts[item], item))
            if row not in lanes
        )
        if has_pool and (
            int(payload.get("level", 0)) <= 23
            or bool(payload.get("stage_is_night", False))
        ):
            opening_defense_rows = tuple(
                row
                for row in shooter_rows
                if row not in (2, 3)
                and shooter_counts[row] >= 1
                and not any(
                    seed_type in defense_type_values
                    for (plant_row, _), seed_types in cells.items()
                    if plant_row == row
                    for seed_type in seed_types
                )
            )
            opening_defense = _pick(
                plants,
                seed_types=DEFENSE_TYPES,
                rows=opening_defense_rows,
                columns=(6, 7, 5),
            )
            if opening_defense is not None:
                return opening_defense
        if min(shooter_counts.values(), default=2) < 2:
            shooter_support = self._support_action(
                plants,
                cells,
                support_type=support_type,
                rows=tuple(
                    row
                    for row in shooter_rows
                    if row in support_rows and shooter_counts[row] < 2
                ),
                columns=shooter_columns,
            )
            if shooter_support is not None:
                return shooter_support
            action = _pick(
                plants,
                seed_types=self._strategic_shooter_types(payload),
                rows=shooter_rows,
                columns=shooter_columns,
            )
            if action is not None:
                return action
            return 0

        defense_rows = tuple(
            row
            for row in shooter_rows
            if not any(
                seed_type in defense_type_values
                for (plant_row, _), seed_types in cells.items()
                if plant_row == row
                for seed_type in seed_types
            )
        )
        if defense_rows:
            defense_support = self._support_action(
                plants,
                cells,
                support_type=support_type,
                rows=tuple(row for row in defense_rows if row in support_rows),
                columns=(6, 7, 5),
            )
            if defense_support is not None:
                return defense_support
            action = _pick(
                plants,
                seed_types=DEFENSE_TYPES,
                rows=defense_rows,
                columns=(6, 7, 5),
            )
            if action is not None:
                return action

        pumpkin_value = int(SeedType.PUMPKIN)
        pumpkin_targets = {
            (row, col)
            for (row, col), seed_types in cells.items()
            if {
                int(SeedType.TORCHWOOD),
                int(SeedType.CATTAIL),
            }.intersection(seed_types)
            and pumpkin_value not in seed_types
        }
        pumpkin_actions = [
            plant
            for plant in plants
            if plant.seed_type == pumpkin_value
            and (plant.row, plant.col) in pumpkin_targets
        ]
        if pumpkin_actions:
            return min(
                pumpkin_actions,
                key=lambda plant: (
                    min(lanes.get(plant.row, [float("inf")])),
                    plant.row,
                    -plant.col,
                ),
            ).action

        if has_pool or has_roof:
            action = _pick(
                plants,
                seed_types=SUPPORT_TYPES,
                rows=shooter_rows,
                columns=(2, 3, 4, 1, 5, 0, 6),
            )
            if action is not None:
                return action

        action = _pick(
            plants,
            seed_types=self._strategic_shooter_types(payload),
            rows=shooter_rows,
            columns=(3, 4, 2, 5, 1, 6),
        )
        if action is not None:
            return action

        instant = self._instant_action(payload, plants, lanes)
        if instant is not None and (
            not reserve_roof_instants
            or any(min(xs) < instant_trigger_x for xs in lanes.values())
        ):
            return instant
        return 0

    def _early_adventure_action(
        self, payload: dict[str, Any], plants: tuple[LegalPlant, ...]
    ) -> int:
        row_count = int(payload.get("row_count", 5))
        rows = tuple(range(row_count))
        cells = _plants_by_cell(payload)
        lanes = _lane_zombies(payload)
        instant = self._instant_action(payload, plants, lanes)
        if instant is not None and any(min(xs) < 360.0 for xs in lanes.values()):
            return instant

        economy_type_values = {int(seed_type) for seed_type in ECONOMY_TYPES}
        economy_counts = {
            row: sum(
                seed_type in economy_type_values
                for (plant_row, _), seed_types in cells.items()
                if plant_row == row
                for seed_type in seed_types
            )
            for row in rows
        }
        economy_target = 1 if bool(payload.get("stage_is_night", False)) else 2
        economy_rows = tuple(sorted(rows, key=lambda row: (economy_counts[row], row)))
        if min(economy_counts.values(), default=economy_target) < economy_target:
            action = _pick(
                plants,
                seed_types=ECONOMY_TYPES,
                rows=economy_rows,
                columns=(0, 1, 2),
            )
            if action is not None:
                return action

        shooter_counts = _shooter_counts(cells, rows)
        threatened_rows = tuple(
            row
            for row, _ in sorted(
                lanes.items(),
                key=lambda item: (
                    shooter_counts.get(item[0], 0),
                    min(item[1]),
                    item[0],
                ),
            )
        )
        shooter_rows = threatened_rows + tuple(
            row
            for row in sorted(rows, key=lambda item: (shooter_counts[item], item))
            if row not in lanes
        )
        if min(shooter_counts.values(), default=2) < 2:
            action = _pick(
                plants,
                seed_types=SHOOTER_TYPES,
                rows=shooter_rows,
                columns=(2, 3, 4, 1, 5),
            )
            if action is not None:
                return action

        defense_type_values = {int(seed_type) for seed_type in DEFENSE_TYPES}
        defense_rows = tuple(
            row
            for row in shooter_rows
            if not any(
                seed_type in defense_type_values
                for (plant_row, _), seed_types in cells.items()
                if plant_row == row
                for seed_type in seed_types
            )
        )
        if defense_rows:
            action = _pick(
                plants,
                seed_types=DEFENSE_TYPES,
                rows=defense_rows,
                columns=(6, 7, 5),
            )
            if action is not None:
                return action

        action = _pick(
            plants,
            seed_types=SHOOTER_TYPES,
            rows=shooter_rows,
            columns=(3, 4, 2, 5, 1, 6),
        )
        if action is not None:
            return action
        if instant is not None:
            return instant
        return 0

    def _preferred_shooter_types(self, payload: dict[str, Any]) -> set[SeedType]:
        selected = {
            SeedType(int(packet["type"]))
            for packet in payload.get("seed_packets", [])
            if int(packet["type"]) in {int(seed_type) for seed_type in SHOOTER_TYPES}
        }
        if not selected:
            return SHOOTER_TYPES
        best_score = max(SHOOTER_PRIORITY.get(seed_type, 0) for seed_type in selected)
        return {
            seed_type
            for seed_type in selected
            if SHOOTER_PRIORITY.get(seed_type, 0) == best_score
        }

    def _strategic_shooter_types(self, payload: dict[str, Any]) -> set[SeedType]:
        if bool(payload.get("stage_is_night", False)):
            return SHOOTER_TYPES - {SeedType.CACTUS}
        if (
            1 <= int(payload.get("game_mode", 0)) <= 10
            and int(payload.get("survival_stage", 0)) >= 1
        ):
            selected = {
                SeedType(int(packet["type"]))
                for packet in payload.get("seed_packets", [])
                if int(packet["type"])
                in {int(seed_type) for seed_type in SHOOTER_TYPES}
            }
            if selected:
                return selected
        if int(payload.get("level", 0)) <= 20:
            return SHOOTER_TYPES
        return self._preferred_shooter_types(payload)

    def _support_action(
        self,
        plants: tuple[LegalPlant, ...],
        cells: dict[tuple[int, int], list[int]],
        *,
        support_type: SeedType,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
    ) -> int | None:
        if not rows:
            return None
        rows_needing_support = tuple(
            row
            for row in rows
            if not any(
                int(support_type) in cells.get((row, col), [])
                and all(
                    seed_type == int(support_type) for seed_type in cells[(row, col)]
                )
                for col in columns
            )
        )
        if not rows_needing_support:
            return None
        row_rank = {row: index for index, row in enumerate(rows_needing_support)}
        col_rank = {col: index for index, col in enumerate(columns)}
        choices = [
            plant
            for plant in plants
            if plant.seed_type == support_type
            and plant.row in row_rank
            and plant.col in col_rank
            and int(support_type) not in cells.get((plant.row, plant.col), [])
        ]
        if not choices:
            return None
        return min(
            choices,
            key=lambda plant: (
                row_rank[plant.row],
                col_rank[plant.col],
                plant.action,
            ),
        ).action

    def _instant_action(
        self,
        payload: dict[str, Any],
        plants: tuple[LegalPlant, ...],
        lanes: dict[int, list[float]],
    ) -> int | None:
        if not lanes:
            return None
        zombie_types: dict[int, list[int]] = {}
        for zombie in payload.get("zombies", []):
            if bool(zombie.get("mind_controlled", False)):
                continue
            zombie_types.setdefault(int(zombie["row"]), []).append(
                int(zombie.get("type", -1))
            )
        priority_zombie_types = {3, 4, 7, 18, 21, 23, 24}
        emergency_x = (
            470.0
            if int(payload.get("game_mode", 0)) in {5, 10}
            and int(payload.get("survival_stage", 0)) >= 1
            else 260.0
        )
        reserve_roof_instants = (
            int(payload.get("game_mode", 0)) in {5, 10}
            and int(payload.get("survival_stage", 0)) >= 1
        )
        ready_mower_rows = {
            int(mower["row"])
            for mower in payload.get("mowers", [])
            if int(mower.get("state", -1)) == 1
        }

        def threat_key(
            item: tuple[int, list[float]],
        ) -> tuple[bool, bool, float, bool, float, int]:
            row, xs = item
            closest = min(xs)
            emergency = closest < emergency_x
            priority = any(
                zombie_type in priority_zombie_types
                for zombie_type in zombie_types.get(row, [])
            )
            return (
                not emergency,
                reserve_roof_instants and row in ready_mower_rows,
                closest if emergency else 0.0,
                not priority,
                closest,
                row,
            )

        threat_rows = tuple(row for row, _ in sorted(lanes.items(), key=threat_key))

        def useful(plant: LegalPlant) -> bool:
            if int(payload.get("level", 0)) <= 20:
                return True
            seed_type = SeedType(plant.seed_type)
            if seed_type == SeedType.JALAPENO:
                lane_types = zombie_types.get(plant.row, [])
                return len(lane_types) >= 4 or any(
                    zombie_type in {7, 12, 13, 18, 21, 23, 24}
                    for zombie_type in lane_types
                )
            if seed_type == SeedType.CHERRY_BOMB:
                nearby_count = sum(
                    len(zombie_types.get(row, []))
                    for row in (plant.row - 1, plant.row, plant.row + 1)
                )
                return nearby_count >= 3
            return True

        choices = [
            plant
            for plant in plants
            if plant.seed_type in INSTANT_TYPES and plant.row in lanes and useful(plant)
        ]
        if not choices:
            return None
        row_rank = {row: index for index, row in enumerate(threat_rows)}

        def key(plant: LegalPlant) -> tuple[int, int, float, int]:
            zombie_x = min(lanes[plant.row])
            if plant.seed_type == int(SeedType.POTATO_MINE):
                # Mines need 15 seconds to arm. Planting directly beneath the
                # current zombie gets the packet eaten immediately; use the
                # established defensive columns while the lane is still long.
                mine_columns = (3, 2, 4, 1, 5)
                column_rank = (
                    mine_columns.index(plant.col)
                    if plant.col in mine_columns
                    else len(mine_columns) + plant.col
                )
                distance = float(column_rank)
            else:
                target_col = max(
                    0,
                    min(
                        self.codec.cols - 1,
                        int((zombie_x - 40) // 80),
                    ),
                )
                distance = abs(plant.col - target_col)
            return (
                row_rank[plant.row],
                INSTANT_PRIORITY.get(SeedType(plant.seed_type), 99),
                distance,
                plant.action,
            )

        return min(choices, key=key).action

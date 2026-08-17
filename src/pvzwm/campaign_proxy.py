from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb

PROXY_CAMPAIGN_STAGE_IDS = (
    "1-3",
    "1-4",
    "1-6",
    "1-7",
    "1-8",
    "1-9",
    "2-1",
    "2-2",
    "2-3",
    "2-4",
    "2-6",
    "2-7",
    "2-8",
    "2-9",
)

_PORT_ZOMBIE_NAMES = {
    "Zombie": "Zombie",
    "Flag Zombie": "FlagZombie",
    "Conehead Zombie": "ConeheadZombie",
    "Buckethead Zombie": "BucketheadZombie",
    "Newspaper Zombie": "NewspaperZombie",
}

_UNSUPPORTED_ZOMBIE_PROXIES = {
    "Pole Vaulting Zombie": "Zombie",
    "Screen Door Zombie": "BucketheadZombie",
    "Football Zombie": "BucketheadZombie",
    "Dancing Zombie": "Zombie",
    "Backup Dancer": "Zombie",
}


def campaign_database_path() -> Path:
    configured = os.environ.get("PVZWM_CAMPAIGN_DATABASE")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(__file__).resolve().parents[2] / "data" / "campaign.duckdb",
        Path.cwd() / "data" / "campaign.duckdb",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "The revision-pinned campaign archive is missing; run "
        "`uv run pvzwm sync-campaign-wiki --database data/campaign.duckdb`"
    )


@lru_cache(maxsize=64)
def _level_facts(stage_id: str, database: str) -> dict[str, Any]:
    connection = duckdb.connect(database, read_only=True)
    try:
        row = connection.execute(
            """
            SELECT p.canonical_url, p.revision_id, p.revision_timestamp,
                   p.content_sha256, f.location, f.level_type,
                   f.adventure1_wave_count, f.adventure1_waves,
                   f.zombie_types, f.first_reward_items
            FROM wiki_level_facts f
            JOIN wiki_level_pages p USING (stage_id)
            WHERE f.stage_id = ?
            """,
            [stage_id],
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError(f"Campaign archive has no facts for stage {stage_id}")
    return {
        "canonical_url": str(row[0]),
        "revision_id": int(row[1]),
        "revision_timestamp": str(row[2]),
        "content_sha256": str(row[3]),
        "location": str(row[4]),
        "level_type": str(row[5]),
        "wave_count": int(row[6]),
        "waves": json.loads(row[7]),
        "zombie_types": json.loads(row[8]),
        "reward_items": json.loads(row[9]),
    }


def _proxy_zombie(name: str) -> str:
    if name in _PORT_ZOMBIE_NAMES:
        return _PORT_ZOMBIE_NAMES[name]
    return _UNSUPPORTED_ZOMBIE_PROXIES.get(name, "BucketheadZombie")


def build_campaign_proxy(
    stage_id: str,
    seed: int = 0,
    database: str | Path | None = None,
) -> dict[str, Any]:
    """Build a seeded five-row proxy from archived first-playthrough wave facts."""
    if stage_id not in PROXY_CAMPAIGN_STAGE_IDS:
        raise ValueError(f"Stage {stage_id} has no mechanically reduced proxy")
    archive = Path(database).resolve() if database is not None else campaign_database_path()
    facts = _level_facts(stage_id, str(archive))
    if facts["wave_count"] < 1 or facts["level_type"] not in {"Regular", "Normal"}:
        raise ValueError(f"Stage {stage_id} is not a regular wave-based level")

    world, level = (int(part) for part in stage_id.split("-"))
    supported_plants = ["Peashooter", "Sunflower"]
    if world > 1 or level >= 4:
        supported_plants.append("Wall-nut")
    wave_facts = {int(wave["index"]): wave for wave in facts["waves"]}
    candidate_zombies = [
        name for name in facts["zombie_types"] if name != "Flag Zombie"
    ] or ["Zombie"]
    rng = random.Random(world * 1_000_000 + level * 10_000 + seed)
    wave_count = int(facts["wave_count"])
    wave_interval = max(8_000, 160_000 // wave_count)
    zombie_list: list[dict[str, int | str]] = []
    proxy_mappings: dict[str, str] = {}
    for wave_index in range(1, wave_count + 1):
        archived = wave_facts.get(wave_index)
        if archived is not None:
            original_names = [str(name) for name in archived["zombies"]]
        else:
            count = 1 + (wave_index - 1) // 5
            original_names = [
                candidate_zombies[(wave_index + offset - 1) % len(candidate_zombies)]
                for offset in range(count)
            ]
        wave_time = 20_000 + (wave_index - 1) * wave_interval
        rows = list(range(5))
        rng.shuffle(rows)
        for offset, original_name in enumerate(original_names):
            proxy_name = _proxy_zombie(original_name)
            if original_name not in _PORT_ZOMBIE_NAMES:
                proxy_mappings[original_name] = proxy_name
            zombie_list.append(
                {
                    "time": wave_time + offset * 750,
                    "map_y": rows[offset % len(rows)],
                    "name": proxy_name,
                    "wave": wave_index,
                }
            )

    starting_sun = 50
    return {
        "stage_id": stage_id,
        "fidelity": "mechanics-reduced-proxy",
        "source_urls": [facts["canonical_url"]],
        "wiki_revision_id": facts["revision_id"],
        "known_constraints": {
            "active_rows": list(range(5)),
            "available_plants": supported_plants,
            "starting_sun": starting_sun,
            "adventure1_wave_count": wave_count,
            "reward_unlock": facts["reward_items"],
            "proxy_zombie_mappings": proxy_mappings,
        },
        "provenance": {
            "database": str(archive),
            "revision_timestamp": facts["revision_timestamp"],
            "content_sha256": facts["content_sha256"],
            "sourced": [
                "location",
                "level_type",
                "adventure1_wave_count",
                "archived_wave_compositions",
                "zombie_types",
                "reward_unlock",
            ],
            "inferred": ["spawn_times", "spawn_rows", "unlisted_wave_fill"],
            "approximated": [
                "unsupported_zombie_mechanics",
                "three-plant_action_surface",
            ],
        },
        "engine": {
            "background_type": 0 if facts["location"] == "Day" else 1,
            "init_sun_value": starting_sun,
            "zombie_list": zombie_list,
        },
    }

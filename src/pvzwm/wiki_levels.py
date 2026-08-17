from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import duckdb

from pvzwm.assets import WIKI_API
from pvzwm.campaign import CampaignStage, campaign_stages

CONTENT_LICENSE = "CC-BY-SA-4.0"
USER_AGENT = "PVZWM/0.1 campaign research (local provenance archive)"


@dataclass(frozen=True, slots=True)
class WikiLevelPage:
    title: str
    canonical_url: str
    revision_id: int
    revision_timestamp: str
    extract: str
    wikitext: str

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.wikitext.encode()).hexdigest()


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def parse_level_query(payload: bytes) -> list[WikiLevelPage]:
    document = json.loads(payload)
    pages = document.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    result: list[WikiLevelPage] = []
    for page in pages:
        if page.get("missing") is not None:
            raise ValueError(f"Wiki page does not exist: {page.get('title')}")
        revisions = page.get("revisions", [])
        if not revisions:
            raise ValueError(f"Wiki page has no revision metadata: {page.get('title')}")
        revision = revisions[0]
        slots = revision.get("slots", {})
        main_slot = slots.get("main", {}) if isinstance(slots, dict) else {}
        result.append(
            WikiLevelPage(
                title=str(page["title"]),
                canonical_url=str(page["canonicalurl"]),
                revision_id=int(revision["revid"]),
                revision_timestamp=str(revision["timestamp"]),
                extract=str(page.get("extract", "")),
                wikitext=str(main_slot.get("content", revision.get("content", ""))),
            )
        )
    return result


def query_level_pages(
    titles: list[str],
    opener: Callable[..., object] = urllib.request.urlopen,
) -> list[WikiLevelPage]:
    pages: list[WikiLevelPage] = []
    for title_batch in _chunks(titles, 20):
        parameters = {
            "action": "query",
            "titles": "|".join(title_batch),
            "prop": "extracts|revisions|info",
            "explaintext": "1",
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "inprop": "url",
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
        }
        request = urllib.request.Request(
            WIKI_API + "?" + urllib.parse.urlencode(parameters),
            headers={"User-Agent": USER_AGENT},
        )
        with opener(request, timeout=30) as response:
            pages.extend(parse_level_query(response.read()))
    return pages


def _stage_title(stage: CampaignStage) -> str:
    return f"Level {stage.stage_id}"


def parse_level_wikitext(wikitext: str) -> dict[str, object]:
    """Extract conservative first-playthrough facts while retaining raw markup."""
    infobox: dict[str, str] = {}
    for key, value in re.findall(r"^\|\s*([A-Za-z]+)\s*=\s*(.*)$", wikitext, re.MULTILINE):
        infobox[key.lower()] = value.strip()

    first_playthrough = ""
    match = re.search(
        r"===\s*Adventure(?: Mode)? 1\s*===(.*?)(?=\n===|\n==[^=]|\Z)",
        wikitext,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        first_playthrough = match.group(1)
    waves: list[dict[str, object]] = []
    for wave_match in re.finditer(
        r"^\|(\d+)z\s*=\s*(.*)$", first_playthrough, re.MULTILINE
    ):
        index = int(wave_match.group(1))
        markup = wave_match.group(2).strip()
        names = re.findall(r"\{\{S\|([^|}]+)", markup, re.IGNORECASE)
        note_match = re.search(
            rf"^\|{index}n\s*=\s*(.*)$", first_playthrough, re.MULTILINE
        )
        waves.append(
            {
                "index": index,
                "zombies": names,
                "note": note_match.group(1).strip() if note_match else None,
                "kind": "explicit",
                "source_markup": markup,
            }
        )
    for row_match in re.finditer(
        r"^\|-\s*\n(?:\|'''\s*(\d+)\s*'''|!\s*(\d+))\s*\n\|(.*?)\n\|(.*?)(?=\n\|-|\n\|})",
        first_playthrough,
        re.MULTILINE | re.DOTALL,
    ):
        index = int(row_match.group(1) or row_match.group(2))
        markup = row_match.group(3).strip()
        waves.append(
            {
                "index": index,
                "zombies": re.findall(r"\{\{S\|([^|}]+)", markup, re.IGNORECASE),
                "note": row_match.group(4).strip() or None,
                "kind": "non_dynamic",
                "source_markup": markup,
            }
        )
    waves.sort(key=lambda wave: int(wave["index"]))

    def template_names(markup: str | None) -> list[str]:
        return (
            []
            if not markup
            else re.findall(r"\{\{S\|([^|}]+)", markup, re.IGNORECASE)
        )

    return {
        "location": infobox.get("loc"),
        "level_type": infobox.get("type"),
        "flags": infobox.get("flag"),
        "plants_markup": infobox.get("plant"),
        "zombies_markup": infobox.get("zombie"),
        "first_reward_markup": infobox.get("fr"),
        "before": infobox.get("before"),
        "after": infobox.get("after"),
        "available_plants": template_names(infobox.get("plant")),
        "zombie_types": template_names(infobox.get("zombie")),
        "first_reward_items": template_names(infobox.get("fr")),
        "adventure1_waves": waves,
    }


def sync_level_wiki(
    database: str | Path = "data/campaign.duckdb",
    *,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> dict[str, object]:
    stages = campaign_stages()
    pages = query_level_pages([_stage_title(stage) for stage in stages], opener)
    pages_by_title = {page.title.replace("_", " "): page for page in pages}
    missing = [
        _stage_title(stage)
        for stage in stages
        if _stage_title(stage) not in pages_by_title
    ]
    if missing:
        raise RuntimeError(f"Wiki response omitted campaign pages: {', '.join(missing)}")

    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    retrieved_at = time.time()
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_level_pages (
                stage_id VARCHAR PRIMARY KEY,
                ordinal INTEGER NOT NULL,
                previous_stage VARCHAR,
                next_stage VARCHAR,
                world INTEGER NOT NULL,
                level INTEGER NOT NULL,
                area VARCHAR NOT NULL,
                rows INTEGER NOT NULL,
                mode VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                canonical_url VARCHAR NOT NULL,
                revision_id BIGINT NOT NULL,
                revision_timestamp VARCHAR NOT NULL,
                retrieved_at DOUBLE NOT NULL,
                content_license VARCHAR NOT NULL,
                content_sha256 VARCHAR NOT NULL,
                extract_text VARCHAR NOT NULL,
                wikitext VARCHAR
            )
            """
        )
        connection.execute(
            "ALTER TABLE wiki_level_pages ADD COLUMN IF NOT EXISTS wikitext VARCHAR"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS wiki_level_facts (
                stage_id VARCHAR PRIMARY KEY,
                location VARCHAR,
                level_type VARCHAR,
                flags VARCHAR,
                plants_markup VARCHAR,
                zombies_markup VARCHAR,
                first_reward_markup VARCHAR,
                before_stage VARCHAR,
                after_stage VARCHAR,
                available_plants JSON,
                zombie_types JSON,
                first_reward_items JSON,
                adventure1_wave_count INTEGER NOT NULL,
                adventure1_waves JSON NOT NULL
            )
            """
        )
        connection.execute(
            "ALTER TABLE wiki_level_facts ADD COLUMN IF NOT EXISTS available_plants JSON"
        )
        connection.execute(
            "ALTER TABLE wiki_level_facts ADD COLUMN IF NOT EXISTS zombie_types JSON"
        )
        connection.execute(
            "ALTER TABLE wiki_level_facts ADD COLUMN IF NOT EXISTS first_reward_items JSON"
        )
        for index, stage in enumerate(stages):
            page = pages_by_title[_stage_title(stage)]
            facts = parse_level_wikitext(page.wikitext)
            waves = facts["adventure1_waves"]
            connection.execute(
                """
                INSERT INTO wiki_level_pages (
                    stage_id, ordinal, previous_stage, next_stage, world, level,
                    area, rows, mode, title, canonical_url, revision_id,
                    revision_timestamp, retrieved_at, content_license,
                    content_sha256, extract_text, wikitext
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                ) ON CONFLICT (stage_id) DO UPDATE SET
                    ordinal = excluded.ordinal,
                    previous_stage = excluded.previous_stage,
                    next_stage = excluded.next_stage,
                    world = excluded.world,
                    level = excluded.level,
                    area = excluded.area,
                    rows = excluded.rows,
                    mode = excluded.mode,
                    title = excluded.title,
                    canonical_url = excluded.canonical_url,
                    revision_id = excluded.revision_id,
                    revision_timestamp = excluded.revision_timestamp,
                    retrieved_at = excluded.retrieved_at,
                    content_license = excluded.content_license,
                    content_sha256 = excluded.content_sha256,
                    extract_text = excluded.extract_text,
                    wikitext = excluded.wikitext
                """,
                [
                    stage.stage_id,
                    index,
                    stages[index - 1].stage_id if index else None,
                    stages[index + 1].stage_id if index + 1 < len(stages) else None,
                    stage.world,
                    stage.level,
                    stage.area,
                    stage.rows,
                    stage.mode,
                    page.title,
                    page.canonical_url,
                    page.revision_id,
                    page.revision_timestamp,
                    retrieved_at,
                    CONTENT_LICENSE,
                    page.content_sha256,
                    page.extract,
                    page.wikitext,
                ],
            )
            connection.execute(
                """
                INSERT INTO wiki_level_facts (
                    stage_id, location, level_type, flags, plants_markup,
                    zombies_markup, first_reward_markup, before_stage,
                    after_stage, available_plants, zombie_types,
                    first_reward_items, adventure1_wave_count, adventure1_waves
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                ) ON CONFLICT (stage_id) DO UPDATE SET
                    location = excluded.location,
                    level_type = excluded.level_type,
                    flags = excluded.flags,
                    plants_markup = excluded.plants_markup,
                    zombies_markup = excluded.zombies_markup,
                    first_reward_markup = excluded.first_reward_markup,
                    before_stage = excluded.before_stage,
                    after_stage = excluded.after_stage,
                    available_plants = excluded.available_plants,
                    zombie_types = excluded.zombie_types,
                    first_reward_items = excluded.first_reward_items,
                    adventure1_wave_count = excluded.adventure1_wave_count,
                    adventure1_waves = excluded.adventure1_waves
                """,
                [
                    stage.stage_id,
                    facts["location"],
                    facts["level_type"],
                    facts["flags"],
                    facts["plants_markup"],
                    facts["zombies_markup"],
                    facts["first_reward_markup"],
                    facts["before"],
                    facts["after"],
                    json.dumps(facts["available_plants"]),
                    json.dumps(facts["zombie_types"]),
                    json.dumps(facts["first_reward_items"]),
                    max((int(wave["index"]) for wave in waves), default=0),
                    json.dumps(waves),
                ],
            )
        count, distinct_revisions = connection.execute(
            "SELECT count(*), count(DISTINCT revision_id) FROM wiki_level_pages"
        ).fetchone()
        structured_stages, documented_waves = connection.execute(
            """
            SELECT count(*) FILTER (WHERE adventure1_wave_count > 0),
                   coalesce(sum(adventure1_wave_count), 0)
            FROM wiki_level_facts
            """
        ).fetchone()
    return {
        "database": str(path),
        "pages": int(count),
        "distinct_revisions": int(distinct_revisions),
        "stages_with_structured_waves": int(structured_stages),
        "documented_adventure1_waves": int(documented_waves),
        "first_stage": stages[0].stage_id,
        "last_stage": stages[-1].stage_id,
        "source": WIKI_API,
        "content_license": CONTENT_LICENSE,
        "retrieved_at_unix": retrieved_at,
    }

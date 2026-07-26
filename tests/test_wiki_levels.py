import json
import urllib.parse
from pathlib import Path

import duckdb

from pvzwm.wiki_levels import (
    CONTENT_LICENSE,
    parse_level_query,
    parse_level_wikitext,
    sync_level_wiki,
)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _page(title: str, revision: int = 7) -> dict[str, object]:
    slug = title.replace(" ", "_")
    return {
        "title": title,
        "canonicalurl": f"https://plantsvszombies.wiki.gg/wiki/{slug}",
        "revisions": [
            {
                "revid": revision,
                "timestamp": "2026-01-02T03:04:05Z",
                "slots": {"main": {"content": f"'''{title}''' test page"}},
            }
        ],
        "extract": f"{title}\nCampaign facts for {title}.",
    }


def test_parse_level_query_preserves_revision_and_hash() -> None:
    pages = parse_level_query(
        json.dumps({"query": {"pages": [_page("Level 1-1")]}}).encode()
    )
    assert pages[0].revision_id == 7
    assert pages[0].content_sha256


def test_parse_level_wikitext_extracts_first_playthrough_waves() -> None:
    facts = parse_level_wikitext(
        """{{Infobox level
|Loc = Day
|Type = Regular
|Flag = One
|Plant = {{S|Peashooter|1}} {{S|Sunflower|1}}
|Zombie = {{S|Zombie|1}} {{S|Flag Zombie|1}}
|FR = {{S|Cherry Bomb|1}}
|before = Level 1-1
|after = Level 1-3
}}
===Adventure Mode 1===
{{PvZ Waves
|1z = {{S|Zombie|1}}
|2z = {{S|Zombie|1}} {{S|Flag Zombie|1}}
|2n = Final wave.
}}
===Adventure Mode 2===
"""
    )
    assert facts["location"] == "Day"
    assert facts["first_reward_markup"] == "{{S|Cherry Bomb|1}}"
    assert facts["adventure1_waves"] == [
        {
            "index": 1,
            "zombies": ["Zombie"],
            "note": None,
            "kind": "explicit",
            "source_markup": "{{S|Zombie|1}}",
        },
        {
            "index": 2,
            "zombies": ["Zombie", "Flag Zombie"],
            "note": "Final wave.",
            "kind": "explicit",
            "source_markup": "{{S|Zombie|1}} {{S|Flag Zombie|1}}",
        },
    ]


def test_sync_level_wiki_stores_order_and_attribution(tmp_path: Path) -> None:
    def opener(request, timeout):
        assert timeout == 30
        query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        titles = query["titles"][0].split("|")
        payload = {"query": {"pages": [_page(title, index + 1) for index, title in enumerate(titles)]}}
        return _Response(json.dumps(payload).encode())

    database = tmp_path / "campaign.duckdb"
    status = sync_level_wiki(database, opener=opener)
    assert status["pages"] == 50
    with duckdb.connect(str(database), read_only=True) as connection:
        first, last, licenses = connection.execute(
            """
            SELECT
                min_by(stage_id, ordinal),
                max_by(stage_id, ordinal),
                count(DISTINCT content_license)
            FROM wiki_level_pages
            """
        ).fetchone()
        next_stage = connection.execute(
            "SELECT next_stage FROM wiki_level_pages WHERE stage_id = '1-10'"
        ).fetchone()[0]
        wikitext = connection.execute(
            "SELECT wikitext FROM wiki_level_pages WHERE stage_id = '1-1'"
        ).fetchone()[0]
    assert (first, last, licenses) == ("1-1", "5-10", 1)
    assert next_stage == "2-1"
    assert "Level 1-1" in wikitext
    assert status["content_license"] == CONTENT_LICENSE

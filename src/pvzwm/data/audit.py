from __future__ import annotations

import json
import subprocess
from pathlib import Path

import duckdb


def discover_macos_pvz() -> list[str]:
    try:
        result = subprocess.run(
            ["mdfind", "kMDItemDisplayName == '*Plants*Zombies*'cd"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return sorted(line for line in result.stdout.splitlines() if line.strip())


def _database_sources(path: Path) -> list[dict[str, object]]:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        if not {"episodes", "transitions"}.issubset(tables):
            return []
        rows = connection.execute(
            """
            SELECT e.source, e.metadata::VARCHAR, count(t.step)
            FROM episodes e LEFT JOIN transitions t USING (episode_id)
            GROUP BY e.episode_id, e.source, e.metadata
            ORDER BY e.source
            """
        ).fetchall()
    finally:
        connection.close()
    results: list[dict[str, object]] = []
    for source, metadata_text, count in rows:
        try:
            metadata = json.loads(metadata_text) if metadata_text else {}
        except json.JSONDecodeError:
            metadata = {"unparsed": metadata_text}
        results.append(
            {
                "source": str(source),
                "metadata": metadata,
                "transitions": int(count),
            }
        )
    return results


def audit_data(data_root: str | Path = "data") -> dict[str, object]:
    root = Path(data_root)
    databases: list[dict[str, object]] = []
    original_transitions = 0
    synthetic_transitions = 0
    python_port_transitions = 0
    for database_path in sorted(root.glob("*.duckdb")):
        try:
            sources = _database_sources(database_path)
            error = None
        except (duckdb.Error, OSError) as caught:
            sources = []
            error = str(caught)
        databases.append(
            {"path": str(database_path), "sources": sources, "error": error}
        )
        for source in sources:
            count = int(source["transitions"])
            metadata = source["metadata"]
            if source["source"] == "symbolic":
                synthetic_transitions += count
            if source["source"] == "python_port":
                python_port_transitions += count
            if (
                source["source"] == "live_screen"
                and isinstance(metadata, dict)
                and metadata.get("game_id") == "original_pvz"
            ):
                original_transitions += count

    raw_root = root / "raw"
    raw_files = [
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ] if raw_root.exists() else []
    reference_files = list((root / "reference").glob("*.json")) if (root / "reference").exists() else []
    installations = discover_macos_pvz()
    return {
        "data_root": str(root),
        "original_game_installations": installations,
        "original_game_transitions": original_transitions,
        "synthetic_transitions": synthetic_transitions,
        "python_port_transitions": python_port_transitions,
        "wiki_asset_files": len(raw_files),
        "reference_metadata_files": len(reference_files),
        "databases": databases,
        "ready_for_original_game_training": original_transitions > 0,
        "ready_for_python_port_training": python_port_transitions > 0,
    }

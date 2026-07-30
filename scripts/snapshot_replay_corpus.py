from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import duckdb


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def inspect_database(path: Path) -> dict[str, object]:
    connection = duckdb.connect(str(path), read_only=True)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                ORDER BY table_name
                """
            ).fetchall()
        ]
        counts = {
            table: int(
                connection.execute(
                    f"SELECT count(*) FROM {quote_identifier(table)}"
                ).fetchone()[0]
            )
            for table in tables
        }
        schema = [
            {
                "table": row[0],
                "column": row[1],
                "type": row[2],
                "nullable": row[3] == "YES",
            }
            for row in connection.execute(
                """
                SELECT table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'main'
                ORDER BY table_name, ordinal_position
                """
            ).fetchall()
        ]
    finally:
        connection.close()
    return {"tables": counts, "schema": schema}


def snapshot_database(
    source: Path,
    destination: Path,
    destination_root: Path,
    remote_repo: str,
) -> dict[str, object]:
    wal = source.with_name(source.name + ".wal")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    wal_bytes = 0
    checkpointed_wal = False
    if wal.is_file():
        destination_wal = destination.with_name(destination.name + ".wal")
        shutil.copy2(wal, destination_wal)
        wal_bytes = wal.stat().st_size
        connection = duckdb.connect(str(destination))
        try:
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        checkpointed_wal = True

    metadata = inspect_database(destination)
    relative = destination.relative_to(destination_root)
    return {
        "path": relative.as_posix(),
        "remote_uri": f"hf://datasets/{remote_repo}/{relative.as_posix()}",
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "source_bytes": source.stat().st_size,
        "source_wal_bytes": wal_bytes,
        "checkpointed_wal": checkpointed_wal,
        **metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lossless, checkpointed DuckDB replay snapshot"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-repo", required=True)
    parser.add_argument("--roots", nargs="+", default=["data", "artifacts"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    destination_root = args.destination.resolve()
    sources = sorted(
        path
        for root in args.roots
        for path in (project_root / root).rglob("*.duckdb")
        if path.is_file()
    )
    if not sources:
        raise SystemExit("No DuckDB replay files found")

    entries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, source in enumerate(sources, start=1):
        relative = source.relative_to(project_root)
        destination = destination_root / relative
        try:
            entry = snapshot_database(
                source,
                destination,
                destination_root,
                args.remote_repo,
            )
            entries.append(entry)
            print(
                f"[{index}/{len(sources)}] {relative} "
                f"{entry['bytes'] / 2**20:.1f} MiB"
            )
        except Exception as error:
            failures.append({"path": relative.as_posix(), "error": str(error)})
            print(f"[{index}/{len(sources)}] FAILED {relative}: {error}")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "remote_repo": args.remote_repo,
        "source_roots": args.roots,
        "database_count": len(entries),
        "database_bytes": sum(int(entry["bytes"]) for entry in entries),
        "checkpointed_wal_count": sum(
            bool(entry["checkpointed_wal"]) for entry in entries
        ),
        "source_wal_bytes": sum(int(entry["source_wal_bytes"]) for entry in entries),
        "failures": failures,
        "databases": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} databases failed")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pvzwm.data.store import ReplayStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk-copy a replay into the compact DuckDB layout without decoding arrays"
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Source replay does not exist: {args.source}")
    if args.destination.exists():
        raise SystemExit(f"Destination already exists: {args.destination}")

    with ReplayStore(args.destination) as store:
        counts = store.merge_from([args.source])
        store.connection.execute("CHECKPOINT")
        database_size = store.connection.execute("PRAGMA database_size").fetchall()
        storage = store.connection.execute(
            """
            SELECT column_name, compression, count(*) AS chunks, sum(count) AS rows
            FROM pragma_storage_info('transitions')
            GROUP BY column_name, compression
            ORDER BY column_name, compression
            """
        ).fetchall()
        nullable_next = store.connection.execute(
            """
            SELECT count(*) FILTER (WHERE next_observation IS NULL), count(*)
            FROM transitions
            """
        ).fetchone()
    print(
        json.dumps(
            {
                "source": str(args.source),
                "destination": str(args.destination),
                "counts": counts,
                "source_bytes": args.source.stat().st_size,
                "destination_bytes": args.destination.stat().st_size,
                "database_size": database_size,
                "next_observation_nulls": nullable_next,
                "storage": storage,
            }
        )
    )


if __name__ == "__main__":
    main()

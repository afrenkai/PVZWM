from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PVZWM replay DuckDB")
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    connection = duckdb.connect(str(args.database), read_only=True)
    try:
        episodes, transitions = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM episodes),
                (SELECT count(*) FROM transitions)
            """
        ).fetchone()
        sources = connection.execute(
            """
            SELECT source, count(*), coalesce(sum(length), 0)
            FROM episodes
            GROUP BY source
            ORDER BY source
            """
        ).fetchall()
        levels = connection.execute(
            """
            SELECT
                coalesce(json_extract_string(metadata, '$.level_id'), 'unknown'),
                count(*),
                coalesce(sum(length), 0),
                count(*) FILTER (
                    WHERE coalesce(json_extract_string(metadata, '$.won'), 'false')
                        = 'true'
                ),
                coalesce(avg(return), 0),
                coalesce(avg(length), 0),
                count(*) FILTER (
                    WHERE coalesce(
                        json_extract_string(metadata, '$.truncated'), 'false'
                    ) = 'true'
                ),
                min(json_extract_string(metadata, '$.observation_shape')),
                min(json_extract_string(metadata, '$.action_semantics'))
            FROM episodes
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchall()
    finally:
        connection.close()
    print(
        json.dumps(
            {
                "database": str(args.database),
                "episodes": int(episodes),
                "transitions": int(transitions),
                "sources": {
                    source: {"episodes": int(count), "transitions": int(length)}
                    for source, count, length in sources
                },
                "levels": {
                    level: {
                        "episodes": int(count),
                        "transitions": int(length),
                        "wins": int(wins),
                        "mean_return": float(mean_return),
                        "mean_length": float(mean_length),
                        "truncations": int(truncations),
                        "observation_shape": observation_shape,
                        "action_semantics": action_semantics,
                    }
                    for (
                        level,
                        count,
                        length,
                        wins,
                        mean_return,
                        mean_length,
                        truncations,
                        observation_shape,
                        action_semantics,
                    ) in levels
                },
            }
        )
    )


if __name__ == "__main__":
    main()

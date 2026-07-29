from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify compact replay equivalence without decoding array BLOBs"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if not args.source.is_file() or not args.candidate.is_file():
        raise SystemExit("Both replay databases must exist")

    connection = duckdb.connect()
    try:
        source = str(args.source.resolve()).replace("'", "''")
        candidate = str(args.candidate.resolve()).replace("'", "''")
        connection.execute(f"ATTACH '{source}' AS source (READ_ONLY)")
        connection.execute(f"ATTACH '{candidate}' AS candidate (READ_ONLY)")
        episode_mismatches = connection.execute(
            """
            SELECT count(*) FROM (
                (SELECT * FROM source.episodes EXCEPT SELECT * FROM candidate.episodes)
                UNION ALL
                (SELECT * FROM candidate.episodes EXCEPT SELECT * FROM source.episodes)
            )
            """
        ).fetchone()[0]
        experiment_mismatches = connection.execute(
            """
            SELECT count(*) FROM (
                (SELECT * FROM source.experiments EXCEPT SELECT * FROM candidate.experiments)
                UNION ALL
                (SELECT * FROM candidate.experiments EXCEPT SELECT * FROM source.experiments)
            )
            """
        ).fetchone()[0]
        transition_mismatches = connection.execute(
            """
            WITH resolved_source AS (
                SELECT episode_id, step, observation, action, reward,
                       coalesce(
                           lead(observation) OVER (
                               PARTITION BY episode_id ORDER BY step
                           ),
                           next_observation
                       ) AS next_observation,
                       terminated, truncated, legal_mask
                FROM source.transitions
            ),
            resolved_candidate AS (
                SELECT episode_id, step, observation, action, reward,
                       coalesce(
                           lead(observation) OVER (
                               PARTITION BY episode_id ORDER BY step
                           ),
                           next_observation
                       ) AS next_observation,
                       terminated, truncated, legal_mask
                FROM candidate.transitions
            )
            SELECT count(*)
            FROM resolved_source source
            FULL OUTER JOIN resolved_candidate candidate
                USING (episode_id, step)
            WHERE source.episode_id IS NULL
               OR candidate.episode_id IS NULL
               OR source.observation IS DISTINCT FROM candidate.observation
               OR source.action IS DISTINCT FROM candidate.action
               OR source.reward IS DISTINCT FROM candidate.reward
               OR source.next_observation IS DISTINCT FROM candidate.next_observation
               OR source.terminated IS DISTINCT FROM candidate.terminated
               OR source.truncated IS DISTINCT FROM candidate.truncated
               OR source.legal_mask IS DISTINCT FROM candidate.legal_mask
            """
        ).fetchone()[0]
    finally:
        connection.close()

    result = {
        "source": str(args.source),
        "candidate": str(args.candidate),
        "episode_mismatches": int(episode_mismatches),
        "experiment_mismatches": int(experiment_mismatches),
        "transition_mismatches": int(transition_mismatches),
    }
    print(json.dumps(result))
    if any(result[key] for key in result if key.endswith("_mismatches")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import io
import json
import time
import uuid
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import duckdb
import numpy as np


_COMPRESSED_ARRAY_MAGIC = b"PVZWM-ZLIB-1\0"
_REPLAY_ROW_GROUP_SIZE = 4_096


def _pack_array(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    raw = buffer.getvalue()
    if len(raw) < 4_096:
        return raw
    compressed = zlib.compress(raw, level=3)
    if len(compressed) + len(_COMPRESSED_ARRAY_MAGIC) >= len(raw):
        return raw
    return _COMPRESSED_ARRAY_MAGIC + compressed


def _unpack_array(value: bytes) -> np.ndarray:
    if value.startswith(_COMPRESSED_ARRAY_MAGIC):
        value = zlib.decompress(value[len(_COMPRESSED_ARRAY_MAGIC) :])
    return np.load(io.BytesIO(value), allow_pickle=False)


@dataclass(frozen=True, slots=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool = False
    legal_mask: np.ndarray | None = None
    next_legal_mask: np.ndarray | None = None


class ReplayStore:
    """DuckDB-backed episode metadata and transition replay."""

    def __init__(self, path: str | Path = "data/replay.duckdb", read_only: bool = False) -> None:
        self.path = Path(path)
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        if not read_only and is_new:
            escaped_path = str(self.path.resolve()).replace("'", "''")
            self.connection = duckdb.connect()
            self.connection.execute(
                f"ATTACH '{escaped_path}' AS replay_data "
                f"(ROW_GROUP_SIZE {_REPLAY_ROW_GROUP_SIZE})"
            )
            self.connection.execute("USE replay_data")
        else:
            self.connection = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            self.connection.execute("SET checkpoint_threshold = '256MB'")
            self._initialize()
        columns = self.connection.execute("PRAGMA table_info('transitions')").fetchall()
        next_column = next(row for row in columns if row[1] == "next_observation")
        self._compact_next_observation = not bool(next_column[3])

    def _initialize(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id VARCHAR PRIMARY KEY,
                source VARCHAR NOT NULL,
                seed BIGINT,
                started_at DOUBLE NOT NULL,
                ended_at DOUBLE,
                return DOUBLE,
                length BIGINT,
                metadata JSON
            );
            CREATE TABLE IF NOT EXISTS transitions (
                episode_id VARCHAR NOT NULL,
                step BIGINT NOT NULL,
                observation BLOB NOT NULL,
                action BIGINT NOT NULL,
                reward DOUBLE NOT NULL,
                next_observation BLOB,
                terminated BOOLEAN NOT NULL,
                truncated BOOLEAN NOT NULL,
                legal_mask BLOB
            );
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id VARCHAR PRIMARY KEY,
                created_at DOUBLE NOT NULL,
                model VARCHAR NOT NULL,
                agent VARCHAR,
                seed BIGINT,
                device VARCHAR,
                config JSON NOT NULL,
                metrics JSON,
                git_revision VARCHAR
            );
            DROP INDEX IF EXISTS transitions_episode_idx;
            """
        )

    def episode(self, source: str, seed: int, metadata: dict[str, object] | None = None) -> EpisodeWriter:
        return EpisodeWriter(self, source, seed, metadata or {})

    def count_transitions(self) -> int:
        return int(self.connection.execute("SELECT count(*) FROM transitions").fetchone()[0])

    def infer_action_dim(
        self,
        episode_ids: list[str] | None = None,
        fallback: int | None = None,
    ) -> int:
        """Infer a replay vocabulary from legal masks and observed actions."""

        clause, params = self._episode_clause(episode_ids)
        maximum = self.connection.execute(
            "SELECT max(action) FROM transitions" + clause,
            params,
        ).fetchone()[0]
        if maximum is None:
            raise ValueError("Cannot infer an action vocabulary from empty replay")
        observed_dim = int(maximum) + 1

        mask_query = "SELECT legal_mask FROM transitions WHERE legal_mask IS NOT NULL"
        mask_params: list[object] = []
        if episode_ids is not None:
            if not episode_ids:
                raise ValueError("Cannot infer an action vocabulary from empty replay")
            placeholders = ", ".join("?" for _ in episode_ids)
            mask_query += f" AND episode_id IN ({placeholders})"
            mask_params.extend(episode_ids)
        mask_query += " LIMIT 1"
        mask_row = self.connection.execute(mask_query, mask_params).fetchone()
        if mask_row is not None:
            mask = _unpack_array(mask_row[0])
            if mask.ndim != 1 or mask.size < 1:
                raise ValueError("Replay legal-action mask must be a nonempty vector")
            action_dim = int(mask.size)
            if observed_dim > action_dim:
                raise ValueError(
                    f"Replay action {observed_dim - 1} exceeds legal-mask size {action_dim}"
                )
            return action_dim
        if fallback is None:
            return observed_dim
        if fallback < 1:
            raise ValueError("Action-dimension fallback must be positive")
        return max(observed_dim, fallback)

    def merge_from(self, sources: list[str | Path]) -> dict[str, int]:
        """Copy complete shards without decoding arrays, compacting adjacent frames."""
        totals = {"databases": 0, "episodes": 0, "transitions": 0, "experiments": 0}
        destination = self.path.resolve()
        for index, source_value in enumerate(sources):
            source = Path(source_value).resolve()
            if source == destination:
                raise ValueError("A replay database cannot be merged into itself")
            if not source.is_file():
                raise FileNotFoundError(source)
            alias = f"replay_shard_{index}"
            escaped_source = str(source).replace("'", "''")
            self.connection.execute(
                f"ATTACH '{escaped_source}' AS {alias} (READ_ONLY)"
            )
            try:
                episode_conflicts = int(
                    self.connection.execute(
                        f"""
                        SELECT count(*) FROM episodes destination
                        JOIN {alias}.episodes source USING (episode_id)
                        """
                    ).fetchone()[0]
                )
                experiment_conflicts = int(
                    self.connection.execute(
                        f"""
                        SELECT count(*) FROM experiments destination
                        JOIN {alias}.experiments source USING (experiment_id)
                        """
                    ).fetchone()[0]
                )
                if episode_conflicts or experiment_conflicts:
                    raise ValueError(
                        f"Replay shard {source} contains identifiers already in the destination"
                    )
                counts = {
                    name: int(
                        self.connection.execute(
                            f"SELECT count(*) FROM {alias}.{name}"
                        ).fetchone()[0]
                    )
                    for name in ("episodes", "transitions", "experiments")
                }
                self.connection.execute("BEGIN TRANSACTION")
                try:
                    self.connection.execute(
                        f"INSERT INTO episodes SELECT * FROM {alias}.episodes"
                    )
                    if self._compact_next_observation:
                        self.connection.execute(
                            f"""
                            INSERT INTO transitions
                            SELECT episode_id, step, observation, action, reward,
                                   CASE WHEN lead(step) OVER (
                                       PARTITION BY episode_id ORDER BY step
                                   ) IS NULL THEN next_observation ELSE NULL END,
                                   terminated, truncated, legal_mask
                            FROM {alias}.transitions
                            """
                        )
                    else:
                        self.connection.execute(
                            f"INSERT INTO transitions SELECT * FROM {alias}.transitions"
                        )
                    self.connection.execute(
                        f"INSERT INTO experiments SELECT * FROM {alias}.experiments"
                    )
                    self.connection.execute("COMMIT")
                except Exception:
                    self.connection.execute("ROLLBACK")
                    raise
                totals["databases"] += 1
                for name, count in counts.items():
                    totals[name] += count
            finally:
                self.connection.execute(f"DETACH {alias}")
        return totals

    def record_experiment(
        self,
        model: str,
        config: dict[str, object],
        metrics: dict[str, object],
        *,
        agent: str | None = None,
        seed: int | None = None,
        device: str | None = None,
        git_revision: str | None = None,
    ) -> str:
        experiment_id = str(uuid.uuid4())
        self.connection.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                experiment_id,
                time.time(),
                model,
                agent,
                seed,
                device,
                json.dumps(config),
                json.dumps(metrics),
                git_revision,
            ],
        )
        return experiment_id

    def episode_ids(self, source: str | None = None) -> list[str]:
        query = "SELECT episode_id FROM episodes"
        params: list[object] = []
        if source is not None:
            query += " WHERE source = ?"
            params.append(source)
        query += " ORDER BY started_at, episode_id"
        return [str(row[0]) for row in self.connection.execute(query, params).fetchall()]

    def successful_episode_ids(self, source: str | None = None) -> list[str]:
        """Return completed successes, preferring an explicit environment outcome.

        Positive shaped return is only a legacy fallback. Long time-limit episodes
        can accumulate positive progress reward without winning, so an explicit
        ``metadata.won`` value always takes precedence when it is present.
        """
        query = "SELECT episode_id, return, metadata FROM episodes WHERE return IS NOT NULL"
        params: list[object] = []
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY started_at, episode_id"
        successful: list[str] = []
        for episode_id, episode_return, raw_metadata in self.connection.execute(
            query, params
        ).fetchall():
            metadata = (
                json.loads(raw_metadata)
                if isinstance(raw_metadata, str)
                else raw_metadata
            )
            if isinstance(metadata, dict) and "won" in metadata:
                is_success = metadata["won"] is True
            else:
                is_success = float(episode_return) > 0.0
            if is_success:
                successful.append(str(episode_id))
        return successful

    def top_episode_ids(
        self,
        fraction: float,
        source: str | None = None,
    ) -> list[str]:
        """Return the highest-return completed episodes for elite policy iteration."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        count_query = "SELECT count(*) FROM episodes WHERE return IS NOT NULL"
        params: list[object] = []
        if source is not None:
            count_query += " AND source = ?"
            params.append(source)
        count = int(self.connection.execute(count_query, params).fetchone()[0])
        if count == 0:
            return []
        limit = max(1, int(np.ceil(count * fraction)))
        query = "SELECT episode_id FROM episodes WHERE return IS NOT NULL"
        if source is not None:
            query += " AND source = ?"
        query += " ORDER BY return DESC, ended_at, episode_id LIMIT ?"
        return [
            str(row[0])
            for row in self.connection.execute(query, [*params, limit]).fetchall()
        ]

    def split_episode_ids(
        self,
        source: str | None = None,
        validation_fraction: float = 0.2,
        seed: int = 0,
    ) -> tuple[list[str], list[str]]:
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        episode_ids = self.episode_ids(source)
        if len(episode_ids) < 2 or validation_fraction == 0.0:
            return episode_ids, []
        rng = np.random.default_rng(seed)
        shuffled = [episode_ids[int(index)] for index in rng.permutation(len(episode_ids))]
        validation_count = max(1, round(len(episode_ids) * validation_fraction))
        validation_count = min(validation_count, len(episode_ids) - 1)
        validation = set(shuffled[:validation_count])
        return (
            [episode_id for episode_id in episode_ids if episode_id not in validation],
            [episode_id for episode_id in episode_ids if episode_id in validation],
        )

    def split_episode_ids_grouped(
        self,
        metadata_key: str,
        source: str | None = None,
        validation_fraction: float = 0.2,
        seed: int = 0,
    ) -> tuple[list[str], list[str]]:
        """Split whole metadata groups so scenarios cannot leak across folds."""

        if not metadata_key:
            raise ValueError("A nonempty metadata key is required")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        query = "SELECT episode_id, metadata FROM episodes"
        params: list[object] = []
        if source is not None:
            query += " WHERE source = ?"
            params.append(source)
        query += " ORDER BY started_at, episode_id"
        rows = self.connection.execute(query, params).fetchall()
        groups: dict[str, list[str]] = {}
        ordered_ids: list[str] = []
        for episode_id, raw_metadata in rows:
            metadata = (
                json.loads(raw_metadata)
                if isinstance(raw_metadata, str)
                else raw_metadata
            )
            if not isinstance(metadata, dict) or metadata_key not in metadata:
                raise ValueError(
                    f"Episode {episode_id} has no metadata group {metadata_key!r}"
                )
            group = json.dumps(metadata[metadata_key], sort_keys=True)
            groups.setdefault(group, []).append(str(episode_id))
            ordered_ids.append(str(episode_id))
        if len(groups) < 2 or validation_fraction == 0.0:
            return ordered_ids, []
        rng = np.random.default_rng(seed)
        group_names = list(groups)
        validation_count = max(1, round(len(group_names) * validation_fraction))
        validation_count = min(validation_count, len(group_names) - 1)
        validation_groups = {
            group_names[int(index)]
            for index in rng.permutation(len(group_names))[:validation_count]
        }
        validation_ids = {
            episode_id
            for group in validation_groups
            for episode_id in groups[group]
        }
        return (
            [episode_id for episode_id in ordered_ids if episode_id not in validation_ids],
            [episode_id for episode_id in ordered_ids if episode_id in validation_ids],
        )

    @staticmethod
    def _episode_clause(episode_ids: list[str] | None) -> tuple[str, list[object]]:
        if episode_ids is None:
            return "", []
        if not episode_ids:
            return " WHERE FALSE", []
        placeholders = ", ".join("?" for _ in episode_ids)
        return f" WHERE episode_id IN ({placeholders})", list(episode_ids)

    def transitions(
        self,
        limit: int | None = None,
        offset: int = 0,
        episode_ids: list[str] | None = None,
    ) -> Iterator[Transition]:
        clause, episode_params = self._episode_clause(episode_ids)
        query = f"""
            SELECT observation, action, reward,
                   coalesce(following_observation, stored_next_observation),
                   terminated, truncated, legal_mask, next_legal_mask
            FROM (
                SELECT observation, action, reward,
                       next_observation AS stored_next_observation,
                       lead(observation) OVER (
                           PARTITION BY episode_id ORDER BY step
                       ) AS following_observation,
                       terminated, truncated, legal_mask,
                       lead(legal_mask) OVER (
                           PARTITION BY episode_id ORDER BY step
                       ) AS next_legal_mask,
                       episode_id, step
                FROM transitions{clause}
            ) resolved
            ORDER BY episode_id, step
        """
        params: list[object] = episode_params
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = self.connection.execute(query, params).fetchall()
        for (
            obs,
            action,
            reward,
            next_obs,
            terminated,
            truncated,
            mask,
            next_mask,
        ) in rows:
            if next_obs is None:
                raise ValueError("Replay transition has no resolvable next observation")
            yield Transition(
                _unpack_array(obs),
                int(action),
                float(reward),
                _unpack_array(next_obs),
                bool(terminated),
                bool(truncated),
                None if mask is None else _unpack_array(mask),
                None if next_mask is None else _unpack_array(next_mask),
            )

    def transition_sequences(
        self,
        length: int,
        stride: int = 1,
        episode_ids: list[str] | None = None,
    ) -> Iterator[list[Transition]]:
        """Yield fixed windows without crossing episode boundaries."""
        if length < 1 or stride < 1:
            raise ValueError("length and stride must be positive")
        query = "SELECT episode_id FROM episodes WHERE length >= ?"
        params: list[object] = [length]
        if episode_ids is not None:
            if not episode_ids:
                return
            placeholders = ", ".join("?" for _ in episode_ids)
            query += f" AND episode_id IN ({placeholders})"
            params.extend(episode_ids)
        query += " ORDER BY started_at, episode_id"
        selected_ids = self.connection.execute(query, params).fetchall()
        for (episode_id,) in selected_ids:
            rows = self.connection.execute(
                """
                SELECT observation, action, reward, next_observation,
                       terminated, truncated, legal_mask
                FROM transitions WHERE episode_id = ? ORDER BY step
                """,
                [episode_id],
            ).fetchall()
            resolved_next = [
                rows[index + 1][0]
                if index + 1 < len(rows)
                else rows[index][3]
                for index in range(len(rows))
            ]
            if any(value is None for value in resolved_next):
                raise ValueError(f"Episode {episode_id} has no final next observation")
            episode = [
                Transition(
                    _unpack_array(obs),
                    int(action),
                    float(reward),
                    _unpack_array(resolved_next[index]),
                    bool(terminated),
                    bool(truncated),
                    None if mask is None else _unpack_array(mask),
                )
                for index, (
                    obs,
                    action,
                    reward,
                    _next_obs,
                    terminated,
                    truncated,
                    mask,
                ) in enumerate(rows)
            ]
            for index in range(len(episode) - 1):
                episode[index] = replace(
                    episode[index], next_legal_mask=episode[index + 1].legal_mask
                )
            for start in range(0, len(episode) - length + 1, stride):
                yield episode[start : start + length]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ReplayStore:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


class EpisodeWriter:
    def __init__(
        self,
        store: ReplayStore,
        source: str,
        seed: int,
        metadata: dict[str, object],
    ) -> None:
        self.store = store
        self.episode_id = str(uuid.uuid4())
        self.source = source
        self.seed = seed
        self.metadata = metadata
        self.started_at = time.time()
        self.length = 0
        self.total_return = 0.0
        self._closed = False
        self._last_next_observation: bytes | None = None
        self.store.connection.execute("BEGIN TRANSACTION")
        self.store.connection.execute(
            "INSERT INTO episodes VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?)",
            [self.episode_id, source, seed, self.started_at, json.dumps(metadata)],
        )

    def append(self, transition: Transition) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed episode")
        packed_next = _pack_array(transition.next_observation)
        self._last_next_observation = packed_next
        stored_next = (
            packed_next
            if (
                not self.store._compact_next_observation
                or transition.terminated
                or transition.truncated
            )
            else None
        )
        self.store.connection.execute(
            "INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                self.episode_id,
                self.length,
                _pack_array(transition.observation),
                transition.action,
                transition.reward,
                stored_next,
                transition.terminated,
                transition.truncated,
                None if transition.legal_mask is None else _pack_array(transition.legal_mask),
            ],
        )
        self.length += 1
        self.total_return += transition.reward

    def close(self) -> None:
        if self._closed:
            return
        if (
            self.store._compact_next_observation
            and self.length
            and self._last_next_observation is not None
        ):
            self.store.connection.execute(
                """
                UPDATE transitions SET next_observation = ?
                WHERE episode_id = ? AND step = ? AND next_observation IS NULL
                """,
                [self._last_next_observation, self.episode_id, self.length - 1],
            )
        self.store.connection.execute(
            """
            UPDATE episodes SET ended_at = ?, return = ?, length = ?, metadata = ?
            WHERE episode_id = ?
            """,
            [
                time.time(),
                self.total_return,
                self.length,
                json.dumps(self.metadata),
                self.episode_id,
            ],
        )
        self.store.connection.execute("COMMIT")
        self._closed = True

    def __enter__(self) -> EpisodeWriter:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if exc_type is not None and self.length:
            self.store.connection.execute(
                """
                UPDATE transitions SET truncated = TRUE
                WHERE episode_id = ? AND step = ? AND NOT terminated
                """,
                [self.episode_id, self.length - 1],
            )
        self.close()

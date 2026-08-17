from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from pathlib import Path

import duckdb
import numpy as np

from pvzwm.data.store import _unpack_array


def raw_numpy_blob(blob: bytes) -> bytes:
    array = _unpack_array(blob)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def create_and_measure(path: Path, blobs: list[bytes]) -> dict[str, object]:
    if path.exists():
        raise SystemExit(f"Benchmark output already exists: {path}")
    connection = duckdb.connect(str(path))
    try:
        connection.execute("SET checkpoint_threshold = '256MB'")
        connection.execute("CREATE TABLE payloads (payload BLOB NOT NULL)")
        connection.execute("BEGIN TRANSACTION")
        connection.executemany("INSERT INTO payloads VALUES (?)", [(blob,) for blob in blobs])
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
        storage = connection.execute(
            """
            SELECT compression, count(*) AS chunks, sum(count) AS rows
            FROM pragma_storage_info('payloads')
            GROUP BY compression ORDER BY compression
            """
        ).fetchall()
    finally:
        connection.close()

    durations: list[float] = []
    for _ in range(3):
        connection = duckdb.connect(str(path), read_only=True)
        try:
            started = time.perf_counter()
            rows = connection.execute("SELECT payload FROM payloads").fetchall()
            for (blob,) in rows:
                _unpack_array(blob)
            durations.append(time.perf_counter() - started)
        finally:
            connection.close()
    return {
        "format": "duckdb",
        "path": str(path),
        "bytes": path.stat().st_size,
        "storage": storage,
        "decode_seconds": durations,
        "median_decode_seconds": statistics.median(durations),
    }


def create_parquet_and_measure(path: Path, blobs: list[bytes]) -> dict[str, object]:
    if path.exists():
        raise SystemExit(f"Benchmark output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute("CREATE TABLE payloads (payload BLOB NOT NULL)")
        connection.execute("BEGIN TRANSACTION")
        connection.executemany("INSERT INTO payloads VALUES (?)", [(blob,) for blob in blobs])
        connection.execute("COMMIT")
        escaped_path = str(path.resolve()).replace("'", "''")
        connection.execute(
            f"COPY payloads TO '{escaped_path}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 4096)"
        )
    finally:
        connection.close()

    durations: list[float] = []
    for _ in range(3):
        connection = duckdb.connect()
        try:
            started = time.perf_counter()
            rows = connection.execute(
                "SELECT payload FROM read_parquet(?)", [str(path)]
            ).fetchall()
            for (blob,) in rows:
                _unpack_array(blob)
            durations.append(time.perf_counter() - started)
        finally:
            connection.close()
    return {
        "format": "parquet_zstd",
        "path": str(path),
        "bytes": path.stat().st_size,
        "decode_seconds": durations,
        "median_decode_seconds": statistics.median(durations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Python-zlib and DuckDB-native array BLOB storage remotely"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--limit", type=int, default=1_000)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("Limit must be positive")
    source = duckdb.connect(str(args.source), read_only=True)
    try:
        compressed = [
            bytes(row[0])
            for row in source.execute(
                """
                SELECT observation FROM transitions
                ORDER BY episode_id, step LIMIT ?
                """,
                [args.limit],
            ).fetchall()
        ]
    finally:
        source.close()
    raw = [raw_numpy_blob(blob) for blob in compressed]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "rows": len(compressed),
        "python_zlib": create_and_measure(
            Path(f"{args.output_prefix}-python-zlib.duckdb"), compressed
        ),
        "duckdb_native": create_and_measure(
            Path(f"{args.output_prefix}-duckdb-native.duckdb"), raw
        ),
        "parquet_python_zlib": create_parquet_and_measure(
            Path(f"{args.output_prefix}-python-zlib.parquet"), compressed
        ),
        "parquet_raw_zstd": create_parquet_and_measure(
            Path(f"{args.output_prefix}-raw-zstd.parquet"), raw
        ),
    }
    print(json.dumps(results))


if __name__ == "__main__":
    main()

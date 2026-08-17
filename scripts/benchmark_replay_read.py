from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

from pvzwm.data.dataset import SequenceDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark full episode-safe sequence materialization from DuckDB"
    )
    parser.add_argument("databases", type=Path, nargs="+")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("Repeats must be positive")

    results: list[dict[str, object]] = []
    for database in args.databases:
        durations: list[float] = []
        windows = 0
        for _ in range(args.repeats):
            started = time.perf_counter()
            dataset = SequenceDataset(
                database,
                length=args.sequence_length,
                stride=args.sequence_stride,
            )
            windows = len(dataset)
            durations.append(time.perf_counter() - started)
            del dataset
            gc.collect()
        results.append(
            {
                "database": str(database),
                "bytes": database.stat().st_size,
                "windows": windows,
                "seconds": durations,
                "median_seconds": statistics.median(durations),
            }
        )
    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()

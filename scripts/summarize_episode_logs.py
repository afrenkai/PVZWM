from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize JSONL episode records")
    parser.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.logs:
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not records:
            raise SystemExit(f"Episode log is empty: {path}")
        print(
            json.dumps(
                {
                    "log": str(path),
                    "episodes": len(records),
                    "wins": sum(bool(record["won"]) for record in records),
                    "win_rate": float(np.mean([record["won"] for record in records])),
                    "mean_return": float(
                        np.mean([record["return"] for record in records])
                    ),
                    "mean_length": float(
                        np.mean([record["length"] for record in records])
                    ),
                    "max_return": float(max(record["return"] for record in records)),
                }
            )
        )


if __name__ == "__main__":
    main()

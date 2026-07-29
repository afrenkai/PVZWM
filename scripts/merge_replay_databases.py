from __future__ import annotations

import argparse
import json
from pathlib import Path

from pvzwm.data.store import ReplayStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge complete PVZWM replay shards without decoding arrays"
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with ReplayStore(args.destination) as store:
        result = store.merge_from(args.sources)
    print(
        json.dumps(
            {
                "destination": str(args.destination),
                "sources": [str(source) for source in args.sources],
                **result,
            }
        )
    )


if __name__ == "__main__":
    main()

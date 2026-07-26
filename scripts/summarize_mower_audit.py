from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def load_records(paths: list[Path]) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            seed = int(record["seed"])
            if seed in records:
                raise SystemExit(f"Duplicate seed {seed} in mower audit")
            records[seed] = record
    return records


def summarize(records: dict[int, dict[str, object]]) -> dict[str, object]:
    values = list(records.values())
    wins = [record for record in values if bool(record["won"])]
    mower_episodes = [
        record for record in values if int(record["mowers_triggered"]) > 0
    ]
    return {
        "episodes": len(values),
        "wins": len(wins),
        "win_rate": len(wins) / len(values) if values else None,
        "mean_return": mean(float(record["return"]) for record in values)
        if values
        else None,
        "median_return": median(float(record["return"]) for record in values)
        if values
        else None,
        "mean_length": mean(int(record["length"]) for record in values)
        if values
        else None,
        "episodes_using_mowers": len(mower_episodes),
        "wins_using_mowers": sum(bool(record["won"]) for record in mower_episodes),
        "wins_without_mower_trigger": sum(
            bool(record["won"]) and int(record["mowers_triggered"]) == 0
            for record in values
        ),
        "total_mowers_triggered": sum(
            int(record["mowers_triggered"]) for record in values
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize matched normal and no-mower PvZ evaluations"
    )
    parser.add_argument("--normal", nargs="+", type=Path, required=True)
    parser.add_argument("--disabled", nargs="+", type=Path, required=True)
    args = parser.parse_args()

    normal = load_records(args.normal)
    disabled = load_records(args.disabled)
    if set(normal) != set(disabled):
        missing_normal = sorted(set(disabled) - set(normal))
        missing_disabled = sorted(set(normal) - set(disabled))
        raise SystemExit(
            "Mower audit seeds differ: "
            f"missing normal={missing_normal}, missing disabled={missing_disabled}"
        )

    pairs = {
        "both_win": 0,
        "normal_only_win": 0,
        "disabled_only_win": 0,
        "both_lose": 0,
    }
    for seed in sorted(normal):
        normal_win = bool(normal[seed]["won"])
        disabled_win = bool(disabled[seed]["won"])
        if normal_win and disabled_win:
            pairs["both_win"] += 1
        elif normal_win:
            pairs["normal_only_win"] += 1
        elif disabled_win:
            pairs["disabled_only_win"] += 1
        else:
            pairs["both_lose"] += 1

    normal_mower_seeds = [
        seed
        for seed in sorted(normal)
        if int(normal[seed]["mowers_triggered"]) > 0
    ]

    print(
        json.dumps(
            {
                "seeds": sorted(normal),
                "normal": summarize(normal),
                "disabled": summarize(disabled),
                "paired_outcomes": pairs,
                "normal_mower_triggered_pairs": {
                    "seeds": normal_mower_seeds,
                    "episodes": len(normal_mower_seeds),
                    "normal_wins": sum(
                        bool(normal[seed]["won"]) for seed in normal_mower_seeds
                    ),
                    "disabled_wins": sum(
                        bool(disabled[seed]["won"]) for seed in normal_mower_seeds
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

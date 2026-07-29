from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Iterable

import duckdb

from pvzwm.data.store import ReplayStore
from pvzwm.portable import PortableActionCodec


def plant_action_distribution(
    actions: Iterable[int], codec: PortableActionCodec
) -> dict[str, object]:
    slots: Counter[int] = Counter()
    rows: Counter[int] = Counter()
    cols: Counter[int] = Counter()
    slot_rows: Counter[tuple[int, int]] = Counter()
    cells: set[tuple[int, int]] = set()
    total = 0
    for action in actions:
        if not codec.plant_begin <= action < codec.shovel_begin:
            continue
        decoded = codec.decode(action)
        slots[decoded.slot] += 1
        rows[decoded.row] += 1
        cols[decoded.col] += 1
        slot_rows[(decoded.slot, decoded.row)] += 1
        cells.add((decoded.row, decoded.col))
        total += 1

    def histogram(counter: Counter[int], size: int) -> list[dict[str, float | int]]:
        return [
            {
                "index": index,
                "count": counter[index],
                "fraction": counter[index] / total if total else 0.0,
            }
            for index in range(size)
        ]

    return {
        "total": total,
        "distinct_cells": len(cells),
        "by_slot": histogram(slots, codec.seed_slots),
        "by_row": histogram(rows, codec.rows),
        "by_col": histogram(cols, codec.cols),
        "by_slot_row": [
            {
                "slot": slot,
                "row": row,
                "count": slot_rows[(slot, row)],
                "fraction": slot_rows[(slot, row)] / total if total else 0.0,
            }
            for slot in range(codec.seed_slots)
            for row in range(codec.rows)
        ],
    }


def board_click_alignment(database: Path, codec: PortableActionCodec) -> dict[str, float | int]:
    cells = codec.cells
    zombie_count_begin = 3 * cells
    clicks = 0
    positive_clicks = 0
    clicks_on_observed_zombie = 0
    positive_clicks_on_observed_zombie = 0
    positive_clicks_without_observed_zombie = 0
    clicks_with_neighborhood_zombie = 0
    positive_clicks_with_neighborhood_zombie = 0
    positive_clicks_without_neighborhood_zombie = 0
    reward_on_observed_zombie = 0.0
    reward_without_observed_zombie = 0.0
    nonterminal_on_observed_zombie = 0
    nonterminal_without_observed_zombie = 0
    without_observed_zombie = 0
    with ReplayStore(database, read_only=True) as store:
        for transition in store.transitions():
            if not codec.board_click_begin <= transition.action < codec.collect_action:
                continue
            observation = transition.observation
            if observation.ndim != 1 or observation.size != 585:
                continue
            clicks += 1
            cell = transition.action - codec.board_click_begin
            zombie_grid = observation[
                zombie_count_begin : zombie_count_begin + cells
            ].reshape(codec.rows, codec.cols)
            row, col = divmod(cell, codec.cols)
            observed_zombie = bool(zombie_grid[row, col] > 0.0)
            neighborhood_zombie = bool(
                (
                    zombie_grid[
                        max(0, row - 1) : min(codec.rows, row + 2),
                        max(0, col - 1) : min(codec.cols, col + 2),
                    ]
                    > 0.0
                ).any()
            )
            positive = transition.reward > 0.0
            positive_clicks += int(positive)
            if neighborhood_zombie:
                clicks_with_neighborhood_zombie += 1
                positive_clicks_with_neighborhood_zombie += int(positive)
            else:
                positive_clicks_without_neighborhood_zombie += int(positive)
            if observed_zombie:
                clicks_on_observed_zombie += 1
                positive_clicks_on_observed_zombie += int(positive)
                if not transition.terminated:
                    nonterminal_on_observed_zombie += 1
                    reward_on_observed_zombie += transition.reward
            else:
                without_observed_zombie += 1
                positive_clicks_without_observed_zombie += int(positive)
                if not transition.terminated:
                    nonterminal_without_observed_zombie += 1
                    reward_without_observed_zombie += transition.reward
    return {
        "vector_clicks": clicks,
        "clicks_on_observed_zombie": clicks_on_observed_zombie,
        "click_on_observed_zombie_fraction": (
            clicks_on_observed_zombie / clicks if clicks else 0.0
        ),
        "positive_clicks_on_observed_zombie": positive_clicks_on_observed_zombie,
        "positive_clicks_without_observed_zombie": positive_clicks_without_observed_zombie,
        "positive_click_observed_zombie_fraction": (
            positive_clicks_on_observed_zombie / positive_clicks if positive_clicks else 0.0
        ),
        "positive_rate_on_observed_zombie": (
            positive_clicks_on_observed_zombie / clicks_on_observed_zombie
            if clicks_on_observed_zombie
            else 0.0
        ),
        "positive_rate_without_observed_zombie": (
            positive_clicks_without_observed_zombie / without_observed_zombie
            if without_observed_zombie
            else 0.0
        ),
        "clicks_with_3x3_neighborhood_zombie": clicks_with_neighborhood_zombie,
        "positive_click_3x3_neighborhood_zombie_fraction": (
            positive_clicks_with_neighborhood_zombie / positive_clicks
            if positive_clicks
            else 0.0
        ),
        "positive_rate_with_3x3_neighborhood_zombie": (
            positive_clicks_with_neighborhood_zombie / clicks_with_neighborhood_zombie
            if clicks_with_neighborhood_zombie
            else 0.0
        ),
        "positive_rate_without_3x3_neighborhood_zombie": (
            positive_clicks_without_neighborhood_zombie
            / (clicks - clicks_with_neighborhood_zombie)
            if clicks > clicks_with_neighborhood_zombie
            else 0.0
        ),
        "nonterminal_mean_reward_on_observed_zombie": (
            reward_on_observed_zombie / nonterminal_on_observed_zombie
            if nonterminal_on_observed_zombie
            else 0.0
        ),
        "nonterminal_mean_reward_without_observed_zombie": (
            reward_without_observed_zombie / nonterminal_without_observed_zombie
            if nonterminal_without_observed_zombie
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize authentic portable actions and rewards by action family"
    )
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    codec = PortableActionCodec()
    connection = duckdb.connect(str(args.database), read_only=True)
    try:
        rows = connection.execute(
            """
            WITH classified AS (
                SELECT
                    CASE
                        WHEN action = 0 THEN 'wait'
                        WHEN action >= ? AND action < ? THEN 'plant'
                        WHEN action >= ? AND action < ? THEN 'shovel'
                        WHEN action >= ? AND action < ? THEN 'board_click'
                        WHEN action = ? THEN 'collect'
                        ELSE 'out_of_range'
                    END AS kind,
                    action,
                    reward,
                    terminated
                FROM transitions
            )
            SELECT
                kind,
                count(*) AS transitions,
                count(DISTINCT action) AS distinct_actions,
                coalesce(sum(reward), 0) AS total_reward,
                coalesce(avg(reward), 0) AS mean_reward,
                coalesce(sum(reward) FILTER (WHERE NOT terminated), 0)
                    AS nonterminal_total_reward,
                coalesce(avg(reward) FILTER (WHERE NOT terminated), 0)
                    AS nonterminal_mean_reward,
                count(*) FILTER (WHERE reward > 0) AS positive_reward_transitions,
                count(*) FILTER (WHERE terminated) AS terminal_transitions
            FROM classified
            GROUP BY kind
            ORDER BY kind
            """,
            [
                codec.plant_begin,
                codec.shovel_begin,
                codec.shovel_begin,
                codec.board_click_begin,
                codec.board_click_begin,
                codec.collect_action,
                codec.collect_action,
            ],
        ).fetchall()
        plant_actions = [
            int(row[0])
            for row in connection.execute(
                "SELECT action FROM transitions WHERE action >= ? AND action < ?",
                [codec.plant_begin, codec.shovel_begin],
            ).fetchall()
        ]
    finally:
        connection.close()
    print(
        json.dumps(
            {
                "database": str(args.database),
                "action_dim": codec.size,
                "board_click_alignment": board_click_alignment(args.database, codec),
                "plant_action_distribution": plant_action_distribution(
                    plant_actions, codec
                ),
                "actions": {
                    kind: {
                        "transitions": int(transitions),
                        "distinct_actions": int(distinct_actions),
                        "total_reward": float(total_reward),
                        "mean_reward": float(mean_reward),
                        "nonterminal_total_reward": float(nonterminal_total_reward),
                        "nonterminal_mean_reward": float(nonterminal_mean_reward),
                        "positive_reward_transitions": int(positive),
                        "positive_reward_fraction": float(positive / transitions),
                        "terminal_transitions": int(terminals),
                    }
                    for (
                        kind,
                        transitions,
                        distinct_actions,
                        total_reward,
                        mean_reward,
                        nonterminal_total_reward,
                        nonterminal_mean_reward,
                        positive,
                        terminals,
                    ) in rows
                },
            }
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pvzwm.agents.go_explore import ArchiveCell, TrajectoryArchive
from pvzwm.portable import PortableEnv, PortableProcessConfig
from pvzwm.portable_archive import load_portable_archive, resolve_archive_action


@dataclass(frozen=True, slots=True)
class VerificationTask:
    config: PortableProcessConfig
    cell: ArchiveCell
    encoder_state: tuple[int, int, int, float]
    passes: int


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    reason: str
    seed: int
    actions: tuple[int, ...]
    key: bytes
    episode_return: float
    substitutions: int


def unique_cells_in_depth_range(
    cells: Sequence[ArchiveCell], minimum_depth: int, maximum_depth: int
) -> list[ArchiveCell]:
    selected: list[ArchiveCell] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for cell in sorted(cells, key=lambda item: len(item.actions), reverse=True):
        depth = len(cell.actions)
        identity = (cell.seed, cell.actions)
        if minimum_depth <= depth <= maximum_depth and identity not in seen:
            seen.add(identity)
            selected.append(cell)
    return selected


def verify_cell(task: VerificationTask) -> VerificationResult:
    from pvzwm.agents.go_explore import RandomProjectionCellEncoder

    state_dim, projections, seed, cell_width = task.encoder_state
    encoder = RandomProjectionCellEncoder(
        state_dim, projections, seed, cell_width
    )
    actions = task.cell.actions
    substitutions = 0
    final_key = task.cell.key
    for _ in range(task.passes):
        environment: PortableEnv | None = None
        executed: list[int] = []
        try:
            environment = PortableEnv(task.config)
            observation, _ = environment.reset(seed=task.cell.seed)
            for step, replay_action in enumerate(actions):
                action, substituted = resolve_archive_action(
                    replay_action,
                    environment.legal_action_mask(),
                    environment.codec,
                    True,
                )
                if action is None:
                    return VerificationResult(
                        False,
                        f"illegal_non_housekeeping_action@{step}",
                        task.cell.seed,
                        (),
                        b"",
                        task.cell.episode_return,
                        substitutions,
                    )
                observation, _, terminated, truncated, _ = environment.step(action)
                executed.append(action)
                substitutions += int(substituted)
                if terminated or truncated:
                    return VerificationResult(
                        False,
                        f"episode_boundary@{step}",
                        task.cell.seed,
                        (),
                        b"",
                        task.cell.episode_return,
                        substitutions,
                    )
            final_key = encoder.encode(observation.vector(task.config))
            actions = tuple(executed)
        except Exception as error:
            return VerificationResult(
                False,
                f"{type(error).__name__}:{error}",
                task.cell.seed,
                (),
                b"",
                task.cell.episode_return,
                substitutions,
            )
        finally:
            if environment is not None:
                environment.close()
    return VerificationResult(
        True,
        "verified",
        task.cell.seed,
        actions,
        final_key,
        task.cell.episode_return,
        substitutions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel replay verification and compaction for portable archives"
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--level", type=int, default=21)
    parser.add_argument("--ticks-per-step", type=int, default=25)
    parser.add_argument("--minimum-depth", type=int, default=280)
    parser.add_argument("--maximum-depth", type=int, default=337)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-prefixes", type=int)
    args = parser.parse_args()
    if args.minimum_depth < 0 or args.maximum_depth < args.minimum_depth:
        raise SystemExit("Archive depth range is invalid")
    if args.passes < 1 or args.workers < 1:
        raise SystemExit("Verification passes and workers must be positive")

    archive, encoder = load_portable_archive(args.archive)
    cells = unique_cells_in_depth_range(
        list(archive.cells.values()), args.minimum_depth, args.maximum_depth
    )
    if args.max_prefixes is not None:
        if args.max_prefixes < 1:
            raise SystemExit("--max-prefixes must be positive")
        cells = cells[: args.max_prefixes]
    if not cells:
        raise SystemExit("Depth range selected no unique prefixes")

    config = PortableProcessConfig(
        binary=args.binary,
        resources=args.resources,
        level=args.level,
        ticks_per_step=args.ticks_per_step,
        max_steps=3_600,
        frame_width=0,
        frame_height=0,
        reward_mode="bridge",
    )
    encoder_state = (
        encoder.state_dim,
        encoder.projections,
        encoder.seed,
        encoder.cell_width,
    )
    verified = TrajectoryArchive()
    failures: Counter[str] = Counter()
    substitutions = 0
    completed = 0
    tasks = [VerificationTask(config, cell, encoder_state, args.passes) for cell in cells]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(verify_cell, task) for task in tasks]
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
            except Exception as error:
                failures[f"executor:{type(error).__name__}"] += 1
                continue
            substitutions += result.substitutions
            if result.valid:
                verified.add(
                    result.key,
                    result.seed,
                    result.actions,
                    result.episode_return,
                )
            else:
                failures[result.reason.split("@", 1)[0]] += 1
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "requested": len(tasks),
                            "verified_cells": len(verified.cells),
                            "failures": sum(failures.values()),
                        }
                    ),
                    flush=True,
                )

    if not verified.cells:
        raise SystemExit("No archive prefixes passed verification")
    payload = verified.payload(encoder)
    payload["verification"] = {
        "source_archive": str(args.archive),
        "minimum_depth": args.minimum_depth,
        "maximum_depth": args.maximum_depth,
        "passes": args.passes,
        "prefixes_requested": len(tasks),
        "verified_cells": len(verified.cells),
        "failures": dict(failures),
        "housekeeping_substitutions": substitutions,
        "reward_shaping": False,
        "game_specific_archive_features": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    summary = {
        **payload["verification"],
        "output": str(args.output),
        "verified_minimum_depth": min(
            len(cell.actions) for cell in verified.cells.values()
        ),
        "verified_maximum_depth": max(
            len(cell.actions) for cell in verified.cells.values()
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

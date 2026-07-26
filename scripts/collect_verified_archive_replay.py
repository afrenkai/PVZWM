from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

from pvzwm.agents.go_explore import ArchiveCell
from pvzwm.data.store import ReplayStore, Transition
from pvzwm.portable import PortableEnv, PortableProcessConfig
from pvzwm.portable_archive import load_portable_archive, resolve_archive_action


@dataclass(frozen=True, slots=True)
class PrefixReplayTask:
    config: PortableProcessConfig
    cell: ArchiveCell
    restore_attempts: int = 4


@dataclass(frozen=True, slots=True)
class PrefixReplayResult:
    valid: bool
    reason: str
    seed: int
    key: str
    depth: int
    substitutions: int
    attempts: int
    transitions: tuple[Transition, ...]


def _replay_prefix_once(task: PrefixReplayTask, attempt: int) -> PrefixReplayResult:
    environment: PortableEnv | None = None
    recorded: list[Transition] = []
    substitutions = 0
    try:
        environment = PortableEnv(task.config)
        observation, _ = environment.reset(seed=task.cell.seed)
        for step, replay_action in enumerate(task.cell.actions):
            legal_mask = environment.legal_action_mask()
            action, substituted = resolve_archive_action(
                replay_action,
                legal_mask,
                environment.codec,
                True,
            )
            if action is None:
                return PrefixReplayResult(
                    False,
                    f"illegal_non_housekeeping_action@{step}",
                    task.cell.seed,
                    task.cell.key.hex(),
                    len(task.cell.actions),
                    substitutions,
                    attempt,
                    (),
                )
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            recorded.append(
                Transition(
                    observation.vector(task.config).astype(np.float32, copy=False),
                    action,
                    reward,
                    next_observation.vector(task.config).astype(np.float32, copy=False),
                    terminated,
                    truncated or step + 1 == len(task.cell.actions),
                    legal_mask.astype(np.bool_, copy=True),
                )
            )
            substitutions += int(substituted)
            observation = next_observation
            if terminated or truncated:
                return PrefixReplayResult(
                    False,
                    f"episode_boundary@{step}",
                    task.cell.seed,
                    task.cell.key.hex(),
                    len(task.cell.actions),
                    substitutions,
                    attempt,
                    (),
                )
    except Exception as error:
        return PrefixReplayResult(
            False,
            f"{type(error).__name__}:{error}",
            task.cell.seed,
            task.cell.key.hex(),
            len(task.cell.actions),
            substitutions,
            attempt,
            (),
        )
    finally:
        if environment is not None:
            environment.close()
    return PrefixReplayResult(
        True,
        "replayed",
        task.cell.seed,
        task.cell.key.hex(),
        len(task.cell.actions),
        substitutions,
        attempt,
        tuple(recorded),
    )


def replay_prefix(task: PrefixReplayTask) -> PrefixReplayResult:
    if task.restore_attempts < 1:
        raise ValueError("restore_attempts must be positive")
    result: PrefixReplayResult | None = None
    for attempt in range(1, task.restore_attempts + 1):
        result = _replay_prefix_once(task, attempt)
        if result.valid:
            return result
    assert result is not None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay verified portable archive prefixes into a remote DuckDB corpus "
            "for hindsight goal relabeling"
        )
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--level", type=int, default=21)
    parser.add_argument("--ticks-per-step", type=int, default=25)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--restore-attempts", type=int, default=4)
    parser.add_argument("--source", default="verified_archive_prefix")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.restore_attempts < 1:
        raise SystemExit("--workers and --restore-attempts must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.database.exists():
        raise SystemExit(f"Refusing to overwrite existing database: {args.database}")

    archive, _ = load_portable_archive(args.archive)
    cells = sorted(
        archive.cells.values(),
        key=lambda cell: (len(cell.actions), cell.seed, cell.key),
    )
    if args.limit is not None:
        cells = cells[: args.limit]
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
    completed = 0
    valid = 0
    failed = 0
    transitions = 0
    substitutions = 0
    restore_attempts = 0
    failures: dict[str, int] = {}
    args.database.parent.mkdir(parents=True, exist_ok=True)
    with ReplayStore(args.database) as store:
        tasks = [PrefixReplayTask(config, cell, args.restore_attempts) for cell in cells]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = [executor.submit(replay_prefix, task) for task in tasks]
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                substitutions += result.substitutions
                restore_attempts += result.attempts
                if not result.valid:
                    failed += 1
                    reason = result.reason.split("@", 1)[0]
                    failures[reason] = failures.get(reason, 0) + 1
                else:
                    valid += 1
                    transitions += len(result.transitions)
                    metadata: dict[str, object] = {
                        "level_id": args.level,
                        "won": False,
                        "verified_prefix": True,
                        "archive_key": result.key,
                        "archive_depth": result.depth,
                        "housekeeping_substitutions": result.substitutions,
                        "restore_attempts": result.attempts,
                        "reward_shaping": False,
                    }
                    with store.episode(args.source, result.seed, metadata) as episode:
                        for transition in result.transitions:
                            episode.append(transition)
                if completed % 25 == 0 or completed == len(tasks):
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "requested": len(tasks),
                                "valid": valid,
                                "failed": failed,
                                "transitions": transitions,
                            }
                        ),
                        flush=True,
                    )

    summary = {
        "archive": str(args.archive),
        "database": str(args.database),
        "source": args.source,
        "level": args.level,
        "ticks_per_step": args.ticks_per_step,
        "workers": args.workers,
        "requested_prefixes": len(cells),
        "replayed_prefixes": valid,
        "failed_prefixes": failed,
        "transitions": transitions,
        "housekeeping_substitutions": substitutions,
        "restore_attempts": restore_attempts,
        "maximum_restore_attempts": args.restore_attempts,
        "failures": failures,
        "reward_shaping": False,
        "game_specific_goal_features": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

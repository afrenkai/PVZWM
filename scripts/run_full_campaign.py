from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch


def stage_id(level: int) -> str:
    return f"{(level - 1) // 10 + 1}-{(level - 1) % 10 + 1}"


def authentic_progress(
    start_level: int,
    final_level: int,
    episode_levels: list[int],
    episode_wins: list[bool],
) -> tuple[int, int, bool]:
    if len(episode_levels) != len(episode_wins):
        raise ValueError("Checkpoint episode levels and wins do not align")
    frontier = start_level
    completed_through = start_level - 1
    campaign_complete = False
    for level, won in zip(episode_levels, episode_wins, strict=True):
        if not won or level != frontier:
            continue
        completed_through = frontier
        if frontier == final_level:
            campaign_complete = True
            break
        frontier += 1
    return frontier, completed_through, campaign_complete


def checkpoint_policy_architecture(checkpoint: dict[str, object]) -> str:
    architecture = str(checkpoint.get("policy_architecture", "flat"))
    if architecture not in {
        "flat",
        "spatial",
        "spatial-conv",
        "portable-autoregressive",
        "portable-vdn-autoregressive",
    }:
        raise ValueError(f"Unsupported checkpoint policy architecture: {architecture}")
    return architecture


def level_loadout_archive(template: str | None, level: int) -> Path | None:
    """Resolve a frontier-specific learned loadout archive."""

    if template is None:
        return None
    try:
        rendered = template.format(level=level, stage=stage_id(level))
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Loadout archive template may only use {level} and {stage}"
        ) from error
    return Path(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable, authentic-win-gated PvZ Adventure training"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--portable-binary", type=Path, required=True)
    parser.add_argument("--portable-resources", type=Path, required=True)
    parser.add_argument("--start-level", type=int, choices=range(1, 51), required=True)
    parser.add_argument("--final-level", type=int, choices=range(1, 51), default=50)
    parser.add_argument("--completed-through", type=int, choices=range(0, 50), default=0)
    parser.add_argument("--chunk-steps", type=int, default=400_000)
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--rehearsal-probability", type=float, default=0.3)
    parser.add_argument(
        "--loadout-archive-template",
        help=(
            "Optional frontier-specific MAP-Elites archive path template. "
            "Use {level} for the 1-50 level number or {stage} for e.g. 3-1. "
            "When set, each chunk trains only its current frontier so the next "
            "level can select a different authentic seed-packet loadout."
        ),
    )
    parser.add_argument("--seed", type=int, default=220_000)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.final_level < args.start_level:
        raise SystemExit("Final level cannot precede the start level")
    if args.completed_through != args.start_level - 1:
        raise SystemExit("Completed prefix must end immediately before the start level")
    if args.chunk_steps < 1 or args.max_chunks < 1:
        raise SystemExit("Chunk steps and chunk count must be positive")
    if not 0.0 <= args.rehearsal_probability <= 1.0:
        raise SystemExit("Rehearsal probability must be in [0, 1]")
    if (
        args.loadout_archive_template is not None
        and args.final_level > args.start_level
        and "{level}" not in args.loadout_archive_template
        and "{stage}" not in args.loadout_archive_template
    ):
        raise SystemExit(
            "A multi-level learned-loadout campaign requires {level} or {stage} "
            "in --loadout-archive-template"
        )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    current_checkpoint = args.checkpoint
    initial_checkpoint = torch.load(
        current_checkpoint, map_location="cpu", weights_only=True
    )
    policy_architecture = checkpoint_policy_architecture(initial_checkpoint)
    frontier = args.start_level
    completed_through = args.completed_through
    chunks: list[dict[str, object]] = []
    status_path = args.output_directory / "campaign-progress.json"

    for chunk in range(args.max_chunks):
        loadout_archive = level_loadout_archive(
            args.loadout_archive_template, frontier
        )
        if loadout_archive is not None and not loadout_archive.is_file():
            raise SystemExit(
                f"No learned loadout archive exists for {stage_id(frontier)}: "
                f"{loadout_archive}"
            )
        output = args.output_directory / f"chunk-{chunk:03d}-from-{frontier:02d}.pt"
        experiments = args.output_directory / f"chunk-{chunk:03d}.duckdb"
        command = [
            "uv",
            "run",
            "pvzwm",
            "train-agent",
            "--agent",
            "ppo",
            "--environment",
            "portable",
            "--policy-architecture",
            policy_architecture,
            "--initialize-from",
            str(current_checkpoint),
            "--portable-binary",
            str(args.portable_binary),
            "--portable-resources",
            str(args.portable_resources),
            "--portable-level",
            str(frontier),
            "--portable-reward-mode",
            "clipped-damage",
            "--portable-episode-steps",
            "3600",
            "--steps",
            str(args.chunk_steps),
            "--rollout-steps",
            "2048",
            "--batch-size",
            "256",
            "--learning-rate",
            "1e-4",
            "--discount",
            "1.0",
            "--gae-lambda",
            "1.0",
            "--entropy-weight",
            "0.05",
            "--intrinsic-reward-weight",
            "0.01",
            "--output",
            str(output),
            "--experiment-database",
            str(experiments),
            "--device",
            args.device,
            "--seed",
            str(args.seed + chunk * args.chunk_steps),
        ]
        if loadout_archive is None:
            command.extend(
                [
                    "--portable-curriculum-end",
                    str(args.final_level),
                    "--portable-completed-through",
                    str(completed_through),
                    "--portable-rehearsal-probability",
                    str(args.rehearsal_probability),
                ]
            )
        else:
            command.extend(
                ["--portable-loadout-archive", str(loadout_archive)]
            )
        subprocess.run(command, check=True)
        checkpoint = torch.load(output, map_location="cpu", weights_only=True)
        levels = [int(level) for level in checkpoint.get("episode_levels", [])]
        wins = [bool(won) for won in checkpoint.get("episode_wins", [])]
        if loadout_archive is not None and not levels:
            levels = [frontier] * len(wins)
        next_frontier, next_completed, complete = authentic_progress(
            frontier,
            args.final_level,
            levels,
            wins,
        )
        record = {
            "chunk": chunk,
            "checkpoint": str(output),
            "start_level": frontier,
            "start_stage": stage_id(frontier),
            "episodes": len(levels),
            "authentic_wins": sum(wins),
            "completed_through": next_completed,
            "completed_stage": stage_id(next_completed) if next_completed else None,
            "next_frontier": next_frontier,
            "next_frontier_stage": stage_id(next_frontier),
            "campaign_complete": complete,
            "loadout_archive": (
                None if loadout_archive is None else str(loadout_archive)
            ),
        }
        chunks.append(record)
        status_path.write_text(
            json.dumps(
                {
                    "progression_signal": "ordered authentic terminal wins only",
                    "final_level": args.final_level,
                    "chunks": chunks,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(record), flush=True)
        current_checkpoint = output
        frontier = next_frontier
        completed_through = next_completed
        if complete:
            return
    raise SystemExit(f"Campaign remains at {stage_id(frontier)} after max chunks")


if __name__ == "__main__":
    main()

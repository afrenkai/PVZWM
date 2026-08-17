from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from pvzwm.agents.evolution import (
    CandidateResult,
    EvolutionEpisode,
    aggregate_candidate,
    descriptor_cell,
    mutate_ordered_choice,
)
from pvzwm.data.store import ReplayStore
from pvzwm.portable import PortableEnv, PortableProcessConfig


@dataclass(frozen=True, slots=True)
class GoalLoadoutEvalSpec:
    uv: Path
    evaluator: Path
    database: Path
    checkpoint: Path
    latent_goal_index: Path
    jepa_checkpoint: Path
    calibration_reports: tuple[Path, ...]
    binary: Path
    resources: Path
    level: int
    episode_steps: int
    max_options: int
    max_option_steps: int
    device: str
    stochastic: bool


@dataclass(frozen=True, slots=True)
class LoadoutElite:
    loadout: tuple[int, ...]
    result: CandidateResult
    generation: int


def load_initial_loadouts(path: Path) -> tuple[tuple[int, ...], ...]:
    payload = json.loads(path.read_text())
    raw_archive = payload.get("archive")
    if not isinstance(raw_archive, list) or not raw_archive:
        raise ValueError("Initial loadout archive has no MAP-Elites entries")
    loadouts: list[tuple[int, ...]] = []
    for entry in raw_archive:
        raw_loadout = entry.get("loadout") if isinstance(entry, dict) else None
        if not isinstance(raw_loadout, list):
            raise ValueError("Initial loadout archive contains an invalid entry")
        loadouts.append(tuple(int(item) for item in raw_loadout))
    return tuple(dict.fromkeys(loadouts))


def result_record(result: CandidateResult) -> dict[str, object]:
    return {
        "quality": result.quality,
        "win_rate": result.win_rate,
        "mean_return": result.mean_return,
        "truncation_rate": result.truncation_rate,
        "descriptor": result.descriptor.tolist(),
        "zero_sun_plant_fraction_diagnostic": result.zero_sun_plant_fraction,
    }


def candidate_rank(result: CandidateResult) -> tuple[float, float]:
    """Promote terminal wins first, then the unmodified environment return."""

    return result.win_rate, result.mean_return


def summary_to_candidate_result(summary: dict[str, object]) -> CandidateResult:
    raw_records = summary.get("episode_records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Goal-frontier evaluation has no episode records")
    episodes: list[EvolutionEpisode] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("Goal-frontier evaluation has a malformed episode")
        descriptor = np.asarray(raw.get("behavior_descriptor"), dtype=np.float64)
        if descriptor.ndim != 1 or descriptor.size == 0:
            raise ValueError("Goal-frontier episode has no behavior descriptor")
        episodes.append(
            EvolutionEpisode(
                won=bool(raw["won"]),
                episode_return=float(raw["return"]),
                length=int(raw["steps"]),
                truncated=bool(raw["truncated"] and not raw["terminated"]),
                descriptor=descriptor,
                zero_sun_plant_fraction=float(
                    raw.get("zero_sun_plant_fraction_diagnostic", 0.0)
                ),
            )
        )
    return aggregate_candidate(episodes)


def evaluation_command(
    spec: GoalLoadoutEvalSpec,
    loadout: tuple[int, ...],
    seed: int,
    episodes: int,
    output: Path,
) -> list[str]:
    command = [
        str(spec.uv),
        "run",
        str(spec.evaluator),
        "--database",
        str(spec.database),
        "--checkpoint",
        str(spec.checkpoint),
        "--latent-goal-index",
        str(spec.latent_goal_index),
        "--jepa-checkpoint",
        str(spec.jepa_checkpoint),
    ]
    for report in spec.calibration_reports:
        command.extend(("--calibration-report", str(report)))
    command.extend(
        (
            "--binary",
            str(spec.binary),
            "--resources",
            str(spec.resources),
            "--output",
            str(output),
            "--level",
            str(spec.level),
            "--episode-steps",
            str(spec.episode_steps),
            "--episodes",
            str(episodes),
            "--max-options",
            str(spec.max_options),
            "--max-option-steps",
            str(spec.max_option_steps),
            "--seed",
            str(seed),
            "--seed-types",
            ",".join(str(item) for item in loadout),
            "--device",
            spec.device,
        )
    )
    if spec.stochastic:
        command.append("--stochastic")
    return command


def evaluate_payload(
    payload: tuple[GoalLoadoutEvalSpec, tuple[int, ...], int, int]
) -> CandidateResult:
    spec, loadout, seed, episodes = payload
    with tempfile.TemporaryDirectory(prefix="pvzwm-loadout-") as directory:
        output = Path(directory) / "evaluation.json"
        completed = subprocess.run(
            evaluation_command(spec, loadout, seed, episodes, output),
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "UV_CACHE_DIR": "/tmp/pvzwm-uv-cache"},
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(f"Goal-frontier evaluation failed: {message}")
        return summary_to_candidate_result(json.loads(output.read_text()))


def probe_loadout_space(
    config: PortableProcessConfig, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    environment = PortableEnv(config)
    try:
        observation, _ = environment.reset(seed=seed)
        capacity = observation.seed_bank_capacity
        available = observation.available_seed_types
        compatibility = tuple(
            int(packet["type"]) for packet in observation.payload["seed_packets"]
        )
    finally:
        environment.close()
    if capacity < 1 or len(compatibility) != capacity:
        raise RuntimeError("Bridge did not report a valid seed-bank capacity")
    if len(available) < capacity or not set(compatibility).issubset(available):
        raise RuntimeError("Bridge did not report a valid unlocked seed set")
    return available, compatibility


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Learn the authentic episode-start seed-packet action for a frozen "
            "JEPA-cluster goal-IQL worker using ordered-loadout MAP-Elites"
        )
    )
    parser.add_argument("--uv", type=Path, default=Path("uv"))
    parser.add_argument(
        "--evaluator", type=Path, default=Path("scripts/eval_goal_iql_frontier.py")
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-goal-index", type=Path, required=True)
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--calibration-report", type=Path, action="append", required=True
    )
    parser.add_argument("--initial-archive", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--experiment-database", type=Path, required=True)
    parser.add_argument("--level", type=int, default=21, choices=range(1, 51))
    parser.add_argument("--episode-steps", type=int, default=3_600)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--max-option-steps", type=int, default=136)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--episodes-per-candidate", type=int, default=1)
    parser.add_argument("--selection-episodes", type=int, default=3)
    parser.add_argument("--heldout-episodes", type=int, default=20)
    parser.add_argument("--descriptor-bins", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--random-restart-probability", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=392_100)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if min(
        args.generations,
        args.population,
        args.episodes_per_candidate,
        args.selection_episodes,
    ) < 1:
        raise SystemExit("Generation, population, and episode counts must be positive")
    if args.heldout_episodes < 1 or not 1 <= args.workers <= 8:
        raise SystemExit("Heldout episodes must be positive and workers in [1, 8]")
    if args.descriptor_bins < 2 or args.episode_steps < 1:
        raise SystemExit("Descriptor bins must be at least two and steps positive")
    if not 0.0 <= args.random_restart_probability <= 1.0:
        raise SystemExit("Random restart probability must be in [0, 1]")

    spec = GoalLoadoutEvalSpec(
        args.uv,
        args.evaluator,
        args.database,
        args.checkpoint,
        args.latent_goal_index,
        args.jepa_checkpoint,
        tuple(args.calibration_report),
        args.binary,
        args.resources,
        args.level,
        args.episode_steps,
        args.max_options,
        args.max_option_steps,
        args.device,
        args.stochastic,
    )
    config = PortableProcessConfig(
        binary=args.binary,
        resources=args.resources,
        level=args.level,
        max_steps=args.episode_steps,
        frame_width=0,
        frame_height=0,
        reward_mode="bridge",
    )
    available, compatibility = probe_loadout_space(config, args.seed)
    initial = load_initial_loadouts(args.initial_archive)
    for loadout in initial:
        if len(loadout) != len(compatibility):
            raise SystemExit("Initial archive loadout has the wrong capacity")
        if len(set(loadout)) != len(loadout) or not set(loadout).issubset(available):
            raise SystemExit("Initial archive contains an inauthentic loadout")
    initial_candidates = tuple(dict.fromkeys((compatibility, *initial)))
    rng = np.random.default_rng(args.seed)
    executor = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn")
    )

    def evaluate_many(
        loadouts: Sequence[tuple[int, ...]], seed: int, episodes: int
    ) -> list[CandidateResult]:
        payloads = ((spec, loadout, seed, episodes) for loadout in loadouts)
        return list(executor.map(evaluate_payload, payloads))

    try:
        initial_results = evaluate_many(
            initial_candidates, args.seed, args.episodes_per_candidate
        )
        baseline = initial_results[0]
        archive: dict[tuple[int, ...], LoadoutElite] = {}
        for loadout, result in zip(initial_candidates, initial_results, strict=True):
            cell = descriptor_cell(result.descriptor, args.descriptor_bins)
            incumbent = archive.get(cell)
            if incumbent is None or candidate_rank(result) > candidate_rank(
                incumbent.result
            ):
                archive[cell] = LoadoutElite(loadout, result, -1)
        evaluated = set(initial_candidates)
        history: list[dict[str, object]] = []
        for generation in range(args.generations):
            candidates: list[tuple[int, ...]] = []
            attempts = 0
            while len(candidates) < args.population:
                attempts += 1
                if attempts > args.population * 200:
                    raise RuntimeError("Could not generate unique loadout candidates")
                if rng.random() < args.random_restart_probability:
                    proposal = tuple(
                        int(item)
                        for item in rng.choice(
                            available, size=len(compatibility), replace=False
                        ).tolist()
                    )
                else:
                    elites = tuple(archive.values())
                    parent = elites[int(rng.integers(0, len(elites)))]
                    proposal = mutate_ordered_choice(parent.loadout, available, rng)
                if proposal not in evaluated and proposal not in candidates:
                    candidates.append(proposal)
            seed = args.seed + (generation + 1) * args.episodes_per_candidate
            results = evaluate_many(candidates, seed, args.episodes_per_candidate)
            additions = 0
            replacements = 0
            records: list[dict[str, object]] = []
            for loadout, result in zip(candidates, results, strict=True):
                evaluated.add(loadout)
                cell = descriptor_cell(result.descriptor, args.descriptor_bins)
                incumbent = archive.get(cell)
                if incumbent is None:
                    archive[cell] = LoadoutElite(loadout, result, generation)
                    additions += 1
                elif candidate_rank(result) > candidate_rank(incumbent.result):
                    archive[cell] = LoadoutElite(loadout, result, generation)
                    replacements += 1
                records.append(
                    {"loadout": list(loadout), "cell": list(cell), **result_record(result)}
                )
            best = max(archive.values(), key=lambda item: candidate_rank(item.result))
            generation_record = {
                "generation": generation,
                "seed": seed,
                "archive_size": len(archive),
                "archive_additions": additions,
                "archive_replacements": replacements,
                "best_loadout": list(best.loadout),
                "best": result_record(best.result),
                "candidates": records,
            }
            history.append(generation_record)
            print(json.dumps(generation_record), flush=True)
        selection_seed = args.seed + 500_000
        selection_elites = tuple(archive.values())
        selection_results = evaluate_many(
            [elite.loadout for elite in selection_elites],
            selection_seed,
            args.selection_episodes,
        )
        selection_index = max(
            range(len(selection_results)),
            key=lambda index: candidate_rank(selection_results[index]),
        )
        best = selection_elites[selection_index]
        selection = [
            {
                "loadout": list(elite.loadout),
                **result_record(result),
            }
            for elite, result in zip(
                selection_elites, selection_results, strict=True
            )
        ]
        heldout_seed = args.seed + 1_000_000
        heldout = evaluate_many(
            [best.loadout], heldout_seed, args.heldout_episodes
        )[0]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    archive_records = [
        {
            "cell": list(cell),
            "loadout": list(elite.loadout),
            "generation": elite.generation,
            **result_record(elite.result),
        }
        for cell, elite in sorted(archive.items())
    ]
    evolution = {
        "algorithm": "goal-iql-ordered-loadout-map-elites",
        "source_checkpoint": str(args.checkpoint),
        "level": args.level,
        "available_seed_types": list(available),
        "seed_bank_capacity": len(compatibility),
        "compatibility_loadout": list(compatibility),
        "initial_archive": str(args.initial_archive),
        "initial_loadouts": [list(item) for item in initial_candidates],
        "best_loadout": list(best.loadout),
        "population": args.population,
        "workers": args.workers,
        "generations": args.generations,
        "episodes_per_candidate": args.episodes_per_candidate,
        "selection_episodes": args.selection_episodes,
        "descriptor_bins": args.descriptor_bins,
        "random_restart_probability": args.random_restart_probability,
        "stochastic": args.stochastic,
        "common_random_numbers": True,
        "quality_definition": "unmodified mean episodic environment return",
        "promotion_order": "terminal win rate, then unmodified environment return",
        "loadout_constraints": "authentic unlocked unique ordered seed packets",
        "plant_identity_reward_shaping": False,
        "baseline": result_record(baseline),
        "selection": {
            "seed_begin": selection_seed,
            "episodes_per_elite": args.selection_episodes,
            "candidates": selection,
        },
        "heldout": {
            **result_record(heldout),
            "seed_begin": heldout_seed,
            "episodes": args.heldout_episodes,
        },
        "archive": archive_records,
        "history": history,
    }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    output_checkpoint = dict(checkpoint)
    output_checkpoint["selected_seed_types"] = best.loadout
    output_checkpoint["loadout_evolution"] = evolution
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    metrics = {"output": str(args.output), "database_location": "remote_only", **evolution}
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
    with ReplayStore(args.experiment_database) as store:
        experiment_id = store.record_experiment(
            "goal-iql-ordered-loadout-map-elites",
            {
                "population": args.population,
                "generations": args.generations,
                "episodes_per_candidate": args.episodes_per_candidate,
                "selection_episodes": args.selection_episodes,
                "descriptor_bins": args.descriptor_bins,
                "random_restart_probability": args.random_restart_probability,
                "stochastic": args.stochastic,
            },
            {
                "output": str(args.output),
                "best_loadout": list(best.loadout),
                "baseline": result_record(baseline),
                "heldout": result_record(heldout),
            },
            agent="goal-iql-loadout-qd",
            seed=args.seed,
            device=args.device,
        )
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "output": str(args.output),
                "best_loadout": list(best.loadout),
                "heldout": result_record(heldout),
            }
        )
    )


if __name__ == "__main__":
    main()

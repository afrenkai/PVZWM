from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import json
import multiprocessing as mp
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from pvzwm.agents.dqn import QNetwork
from pvzwm.agents.evolution import (
    CandidateResult,
    EvolutionEpisode,
    aggregate_candidate,
    behavior_descriptor,
    descriptor_cell,
    mutate_ordered_choice,
    zero_sun_plant_fraction,
)
from pvzwm.agents.ppo import build_actor_critic
from pvzwm.data.store import ReplayStore
from pvzwm.device import select_device
from pvzwm.domain import GameConfig
from pvzwm.portable import (
    PortableEnv,
    PortableProcessConfig,
    collect_sun_first_mask,
    portable_process_config_from_checkpoint,
)


@dataclass(frozen=True, slots=True)
class LoadoutElite:
    loadout: tuple[int, ...]
    result: CandidateResult
    generation: int


_worker_model: torch.nn.Module | None = None
_worker_agent: str | None = None
_worker_config: PortableProcessConfig | None = None
_worker_game_config: GameConfig | None = None
_worker_stochastic = False
_worker_epsilon = 0.0


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


def build_controller(
    checkpoint: dict[str, object], device: torch.device
) -> tuple[torch.nn.Module, str]:
    """Restore an actor or value-based portable controller."""
    agent = str(checkpoint.get("agent", ""))
    if agent in {"dqn", "dyna-dqn"}:
        model: torch.nn.Module = QNetwork(
            int(checkpoint["state_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint["hidden_dim"]),
        ).to(device)
    elif agent == "ppo":
        model = build_actor_critic(
            int(checkpoint["state_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint["hidden_dim"]),
            str(checkpoint.get("policy_architecture", "flat")),
        ).to(device)
    else:
        raise ValueError(
            "Loadout evolution requires a PPO, DQN, or Dyna-DQN checkpoint"
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, agent


def controller_scores(
    model: torch.nn.Module, agent: str, state: torch.Tensor
) -> torch.Tensor:
    output = model(state.unsqueeze(0))
    if agent in {"dqn", "dyna-dqn"}:
        assert isinstance(output, torch.Tensor)
        return output.squeeze(0)
    assert isinstance(output, tuple)
    return output[0].squeeze(0)


def select_action(
    scores: torch.Tensor,
    legal: torch.Tensor,
    *,
    stochastic: bool,
    epsilon: float,
) -> int:
    """Select from legal actions with explicit epsilon-greedy exploration."""
    if epsilon > 0.0 and float(torch.rand((), device=scores.device)) < epsilon:
        legal_actions = torch.nonzero(legal, as_tuple=False).flatten()
        index = torch.randint(len(legal_actions), (), device=scores.device)
        return int(legal_actions[index].item())
    masked = scores.masked_fill(~legal, -torch.inf)
    if stochastic:
        return int(torch.distributions.Categorical(logits=masked).sample().item())
    return int(masked.argmax().item())


def evaluate_loadout(
    model: torch.nn.Module,
    agent: str,
    base_config: PortableProcessConfig,
    game_config: GameConfig,
    loadout: tuple[int, ...],
    seeds: Sequence[int],
    *,
    stochastic: bool,
    epsilon: float,
) -> CandidateResult:
    environment = PortableEnv(replace(base_config, seed_types=loadout))
    episodes: list[EvolutionEpisode] = []
    device = next(model.parameters()).device
    try:
        for seed in seeds:
            torch.manual_seed(seed)
            observation, _ = environment.reset(seed=seed)
            actions: list[int] = []
            sun_before_actions: list[float] = []
            episode_return = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                state = torch.from_numpy(observation.vector(game_config)).to(device)
                legal = torch.from_numpy(
                    collect_sun_first_mask(
                        environment.codec, environment.legal_action_mask()
                    )
                ).to(device)
                with torch.inference_mode():
                    action = select_action(
                        controller_scores(model, agent, state),
                        legal,
                        stochastic=stochastic,
                        epsilon=epsilon,
                    )
                actions.append(action)
                sun_before_actions.append(float(observation.payload.get("sun", 0.0)))
                observation, reward, terminated, truncated, _ = environment.step(action)
                episode_return += reward
            episodes.append(
                EvolutionEpisode(
                    won=bool(observation.won),
                    episode_return=episode_return,
                    length=len(actions),
                    truncated=bool(truncated and not terminated),
                    descriptor=behavior_descriptor(actions, base_config.max_steps),
                    zero_sun_plant_fraction=zero_sun_plant_fraction(
                        actions, sun_before_actions
                    ),
                )
            )
    finally:
        environment.close()
    return aggregate_candidate(episodes)


def initialize_worker(
    checkpoint_path: Path,
    config: PortableProcessConfig,
    game_config: GameConfig,
    device_name: str,
    stochastic: bool,
    epsilon: float,
) -> None:
    global _worker_model, _worker_agent, _worker_config, _worker_game_config
    global _worker_stochastic, _worker_epsilon
    device = select_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, agent = build_controller(checkpoint, device.device)
    _worker_model = model
    _worker_agent = agent
    _worker_config = config
    _worker_game_config = game_config
    _worker_stochastic = stochastic
    _worker_epsilon = epsilon


def worker_evaluate(payload: tuple[tuple[int, ...], tuple[int, ...]]) -> CandidateResult:
    loadout, seeds = payload
    if (
        _worker_model is None
        or _worker_agent is None
        or _worker_config is None
        or _worker_game_config is None
    ):
        raise RuntimeError("Loadout worker was not initialized")
    return evaluate_loadout(
        _worker_model,
        _worker_agent,
        _worker_config,
        _worker_game_config,
        loadout,
        seeds,
        stochastic=_worker_stochastic,
        epsilon=_worker_epsilon,
    )


def probe_loadout_space(
    config: PortableProcessConfig, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    environment = PortableEnv(replace(config, seed_types=None))
    try:
        observation, _ = environment.reset(seed=seed)
        capacity = observation.seed_bank_capacity
        available = observation.available_seed_types
        compatibility_loadout = tuple(
            int(packet["type"]) for packet in observation.payload["seed_packets"]
        )
    finally:
        environment.close()
    if capacity < 1 or len(compatibility_loadout) != capacity:
        raise RuntimeError("Bridge did not report a valid authentic seed-bank capacity")
    if len(available) < capacity or not set(compatibility_loadout).issubset(available):
        raise RuntimeError("Bridge did not report a valid unlocked seed set")
    return available, compatibility_loadout


def portable_config(
    checkpoint: dict[str, object], args: argparse.Namespace
) -> PortableProcessConfig:
    raw = checkpoint.get("portable_config")
    if not isinstance(raw, dict):
        raise SystemExit("Checkpoint has no portable engine configuration")
    config = portable_process_config_from_checkpoint(
        raw, state_dim=int(checkpoint["state_dim"])
    )
    return replace(
        config,
        binary=config.binary if args.portable_binary is None else args.portable_binary,
        resources=(
            config.resources if args.portable_resources is None else args.portable_resources
        ),
        level=config.level if args.level is None else args.level,
        max_steps=args.episode_steps,
        frame_width=0,
        frame_height=0,
        seed_types=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MAP-Elites search over authentic ordered seed loadouts for a learned "
            "portable controller"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--initial-archive",
        type=Path,
        help=(
            "Optional prior MAP-Elites archive. Every deck is re-evaluated with "
            "common random numbers before mutation so co-adaptation preserves "
            "diverse loadout hypotheses."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--experiment-database", type=Path, required=True)
    parser.add_argument("--portable-binary", type=Path)
    parser.add_argument("--portable-resources", type=Path)
    parser.add_argument("--level", type=int, choices=range(1, 51))
    parser.add_argument("--episode-steps", type=int, default=3_600)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--episodes-per-candidate", type=int, default=2)
    parser.add_argument("--heldout-episodes", type=int, default=10)
    parser.add_argument("--descriptor-bins", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel authentic engine workers; cap at the assigned CPU-core count",
    )
    parser.add_argument("--random-restart-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=330_100)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Legal-action epsilon-greedy exploration during loadout evaluation",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.generations < 1 or args.population < 1 or args.workers < 1:
        raise SystemExit("Generations and population must be positive")
    if args.episodes_per_candidate < 1 or args.heldout_episodes < 1:
        raise SystemExit("Episode counts must be positive")
    if args.episode_steps < 1 or args.descriptor_bins < 2:
        raise SystemExit("Episode steps must be positive and descriptor bins at least two")
    if not 0.0 <= args.random_restart_probability <= 1.0:
        raise SystemExit("Random restart probability must be in [0, 1]")
    if not 0.0 <= args.epsilon <= 1.0:
        raise SystemExit("Epsilon must be in [0, 1]")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    game_config = GameConfig(**checkpoint["game_config"])
    try:
        model, agent = build_controller(checkpoint, device.device)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    config = portable_config(checkpoint, args)
    available, compatibility_loadout = probe_loadout_space(config, args.seed)
    initial_loadouts = (
        ()
        if args.initial_archive is None
        else load_initial_loadouts(args.initial_archive)
    )
    for loadout in initial_loadouts:
        if len(loadout) != len(compatibility_loadout):
            raise SystemExit("Initial archive loadout has the wrong seed-bank capacity")
        if len(set(loadout)) != len(loadout) or not set(loadout).issubset(available):
            raise SystemExit("Initial archive loadout is not an authentic unlocked deck")
    initial_candidates = tuple(
        dict.fromkeys((compatibility_loadout, *initial_loadouts))
    )
    rng = np.random.default_rng(args.seed)

    executor = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(
                args.checkpoint,
                config,
                game_config,
                args.device,
                args.stochastic,
                args.epsilon,
            ),
        )

    def evaluate_many(
        loadouts: Sequence[tuple[int, ...]], seeds: tuple[int, ...]
    ) -> list[CandidateResult]:
        if executor is None:
            return [
                evaluate_loadout(
                    model,
                    agent,
                    config,
                    game_config,
                    loadout,
                    seeds,
                    stochastic=args.stochastic,
                    epsilon=args.epsilon,
                )
                for loadout in loadouts
            ]
        return list(executor.map(worker_evaluate, ((loadout, seeds) for loadout in loadouts)))

    try:
        baseline_seeds = tuple(
            args.seed + index for index in range(args.episodes_per_candidate)
        )
        initial_results = evaluate_many(initial_candidates, baseline_seeds)
        baseline = initial_results[0]
        archive: dict[tuple[int, ...], LoadoutElite] = {}
        for loadout, result in zip(initial_candidates, initial_results, strict=True):
            cell = descriptor_cell(result.descriptor, args.descriptor_bins)
            incumbent = archive.get(cell)
            if incumbent is None or result.quality > incumbent.result.quality:
                archive[cell] = LoadoutElite(loadout, result, -1)
        evaluated: set[tuple[int, ...]] = set(initial_candidates)
        history: list[dict[str, object]] = []

        for generation in range(args.generations):
            candidates: list[tuple[int, ...]] = []
            attempts = 0
            while len(candidates) < args.population:
                attempts += 1
                if attempts > args.population * 100:
                    raise RuntimeError(
                        "Could not generate enough unique loadout candidates"
                    )
                if rng.random() < args.random_restart_probability:
                    proposal = tuple(
                        int(value)
                        for value in rng.choice(
                            available,
                            size=len(compatibility_loadout),
                            replace=False,
                        ).tolist()
                    )
                else:
                    parents = tuple(archive.values())
                    parent = parents[int(rng.integers(0, len(parents)))]
                    proposal = mutate_ordered_choice(parent.loadout, available, rng)
                if proposal in evaluated or proposal in candidates:
                    continue
                candidates.append(proposal)

            seeds = tuple(
                args.seed
                + (generation + 1) * args.episodes_per_candidate
                + episode
                for episode in range(args.episodes_per_candidate)
            )
            generation_records: list[dict[str, object]] = []
            additions = 0
            replacements = 0
            results = evaluate_many(candidates, seeds)
            for loadout, result in zip(candidates, results, strict=True):
                evaluated.add(loadout)
                cell = descriptor_cell(result.descriptor, args.descriptor_bins)
                incumbent = archive.get(cell)
                if incumbent is None:
                    archive[cell] = LoadoutElite(loadout, result, generation)
                    additions += 1
                elif result.quality > incumbent.result.quality:
                    archive[cell] = LoadoutElite(loadout, result, generation)
                    replacements += 1
                generation_records.append(
                    {
                        "loadout": list(loadout),
                        "cell": list(cell),
                        **result_record(result),
                    }
                )
            best = max(archive.values(), key=lambda elite: elite.result.quality)
            record = {
                "generation": generation,
                "seeds": list(seeds),
                "archive_size": len(archive),
                "archive_additions": additions,
                "archive_replacements": replacements,
                "best_loadout": list(best.loadout),
                "best": result_record(best.result),
                "candidates": generation_records,
            }
            history.append(record)
            print(json.dumps(record), flush=True)

        best = max(archive.values(), key=lambda elite: elite.result.quality)
        heldout_seeds = tuple(
            args.seed + 1_000_000 + index for index in range(args.heldout_episodes)
        )
        heldout = evaluate_many([best.loadout], heldout_seeds)[0]
    finally:
        if executor is not None:
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
        "algorithm": "ordered-loadout-map-elites",
        "source_checkpoint": str(args.checkpoint),
        "level": config.level,
        "available_seed_types": list(available),
        "seed_bank_capacity": len(compatibility_loadout),
        "compatibility_loadout": list(compatibility_loadout),
        "initial_archive": (
            None if args.initial_archive is None else str(args.initial_archive)
        ),
        "initial_loadouts": [list(loadout) for loadout in initial_candidates],
        "best_loadout": list(best.loadout),
        "population": args.population,
        "workers": args.workers,
        "generations": args.generations,
        "episodes_per_candidate": args.episodes_per_candidate,
        "descriptor_bins": args.descriptor_bins,
        "random_restart_probability": args.random_restart_probability,
        "stochastic": args.stochastic,
        "epsilon": args.epsilon,
        "common_random_numbers": True,
        "quality_definition": "mean episodic environment return",
        "loadout_constraints": "authentic unlocked unique ordered seed packets",
        "plant_identity_reward_shaping": False,
        "baseline": result_record(baseline),
        "heldout": {**result_record(heldout), "seeds": list(heldout_seeds)},
        "archive": archive_records,
        "history": history,
    }
    selected_config = replace(config, seed_types=best.loadout)
    output_checkpoint = dict(checkpoint)
    output_checkpoint.update(
        {
            "portable_config": selected_config.checkpoint_dict(),
            "training_method": "ordered-loadout-map-elites",
            "initialization": {"checkpoint": str(args.checkpoint)},
            "loadout_evolution": evolution,
            "summary": {
                "heldout_win_rate": heldout.win_rate,
                "heldout_mean_return": heldout.mean_return,
                "heldout_truncation_rate": heldout.truncation_rate,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    metrics = {"output": str(args.output), "device": device.backend, **evolution}
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
    with ReplayStore(args.experiment_database) as store:
        experiment_id = store.record_experiment(
            "ordered-loadout-map-elites",
            {
                "population": args.population,
                "generations": args.generations,
                "episodes_per_candidate": args.episodes_per_candidate,
                "descriptor_bins": args.descriptor_bins,
                "random_restart_probability": args.random_restart_probability,
                "stochastic": args.stochastic,
                "epsilon": args.epsilon,
            },
            {
                "output": str(args.output),
                "best_loadout": list(best.loadout),
                "baseline": result_record(baseline),
                "heldout": result_record(heldout),
            },
            agent=f"{agent}-loadout-qd",
            seed=args.seed,
            device=device.backend,
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

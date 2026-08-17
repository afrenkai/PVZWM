from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from pvzwm.agents.evolution import (
    AdamAscent,
    CandidateResult,
    EvolutionEpisode,
    aggregate_candidate,
    antithetic_gradient,
    behavior_descriptor,
    centered_ranks,
    parameter_vector,
    policy_parameter_names,
    seeded_noise,
    set_parameter_vector,
    zero_sun_plant_fraction,
)
from pvzwm.agents.ppo import PolicyModel, build_actor_critic
from pvzwm.data.store import ReplayStore
from pvzwm.device import select_device
from pvzwm.domain import GameConfig
from pvzwm.portable import (
    PortableEnv,
    PortableProcessConfig,
    collect_sun_first_mask,
    portable_process_config_from_checkpoint,
)


def parse_seed_types(value: str) -> tuple[int, ...]:
    try:
        seed_types = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Seed types must be comma-separated integer enum values"
        ) from error
    if not 1 <= len(seed_types) <= 10:
        raise argparse.ArgumentTypeError("A seed loadout must contain 1 to 10 types")
    if len(set(seed_types)) != len(seed_types):
        raise argparse.ArgumentTypeError("A seed loadout cannot contain duplicates")
    if any(not 0 <= seed_type < 48 for seed_type in seed_types):
        raise argparse.ArgumentTypeError("Seed types must be in [0, 48)")
    return seed_types


def evaluate_policy(
    model: PolicyModel,
    environment: PortableEnv,
    game_config: GameConfig,
    seeds: Sequence[int],
    *,
    stochastic: bool,
) -> CandidateResult:
    episodes: list[EvolutionEpisode] = []
    device = next(model.parameters()).device
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
                logits, _ = model(state.unsqueeze(0))
                masked = logits.squeeze(0).masked_fill(~legal, -torch.inf)
                if stochastic:
                    action = int(
                        torch.distributions.Categorical(logits=masked).sample().item()
                    )
                else:
                    action = int(masked.argmax().item())
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
                descriptor=behavior_descriptor(actions, environment.config.max_steps),
                zero_sun_plant_fraction=zero_sun_plant_fraction(
                    actions, sun_before_actions
                ),
            )
        )
    return aggregate_candidate(episodes)


def result_record(result: CandidateResult) -> dict[str, object]:
    return {
        "quality": result.quality,
        "win_rate": result.win_rate,
        "mean_return": result.mean_return,
        "truncation_rate": result.truncation_rate,
        "descriptor": result.descriptor.tolist(),
        "zero_sun_plant_fraction_diagnostic": result.zero_sun_plant_fraction,
    }


def build_portable_config(
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
        seed_types=config.seed_types if args.seed_types is None else args.seed_types,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warm-start outcome-only evolution from an authentic portable PPO policy"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--experiment-database", type=Path, required=True)
    parser.add_argument("--portable-binary", type=Path)
    parser.add_argument("--portable-resources", type=Path)
    parser.add_argument("--level", type=int, choices=range(1, 51))
    parser.add_argument(
        "--seed-types",
        type=parse_seed_types,
        help=(
            "Authentic ordered seed-packet loadout to co-evolve with this policy. "
            "The bridge still validates capacity and unlock status."
        ),
    )
    parser.add_argument("--episode-steps", type=int, default=3_600)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument(
        "--population",
        type=int,
        default=8,
        help="Total antithetic population size; must be a positive even number",
    )
    parser.add_argument("--episodes-per-candidate", type=int, default=2)
    parser.add_argument("--heldout-episodes", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=180_000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.generations < 1 or args.population < 2 or args.population % 2:
        raise SystemExit("Generations must be positive and population must be positive and even")
    if args.episodes_per_candidate < 1 or args.heldout_episodes < 1:
        raise SystemExit("Episode counts must be positive")
    if args.episode_steps < 1 or args.sigma <= 0.0 or args.learning_rate <= 0.0:
        raise SystemExit("Episode steps, sigma, and learning rate must be positive")

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("agent") != "ppo":
        raise SystemExit("Evolution currently warm-starts PPO actor checkpoints")
    game_config = GameConfig(**checkpoint["game_config"])
    model = build_actor_critic(
        int(checkpoint["state_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
        str(checkpoint.get("policy_architecture", "flat")),
    ).to(device.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    portable_config = build_portable_config(checkpoint, args)
    names = policy_parameter_names(model)
    center = parameter_vector(model, names)
    optimizer = AdamAscent(center.numel())
    history: list[dict[str, object]] = []
    environment = PortableEnv(portable_config)
    try:
        baseline_seeds = [args.seed + index for index in range(args.episodes_per_candidate)]
        baseline = evaluate_policy(
            model, environment, game_config, baseline_seeds, stochastic=args.stochastic
        )
        for generation in range(args.generations):
            pair_count = args.population // 2
            noises = [
                seeded_noise(
                    center.numel(),
                    args.seed + 10_000 + generation * pair_count + pair,
                    device=center.device,
                    dtype=center.dtype,
                )
                for pair in range(pair_count)
            ]
            seeds = [
                args.seed
                + (generation + 1) * args.episodes_per_candidate
                + episode
                for episode in range(args.episodes_per_candidate)
            ]
            positive: list[CandidateResult] = []
            negative: list[CandidateResult] = []
            for noise in noises:
                set_parameter_vector(model, names, center + args.sigma * noise)
                positive.append(
                    evaluate_policy(
                        model, environment, game_config, seeds, stochastic=args.stochastic
                    )
                )
            for noise in noises:
                set_parameter_vector(model, names, center - args.sigma * noise)
                negative.append(
                    evaluate_policy(
                        model, environment, game_config, seeds, stochastic=args.stochastic
                    )
                )
            candidates = positive + negative
            utilities = centered_ranks(
                [candidate.quality for candidate in candidates]
            )
            gradient = antithetic_gradient(noises, utilities, args.sigma)
            center = optimizer.step(center, gradient, args.learning_rate)
            best_quality = max(candidate.quality for candidate in candidates)
            record = {
                "generation": generation,
                "seeds": seeds,
                "best_quality": best_quality,
                "mean_quality": float(np.mean([item.quality for item in candidates])),
                "candidates": [result_record(candidate) for candidate in candidates],
            }
            history.append(record)
            print(json.dumps(record), flush=True)

        set_parameter_vector(model, names, center)
        heldout_seeds = [
            args.seed + 1_000_000 + index for index in range(args.heldout_episodes)
        ]
        heldout = evaluate_policy(
            model, environment, game_config, heldout_seeds, stochastic=args.stochastic
        )
    finally:
        environment.close()

    evolution_metadata = {
        "algorithm": "warm-start-antithetic-outcome-es",
        "source_checkpoint": str(args.checkpoint),
        "level": portable_config.level,
        "seed_types": (
            None
            if portable_config.seed_types is None
            else list(portable_config.seed_types)
        ),
        "population": args.population,
        "generations": args.generations,
        "episodes_per_candidate": args.episodes_per_candidate,
        "sigma": args.sigma,
        "learning_rate": args.learning_rate,
        "parameter_count": center.numel(),
        "evolved_parameters": list(names),
        "common_random_numbers": True,
        "quality_definition": "mean episodic environment return",
        "diagnostics_not_optimized": [
            "wait_fraction",
            "plant_fraction",
            "collect_fraction",
            "shovel_fraction",
            "seed_slot_coverage",
            "lane_coverage",
            "mean_plant_column",
            "episode_cap_fraction",
            "zero_sun_plant_fraction",
        ],
        "baseline": result_record(baseline),
        "heldout": {**result_record(heldout), "seeds": heldout_seeds},
        "history": history,
    }
    output_checkpoint = dict(checkpoint)
    output_checkpoint.update(
        {
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "portable_config": portable_config.checkpoint_dict(),
            "training_method": "outcome-es",
            "initialization": {"checkpoint": str(args.checkpoint)},
            "evolution": evolution_metadata,
            "summary": {
                "heldout_win_rate": heldout.win_rate,
                "heldout_mean_return": heldout.mean_return,
                "heldout_truncation_rate": heldout.truncation_rate,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    metrics = {
        "output": str(args.output),
        "device": device.backend,
        **evolution_metadata,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
    with ReplayStore(args.experiment_database) as store:
        experiment_id = store.record_experiment(
            "warm-start-antithetic-outcome-es",
            {
                "population": args.population,
                "generations": args.generations,
                "episodes_per_candidate": args.episodes_per_candidate,
                "sigma": args.sigma,
                "learning_rate": args.learning_rate,
            },
            {
                "output": str(args.output),
                "baseline": result_record(baseline),
                "heldout": result_record(heldout),
            },
            agent="ppo-es",
            seed=args.seed,
            device=device.backend,
        )
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "output": str(args.output),
                "baseline": result_record(baseline),
                "heldout": result_record(heldout),
            }
        )
    )


if __name__ == "__main__":
    main()

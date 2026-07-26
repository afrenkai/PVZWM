from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from pvzwm.agents.goal_iql import (
    GoalIQLConfig,
    GoalRelabelConfig,
    train_goal_iql,
)
from pvzwm.device import select_device
from pvzwm.evaluation import build_world_model
from pvzwm.models.jepa import ActionConditionedJEPA


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a hindsight-relabeled goal-conditioned portable IQL worker"
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        help="Replay source to include; repeat for stratified multi-source training",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--latent-goal-index", type=Path)
    parser.add_argument("--jepa-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--advantage-temperature", type=float, default=3.0)
    parser.add_argument("--max-advantage-weight", type=float, default=100.0)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--next-goal-fraction", type=float, default=0.2)
    parser.add_argument("--future-goal-fraction", type=float, default=0.6)
    parser.add_argument("--random-goal-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sources = args.source or ["verified_archive_prefix"]

    relabel = GoalRelabelConfig(
        next_fraction=args.next_goal_fraction,
        future_fraction=args.future_goal_fraction,
        random_fraction=args.random_goal_fraction,
    )
    config = GoalIQLConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        expectile=args.expectile,
        advantage_temperature=args.advantage_temperature,
        max_advantage_weight=args.max_advantage_weight,
        target_tau=args.target_tau,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        relabel=relabel,
    )
    device = select_device(args.device)
    if (args.latent_goal_index is None) != (args.jepa_checkpoint is None):
        raise SystemExit("Latent goal training requires both index and JEPA checkpoint")
    latent_goal_model = None
    latent_goal_centers = None
    latent_goal_prototypes = None
    if args.latent_goal_index is not None:
        latent_index = torch.load(
            args.latent_goal_index, map_location="cpu", weights_only=True
        )
        latent_goal_centers = latent_index.get("centers")
        latent_goal_prototypes = latent_index.get("prototype_states")
        if not isinstance(latent_goal_centers, torch.Tensor) or not isinstance(
            latent_goal_prototypes, torch.Tensor
        ):
            raise SystemExit("Latent goal index is malformed")
        assert args.jepa_checkpoint is not None
        raw_jepa = torch.load(args.jepa_checkpoint, map_location="cpu", weights_only=True)
        rebuilt = build_world_model(raw_jepa).to(device.device).eval()
        if not isinstance(rebuilt, ActionConditionedJEPA):
            raise SystemExit("Latent goal training requires an action-conditioned JEPA")
        rebuilt.requires_grad_(False)
        latent_goal_model = rebuilt
    initializer = None
    if args.initialize_from is not None:
        initializer = torch.load(
            args.initialize_from, map_location="cpu", weights_only=True
        )
        if initializer.get("agent") != "goal_iql":
            raise SystemExit("Goal-IQL initialization requires a goal-IQL checkpoint")
        if int(initializer["hidden_dim"]) != args.hidden_dim:
            raise SystemExit("Goal-IQL initializer hidden dimension does not match")
        if not isinstance(initializer.get("auxiliary"), dict):
            raise SystemExit("Goal-IQL initializer has no auxiliary critic state")
    actor, result, auxiliary = train_goal_iql(
        args.database,
        device,
        config,
        source=sources,
        seed=args.seed,
        initial_state_dict=(
            None if initializer is None else initializer["state_dict"]
        ),
        initial_auxiliary=(
            None if initializer is None else initializer["auxiliary"]
        ),
        latent_goal_model=latent_goal_model,
        latent_goal_centers=latent_goal_centers,
        latent_goal_prototypes=latent_goal_prototypes,
    )
    checkpoint = {
        "agent": "goal_iql",
        "state_dim": actor.state_dim,
        "action_dim": actor.action_dim,
        "hidden_dim": actor.hidden_dim,
        "policy_architecture": "goal-conditioned-portable-autoregressive",
        "state_dict": actor.state_dict(),
        "auxiliary": auxiliary,
        "config": asdict(config),
        "database": str(args.database),
        "source": sources,
        "seed": args.seed,
        "initialization": (
            None
            if args.initialize_from is None
            else {"checkpoint": str(args.initialize_from)}
        ),
        "latent_goal_index": (
            None if args.latent_goal_index is None else str(args.latent_goal_index)
        ),
        "jepa_checkpoint": (
            None if args.jepa_checkpoint is None else str(args.jepa_checkpoint)
        ),
        "metrics": asdict(result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(
        json.dumps(
            {
                "device": device.backend,
                "output": str(args.output),
                **asdict(result),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

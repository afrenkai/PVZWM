from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from pvzwm.agents.goal_iql import (
    GoalConditionedAutoregressiveActor,
    GoalEpisode,
    GoalValueNetwork,
    load_goal_episodes,
)
from pvzwm.agents.latent_goals import JEPALatentGoalEncoder
from pvzwm.data.store import ReplayStore
from pvzwm.device import DeviceInfo, select_device
from pvzwm.evaluation import build_world_model
from pvzwm.models.jepa import ActionConditionedJEPA
from pvzwm.portable import (
    PortableEnv,
    PortableObservation,
    PortableProcessConfig,
    collect_sun_first_mask,
)
from pvzwm.portable_archive import load_portable_archive, resolve_archive_action


def parse_seed_types(value: str) -> tuple[int, ...]:
    seed_types = tuple(int(item.strip()) for item in value.split(","))
    if not 1 <= len(seed_types) <= 10:
        raise argparse.ArgumentTypeError("A seed loadout must contain 1 to 10 types")
    if len(set(seed_types)) != len(seed_types):
        raise argparse.ArgumentTypeError("A seed loadout cannot contain duplicates")
    if any(not 0 <= seed_type < 48 for seed_type in seed_types):
        raise argparse.ArgumentTypeError("Seed types must be in [0, 47]")
    return seed_types


def resolve_source_split_seed(
    checkpoint_sources: object,
    source: str,
    base_seed: int,
    *,
    allow_unseen_source: bool,
) -> tuple[int, bool]:
    if isinstance(checkpoint_sources, list):
        normalized = [str(item) for item in checkpoint_sources]
        if source in normalized:
            return base_seed + normalized.index(source), True
    elif isinstance(checkpoint_sources, str) and source == checkpoint_sources:
        return base_seed, True
    if not allow_unseen_source:
        raise ValueError("Requested source was not included in goal-IQL training")
    return base_seed, False


@dataclass(frozen=True, slots=True)
class ReachabilityTask:
    episode_id: str
    seed: int
    episode: GoalEpisode
    start: int
    goal_index: int
    offset: int
    goal_key: bytes
    goal_state: np.ndarray | None = None


def _restore_environment(
    config: PortableProcessConfig,
    seed: int,
    actions: np.ndarray,
    stop: int,
    attempts: int,
) -> tuple[PortableEnv | None, PortableObservation | None, int]:
    for attempt in range(1, attempts + 1):
        environment = PortableEnv(config)
        try:
            observation, _ = environment.reset(seed=seed)
            restored = True
            for replay_action in actions[:stop]:
                action, _ = resolve_archive_action(
                    int(replay_action),
                    environment.legal_action_mask(),
                    environment.codec,
                    True,
                )
                if action is None:
                    restored = False
                    break
                observation, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    restored = False
                    break
            if restored:
                return environment, observation, attempt
        except Exception:
            pass
        environment.close()
    return None, None, attempts


def _recorded_continuation_reaches(
    task: ReachabilityTask,
    config: PortableProcessConfig,
    encoder,
    restore_attempts: int,
) -> tuple[bool, int]:
    environment, observation, attempts = _restore_environment(
        config,
        task.seed,
        task.episode.actions,
        task.start,
        restore_attempts,
    )
    if environment is None or observation is None:
        return False, attempts
    try:
        for replay_action in task.episode.actions[task.start : task.goal_index]:
            action, _ = resolve_archive_action(
                int(replay_action),
                environment.legal_action_mask(),
                environment.codec,
                True,
            )
            if action is None:
                return False, attempts
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                return False, attempts
        return encoder.encode(observation.vector(config)) == task.goal_key, attempts
    finally:
        environment.close()


def _policy_reaches(
    task: ReachabilityTask,
    config: PortableProcessConfig,
    encoder,
    actor: GoalConditionedAutoregressiveActor,
    value: GoalValueNetwork,
    device: DeviceInfo,
    *,
    restore_attempts: int,
    horizon_multiplier: float,
    horizon_slack: int,
    stochastic: bool,
    generator: torch.Generator,
) -> dict[str, object]:
    environment, observation, attempts = _restore_environment(
        config,
        task.seed,
        task.episode.actions,
        task.start,
        restore_attempts,
    )
    if environment is None or observation is None:
        return {
            "restored": False,
            "restore_attempts": attempts,
            "success": False,
            "steps": 0,
            "predicted_value": None,
            "mowers_triggered": 0,
        }
    goal_array = (
        task.episode.states[task.goal_index]
        if task.goal_state is None
        else task.goal_state
    )
    goal = torch.from_numpy(goal_array).to(device.device)
    start_vector = torch.from_numpy(observation.vector(config)).to(device.device)
    with torch.inference_mode():
        predicted_value = float(value(start_vector[None], goal[None]).item())
    budget = max(
        task.offset + horizon_slack,
        int(np.ceil(task.offset * horizon_multiplier)),
    )
    try:
        for step in range(1, budget + 1):
            state = torch.from_numpy(observation.vector(config)).to(device.device)
            legal_mask = collect_sun_first_mask(
                environment.codec, environment.legal_action_mask()
            )
            legal = torch.from_numpy(legal_mask).to(device.device)
            with torch.inference_mode():
                logits, _ = actor(torch.cat([state, goal])[None])
                masked = logits[0].masked_fill(~legal, -torch.inf)
                if stochastic:
                    probabilities = torch.softmax(masked, dim=-1).cpu()
                    action = int(
                        torch.multinomial(probabilities, 1, generator=generator).item()
                    )
                else:
                    action = int(masked.argmax().item())
            observation, _, terminated, truncated, _ = environment.step(action)
            if encoder.encode(observation.vector(config)) == task.goal_key:
                return {
                    "restored": True,
                    "restore_attempts": attempts,
                    "success": True,
                    "steps": step,
                    "predicted_value": predicted_value,
                    "mowers_triggered": observation.mowers_triggered,
                }
            if terminated or truncated:
                break
        return {
            "restored": True,
            "restore_attempts": attempts,
            "success": False,
            "steps": budget,
            "predicted_value": predicted_value,
            "mowers_triggered": observation.mowers_triggered,
        }
    finally:
        environment.close()


def _build_tasks(
    database: Path,
    source: str,
    validation_fraction: float,
    split_seed: int,
    offsets: list[int],
    encoder,
    *,
    episode_limit: int | None,
    seed: int,
) -> list[ReachabilityTask]:
    with ReplayStore(database, read_only=True) as store:
        _, validation_ids = store.split_episode_ids(
            source,
            validation_fraction=validation_fraction,
            seed=split_seed,
        )
        if episode_limit is not None:
            validation_ids = validation_ids[:episode_limit]
        seeds = {
            episode_id: int(
                store.connection.execute(
                    "SELECT seed FROM episodes WHERE episode_id = ?", [episode_id]
                ).fetchone()[0]
            )
            for episode_id in validation_ids
        }
    episodes, _ = load_goal_episodes(
        database,
        source=source,
        episode_ids=validation_ids,
    )
    generator = np.random.default_rng(seed)
    tasks: list[ReachabilityTask] = []
    for episode_id, episode in zip(validation_ids, episodes, strict=True):
        for offset in offsets:
            candidates = []
            for start in range(0, len(episode.actions) - offset + 1):
                goal_index = start + offset
                if encoder.encode(episode.states[start]) != encoder.encode(
                    episode.states[goal_index]
                ):
                    candidates.append(start)
            if not candidates:
                continue
            start = candidates[int(generator.integers(len(candidates)))]
            goal_index = start + offset
            tasks.append(
                ReachabilityTask(
                    episode_id,
                    seeds[episode_id],
                    episode,
                    start,
                    goal_index,
                    offset,
                    encoder.encode(episode.states[goal_index]),
                    (
                        encoder.goal_state(episode.states[goal_index])
                        if hasattr(encoder, "goal_state")
                        else None
                    ),
                )
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate goal-IQL on held-out authentic prefix reachability"
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source", default="verified_archive_prefix")
    parser.add_argument(
        "--allow-unseen-source",
        action="store_true",
        help=(
            "Permit a replay source excluded from worker training for external "
            "failure calibration; the report records this explicitly"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--latent-goal-index", type=Path)
    parser.add_argument("--jepa-checkpoint", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, default=21)
    parser.add_argument("--ticks-per-step", type=int, default=25)
    parser.add_argument(
        "--seed-types",
        type=parse_seed_types,
        help="Exact replay seed-packet loadout for restoration calibration",
    )
    parser.add_argument("--offsets", default="1,4,16,64")
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--restore-attempts", type=int, default=4)
    parser.add_argument("--horizon-multiplier", type=float, default=2.0)
    parser.add_argument("--horizon-slack", type=int, default=8)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=330000)
    args = parser.parse_args()
    offsets = [int(value) for value in args.offsets.split(",")]
    if not offsets or any(value < 1 for value in offsets):
        raise SystemExit("Offsets must be positive")
    if args.restore_attempts < 1 or args.horizon_multiplier <= 0.0:
        raise SystemExit("Restore attempts and horizon multiplier must be positive")
    if args.horizon_slack < 0:
        raise SystemExit("Horizon slack cannot be negative")
    if (args.archive is None) == (args.latent_goal_index is None):
        raise SystemExit("Choose exactly one of --archive or --latent-goal-index")
    if args.latent_goal_index is not None and args.jepa_checkpoint is None:
        raise SystemExit("--latent-goal-index requires --jepa-checkpoint")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("agent") != "goal_iql":
        raise SystemExit("Checkpoint is not a goal-IQL worker")
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise SystemExit("Goal-IQL checkpoint has no training config")
    validation_fraction = float(raw_config.get("validation_fraction", 0.0))
    split_seed = int(raw_config.get("split_seed", 17))
    checkpoint_sources = checkpoint.get("source")
    try:
        split_seed, source_in_training = resolve_source_split_seed(
            checkpoint_sources,
            args.source,
            split_seed,
            allow_unseen_source=args.allow_unseen_source,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if validation_fraction <= 0.0:
        raise SystemExit("Goal-IQL checkpoint was not trained with a validation split")
    device = select_device(args.device)
    actor = GoalConditionedAutoregressiveActor(
        int(checkpoint["state_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device.device)
    actor.load_state_dict(checkpoint["state_dict"])
    actor.eval()
    value = GoalValueNetwork(
        int(checkpoint["state_dim"]), int(checkpoint["hidden_dim"])
    ).to(device.device)
    auxiliary = checkpoint.get("auxiliary")
    if not isinstance(auxiliary, dict) or "value_state_dict" not in auxiliary:
        raise SystemExit("Goal-IQL checkpoint has no value network")
    value.load_state_dict(auxiliary["value_state_dict"])
    value.eval()
    if args.latent_goal_index is None:
        assert args.archive is not None
        _, encoder = load_portable_archive(args.archive)
    else:
        latent_index = torch.load(
            args.latent_goal_index, map_location="cpu", weights_only=True
        )
        centers = latent_index.get("centers")
        prototypes = latent_index.get("prototype_states")
        if not isinstance(centers, torch.Tensor) or not isinstance(
            prototypes, torch.Tensor
        ):
            raise SystemExit("Latent goal index is malformed")
        assert args.jepa_checkpoint is not None
        raw_jepa = torch.load(
            args.jepa_checkpoint, map_location="cpu", weights_only=True
        )
        rebuilt = build_world_model(raw_jepa).to(device.device).eval()
        if not isinstance(rebuilt, ActionConditionedJEPA):
            raise SystemExit("Latent reachability requires an action-conditioned JEPA")
        encoder = JEPALatentGoalEncoder(
            rebuilt, centers, prototypes, device.device
        )
    tasks = _build_tasks(
        args.database,
        args.source,
        validation_fraction,
        split_seed,
        offsets,
        encoder,
        episode_limit=args.episode_limit,
        seed=args.seed,
    )
    if not tasks:
        raise SystemExit("No nontrivial held-out reachability tasks were available")
    config = PortableProcessConfig(
        binary=args.binary,
        resources=args.resources,
        level=args.level,
        ticks_per_step=args.ticks_per_step,
        max_steps=3_600,
        frame_width=0,
        frame_height=0,
        reward_mode="bridge",
        seed_types=args.seed_types,
    )
    generator = torch.Generator().manual_seed(args.seed)
    details: list[dict[str, object]] = []
    for index, task in enumerate(tasks, start=1):
        replay_success, replay_attempts = _recorded_continuation_reaches(
            task, config, encoder, args.restore_attempts
        )
        policy = _policy_reaches(
            task,
            config,
            encoder,
            actor,
            value,
            device,
            restore_attempts=args.restore_attempts,
            horizon_multiplier=args.horizon_multiplier,
            horizon_slack=args.horizon_slack,
            stochastic=args.stochastic,
            generator=generator,
        )
        details.append(
            {
                "task": index,
                "episode_id": task.episode_id,
                "seed": task.seed,
                "start": task.start,
                "goal_index": task.goal_index,
                "offset": task.offset,
                "recorded_replay_success": replay_success,
                "recorded_replay_restore_attempts": replay_attempts,
                **policy,
            }
        )
        if index % 10 == 0 or index == len(tasks):
            eligible = [item for item in details if item["recorded_replay_success"]]
            print(
                json.dumps(
                    {
                        "completed": index,
                        "tasks": len(tasks),
                        "eligible": len(eligible),
                        "successes": sum(bool(item["success"]) for item in eligible),
                    }
                ),
                flush=True,
            )
    eligible = [item for item in details if item["recorded_replay_success"]]
    successes = [item for item in eligible if item["success"]]
    per_offset = {}
    for offset in offsets:
        group = [item for item in eligible if item["offset"] == offset]
        per_offset[str(offset)] = {
            "eligible": len(group),
            "successes": sum(bool(item["success"]) for item in group),
            "success_rate": (
                sum(bool(item["success"]) for item in group) / len(group)
                if group
                else 0.0
            ),
        }
    summary = {
        "checkpoint": str(args.checkpoint),
        "database": str(args.database),
        "database_location": "remote_only",
        "source": args.source,
        "source_in_training": source_in_training,
        "allow_unseen_source": args.allow_unseen_source,
        "archive": None if args.archive is None else str(args.archive),
        "latent_goal_index": (
            None if args.latent_goal_index is None else str(args.latent_goal_index)
        ),
        "jepa_checkpoint": (
            None if args.jepa_checkpoint is None else str(args.jepa_checkpoint)
        ),
        "level": args.level,
        "seed_types": None if args.seed_types is None else list(args.seed_types),
        "mode": "stochastic" if args.stochastic else "argmax",
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "offsets": offsets,
        "tasks": len(tasks),
        "recorded_replay_reachable": len(eligible),
        "policy_restored": sum(bool(item["restored"]) for item in eligible),
        "policy_successes": len(successes),
        "policy_success_rate": len(successes) / len(eligible) if eligible else 0.0,
        "mean_success_steps": (
            float(np.mean([int(item["steps"]) for item in successes]))
            if successes
            else None
        ),
        "mean_predicted_value": (
            float(
                np.mean(
                    [
                        float(item["predicted_value"])
                        for item in eligible
                        if item["predicted_value"] is not None
                    ]
                )
            )
            if eligible
            else None
        ),
        "mower_episodes": sum(int(item["mowers_triggered"]) > 0 for item in eligible),
        "per_offset": per_offset,
        "reward_shaping": False,
        "game_specific_goal_features": False,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}))


if __name__ == "__main__":
    main()

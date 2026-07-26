from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pvzwm.agents.goal_iql import (
    GoalConditionedAutoregressiveActor,
    GoalValueNetwork,
    load_goal_episodes,
)
from pvzwm.agents.evolution import behavior_descriptor, zero_sun_plant_fraction
from pvzwm.agents.latent_goals import (
    cluster_key,
    frontier_goals_from_checkpoint,
    normalize_latents,
)
from pvzwm.agents.goal_selector import (
    FrontierGoal,
    fit_competence_threshold,
    select_frontier_goal,
)
from pvzwm.data.store import ReplayStore, Transition
from pvzwm.device import select_device
from pvzwm.evaluation import build_world_model
from pvzwm.models.jepa import ActionConditionedJEPA
from pvzwm.portable import PortableEnv, PortableProcessConfig, collect_sun_first_mask
from pvzwm.portable_archive import load_portable_archive


def parse_seed_types(value: str) -> tuple[int, ...]:
    seed_types = tuple(int(item.strip()) for item in value.split(","))
    if not 1 <= len(seed_types) <= 10:
        raise argparse.ArgumentTypeError("A seed loadout must contain 1 to 10 types")
    if len(set(seed_types)) != len(seed_types):
        raise argparse.ArgumentTypeError("A seed loadout cannot contain duplicates")
    if any(not 0 <= seed_type < 48 for seed_type in seed_types):
        raise argparse.ArgumentTypeError("Seed types must be in [0, 47]")
    return seed_types


def load_calibration(paths: list[Path]) -> tuple[float, int, int]:
    values: list[float] = []
    outcomes: list[bool] = []
    for path in paths:
        payload = json.loads(path.read_text())
        details = payload.get("details")
        if not isinstance(details, list):
            raise ValueError(f"Reachability report {path} has no details")
        for detail in details:
            if not isinstance(detail, dict) or not detail.get(
                "recorded_replay_success", False
            ):
                continue
            predicted_value = detail.get("predicted_value")
            if predicted_value is None:
                continue
            values.append(float(predicted_value))
            outcomes.append(bool(detail.get("success", False)))
    threshold = fit_competence_threshold(values, outcomes)
    return threshold, len(values), sum(outcomes)


def load_frontier_goals(
    database: Path, source: str, encoder, stride: int
) -> list[FrontierGoal]:
    episodes, _ = load_goal_episodes(database, source=source)
    by_key: dict[bytes, FrontierGoal] = {}
    for episode in episodes:
        indices = set(range(stride, len(episode.states), stride))
        indices.add(len(episode.states) - 1)
        for depth in sorted(indices):
            state = episode.states[depth].astype(np.float32, copy=False)
            key = encoder.encode(state)
            candidate = FrontierGoal(state, key, depth)
            existing = by_key.get(key)
            if existing is None or candidate.depth > existing.depth:
                by_key[key] = candidate
    return list(by_key.values())


def predict_values(
    value: GoalValueNetwork,
    current: np.ndarray,
    candidates: list[FrontierGoal],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    state = torch.from_numpy(current).to(device)
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            goals = torch.from_numpy(
                np.stack(
                    [candidate.state for candidate in candidates[start : start + batch_size]]
                )
            ).to(device)
            states = state.expand(len(goals), -1)
            predictions.append(value(states, goals).cpu().numpy())
    return np.concatenate(predictions)


@torch.inference_mode()
def latent_goal_key(
    model: ActionConditionedJEPA,
    centers: torch.Tensor,
    state: np.ndarray,
    device: torch.device,
) -> bytes:
    tensor = torch.from_numpy(state).to(device).unsqueeze(0)
    latent = normalize_latents(model.encode(tensor))
    cluster = int((latent @ centers.T).argmax(1).item())
    return cluster_key(cluster)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a held-out-calibrated goal-IQL worker as a fresh-start "
            "generic archive-frontier hierarchy"
        )
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source", default="verified_archive_prefix")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--latent-goal-index",
        type=Path,
        help="VISReg-JEPA cosine-cluster goal index replacing exact archive keys",
    )
    parser.add_argument(
        "--jepa-checkpoint",
        type=Path,
        help="Action-conditioned JEPA used by --latent-goal-index",
    )
    parser.add_argument(
        "--calibration-report", type=Path, action="append", required=True
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path)
    parser.add_argument("--replay-source", default="goal_iql_fresh_frontier")
    parser.add_argument("--level", type=int, default=21)
    parser.add_argument("--ticks-per-step", type=int, default=25)
    parser.add_argument("--episode-steps", type=int, default=3_600)
    parser.add_argument(
        "--seed-types",
        type=parse_seed_types,
        help="Exact authentic ordered seed-packet loadout for this hierarchy gate",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--goal-stride", type=int, default=4)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--max-option-steps", type=int, default=136)
    parser.add_argument("--value-batch-size", type=int, default=1024)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=338_000)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.episodes < 1 or args.goal_stride < 1 or args.max_options < 1:
        raise SystemExit("Episode, stride, and option counts must be positive")
    if args.max_option_steps < 1 or args.value_batch_size < 1:
        raise SystemExit("Option steps and value batch size must be positive")
    if (args.archive is None) == (args.latent_goal_index is None):
        raise SystemExit("Choose exactly one of --archive or --latent-goal-index")
    if args.latent_goal_index is not None and args.jepa_checkpoint is None:
        raise SystemExit("--latent-goal-index requires --jepa-checkpoint")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("agent") != "goal_iql":
        raise SystemExit("Checkpoint is not a goal-IQL worker")
    device = select_device(args.device)
    actor = GoalConditionedAutoregressiveActor(
        int(checkpoint["state_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device.device)
    actor.load_state_dict(checkpoint["state_dict"])
    actor.eval()
    auxiliary = checkpoint.get("auxiliary")
    if not isinstance(auxiliary, dict) or "value_state_dict" not in auxiliary:
        raise SystemExit("Goal-IQL checkpoint has no value network")
    value = GoalValueNetwork(
        int(checkpoint["state_dim"]), int(checkpoint["hidden_dim"])
    ).to(device.device)
    value.load_state_dict(auxiliary["value_state_dict"])
    value.eval()
    encoder = None
    latent_model = None
    latent_centers = None
    if args.latent_goal_index is None:
        assert args.archive is not None
        _, encoder = load_portable_archive(args.archive)
        candidates = load_frontier_goals(
            args.database, args.source, encoder, args.goal_stride
        )
    else:
        latent_index = torch.load(
            args.latent_goal_index, map_location="cpu", weights_only=True
        )
        candidates, latent_centers = frontier_goals_from_checkpoint(latent_index)
        assert args.jepa_checkpoint is not None
        raw_jepa = torch.load(
            args.jepa_checkpoint, map_location="cpu", weights_only=True
        )
        rebuilt = build_world_model(raw_jepa).to(device.device).eval()
        if not isinstance(rebuilt, ActionConditionedJEPA):
            raise SystemExit("Latent frontier requires an action-conditioned JEPA")
        latent_model = rebuilt
        latent_centers = latent_centers.to(device.device)
    threshold, calibration_count, calibration_successes = load_calibration(
        args.calibration_report
    )
    config = PortableProcessConfig(
        binary=args.binary,
        resources=args.resources,
        level=args.level,
        ticks_per_step=args.ticks_per_step,
        max_steps=args.episode_steps,
        frame_width=0,
        frame_height=0,
        reward_mode="bridge",
        seed_types=args.seed_types,
    )
    generator = torch.Generator().manual_seed(args.seed)
    episode_records: list[dict[str, object]] = []
    replay_store = None if args.replay_output is None else ReplayStore(args.replay_output)
    for episode_index in range(args.episodes):
        episode_seed = args.seed + episode_index
        environment = PortableEnv(config)
        option_records: list[dict[str, object]] = []
        blocked_keys: set[bytes] = set()
        maximum_depth = 0
        failure_ceiling: int | None = None
        total_steps = 0
        episode_return = 0.0
        actions: list[int] = []
        sun_before_actions: list[float] = []
        hierarchy_exhausted = False
        fallback_steps = 0
        observation, _ = environment.reset(seed=episode_seed)
        writer = (
            None
            if replay_store is None
            else replay_store.episode(
                args.replay_source,
                episode_seed,
                {
                    "level_id": "3-1",
                    "controller": (
                        "jepa_cluster_goal_iql_frontier"
                        if latent_model is not None
                        else "heldout_calibrated_goal_iql_frontier"
                    ),
                    "mode": "stochastic" if args.stochastic else "argmax",
                    "reward_shaping": False,
                    "game_specific_goal_features": False,
                },
            )
        )
        terminated = False
        truncated = False
        try:
            for option_index in range(args.max_options):
                current = observation.vector(config)
                current_key = (
                    encoder.encode(current)
                    if encoder is not None
                    else latent_goal_key(
                        latent_model, latent_centers, current, device.device
                    )
                )
                predictions = predict_values(
                    value,
                    current,
                    candidates,
                    device.device,
                    args.value_batch_size,
                )
                selected = select_frontier_goal(
                    candidates,
                    predictions,
                    minimum_depth=maximum_depth,
                    competence_threshold=threshold,
                    blocked_keys=blocked_keys | {current_key},
                    maximum_depth=failure_ceiling,
                )
                if selected is None:
                    hierarchy_exhausted = True
                    break
                goal, predicted_value = selected
                goal_tensor = torch.from_numpy(goal.state).to(device.device)
                reached = False
                option_steps = 0
                for _ in range(
                    min(args.max_option_steps, args.episode_steps - total_steps)
                ):
                    state_array = observation.vector(config)
                    state = torch.from_numpy(state_array).to(device.device)
                    raw_legal_mask = environment.legal_action_mask()
                    legal_mask = collect_sun_first_mask(
                        environment.codec, raw_legal_mask
                    )
                    legal = torch.from_numpy(legal_mask).to(device.device)
                    with torch.inference_mode():
                        logits, _ = actor(torch.cat([state, goal_tensor])[None])
                        masked = logits[0].masked_fill(~legal, -torch.inf)
                        if args.stochastic:
                            probabilities = torch.softmax(masked, dim=-1).cpu()
                            action = int(
                                torch.multinomial(
                                    probabilities, 1, generator=generator
                                ).item()
                            )
                        else:
                            action = int(masked.argmax().item())
                    actions.append(action)
                    sun_before_actions.append(
                        float(observation.payload.get("sun", 0.0))
                    )
                    observation, reward, terminated, truncated, _ = environment.step(
                        action
                    )
                    if writer is not None:
                        writer.append(
                            Transition(
                                state_array,
                                action,
                                reward,
                                observation.vector(config),
                                terminated,
                                truncated,
                                raw_legal_mask,
                                environment.legal_action_mask(),
                            )
                        )
                    total_steps += 1
                    option_steps += 1
                    episode_return += reward
                    observed_state = observation.vector(config)
                    observed_key = (
                        encoder.encode(observed_state)
                        if encoder is not None
                        else latent_goal_key(
                            latent_model,
                            latent_centers,
                            observed_state,
                            device.device,
                        )
                    )
                    if observed_key == goal.key:
                        reached = True
                        maximum_depth = max(maximum_depth, goal.depth)
                        break
                    if terminated or truncated or total_steps >= args.episode_steps:
                        break
                if not reached:
                    blocked_keys.add(goal.key)
                    failure_ceiling = (
                        goal.depth
                        if failure_ceiling is None
                        else min(failure_ceiling, goal.depth)
                    )
                else:
                    failure_ceiling = None
                option_records.append(
                    {
                        "option": option_index,
                        "goal_depth": goal.depth,
                        "predicted_value": predicted_value,
                        "reached": reached,
                        "steps": option_steps,
                    }
                )
                if terminated or truncated or total_steps >= args.episode_steps:
                    break
            while (
                not terminated
                and not truncated
                and total_steps < args.episode_steps
            ):
                raw_legal = environment.legal_action_mask()
                prioritized = collect_sun_first_mask(environment.codec, raw_legal)
                action = (
                    environment.codec.collect_action
                    if prioritized[environment.codec.collect_action]
                    else 0
                )
                state_array = observation.vector(config)
                actions.append(action)
                sun_before_actions.append(
                    float(observation.payload.get("sun", 0.0))
                )
                observation, reward, terminated, truncated, _ = environment.step(
                    action
                )
                if writer is not None:
                    writer.append(
                        Transition(
                            state_array,
                            action,
                            reward,
                            observation.vector(config),
                            terminated,
                            truncated,
                            raw_legal,
                            environment.legal_action_mask(),
                        )
                    )
                total_steps += 1
                fallback_steps += 1
                episode_return += reward
        finally:
            if writer is not None:
                writer.metadata.update(
                    {
                        "maximum_reached_archive_depth": maximum_depth,
                        "hierarchy_exhausted": hierarchy_exhausted,
                        "completion_fallback_steps": fallback_steps,
                        "won": bool(observation.won),
                    }
                )
                writer.close()
            environment.close()
        record = {
            "episode": episode_index,
            "seed": episode_seed,
            "won": bool(observation.won),
            "terminated": terminated,
            "truncated": truncated,
            "return": episode_return,
            "steps": total_steps,
            "maximum_reached_archive_depth": maximum_depth,
            "mowers_triggered": observation.mowers_triggered,
            "hierarchy_exhausted": hierarchy_exhausted,
            "completion_fallback_steps": fallback_steps,
            "behavior_descriptor": behavior_descriptor(
                actions, args.episode_steps
            ).tolist(),
            "zero_sun_plant_fraction_diagnostic": zero_sun_plant_fraction(
                actions, sun_before_actions
            ),
            "options": option_records,
        }
        episode_records.append(record)
        print(json.dumps(record), flush=True)

    if replay_store is not None:
        replay_store.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "database": str(args.database),
        "database_location": "remote_only",
        "replay_output": None if args.replay_output is None else str(args.replay_output),
        "replay_source": args.replay_source if args.replay_output is not None else None,
        "archive": None if args.archive is None else str(args.archive),
        "latent_goal_index": (
            None if args.latent_goal_index is None else str(args.latent_goal_index)
        ),
        "jepa_checkpoint": (
            None if args.jepa_checkpoint is None else str(args.jepa_checkpoint)
        ),
        "goal_key_semantics": (
            "jepa_cosine_cluster"
            if args.latent_goal_index is not None
            else "random_projection_archive_cell"
        ),
        "level": args.level,
        "seed_types": None if args.seed_types is None else list(args.seed_types),
        "mode": "stochastic" if args.stochastic else "argmax",
        "episodes": args.episodes,
        "wins": sum(bool(record["won"]) for record in episode_records),
        "mean_return": float(np.mean([record["return"] for record in episode_records])),
        "mean_maximum_reached_archive_depth": float(
            np.mean(
                [record["maximum_reached_archive_depth"] for record in episode_records]
            )
        ),
        "maximum_reached_archive_depth": max(
            int(record["maximum_reached_archive_depth"])
            for record in episode_records
        ),
        "competence_threshold": threshold,
        "calibration_tasks": calibration_count,
        "calibration_successes": calibration_successes,
        "frontier_candidates": len(candidates),
        "goal_stride": args.goal_stride,
        "max_option_steps": args.max_option_steps,
        "reward_shaping": False,
        "game_specific_goal_features": False,
        "mower_episodes": sum(
            int(record["mowers_triggered"]) > 0 for record in episode_records
        ),
        "episodes_requiring_completion_fallback": sum(
            int(record["completion_fallback_steps"]) > 0
            for record in episode_records
        ),
        "episode_records": episode_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "episode_records"}))


if __name__ == "__main__":
    main()

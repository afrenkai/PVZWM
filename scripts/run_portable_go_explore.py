from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from pvzwm.agents.go_explore import (
    RandomProjectionCellEncoder,
    TrajectoryArchive,
)
from pvzwm.agents.ppo import build_actor_critic
from pvzwm.data.store import ReplayStore, Transition
from pvzwm.device import select_device
from pvzwm.portable import (
    PortableActionCodec,
    PortableEnv,
    PortableProcessConfig,
    collect_sun_first_mask,
    portable_process_config_from_checkpoint,
)
from pvzwm.portable_archive import resolve_archive_action


def factored_random_action_groups(
    legal_mask: np.ndarray, codec: PortableActionCodec
) -> list[np.ndarray]:
    """Partition legal actions by semantic family and seed slot.

    Uniform sampling over the flat 650-way vocabulary strongly overweights
    board coordinates and makes the singleton WAIT action almost unreachable.
    The groups preserve all legal actions while removing this cardinality bias.
    """

    legal = np.asarray(legal_mask, dtype=np.bool_)
    if legal.shape != (codec.size,):
        raise ValueError(f"Expected legal mask shape {(codec.size,)}, got {legal.shape}")
    groups: list[np.ndarray] = []

    def append_range(begin: int, end: int) -> None:
        actions = np.flatnonzero(legal[begin:end]) + begin
        if actions.size:
            groups.append(actions)

    append_range(0, 1)
    for slot in range(codec.seed_slots):
        begin = codec.plant_begin + slot * codec.cells
        append_range(begin, begin + codec.cells)
    append_range(codec.shovel_begin, codec.board_click_begin)
    append_range(codec.board_click_begin, codec.collect_action)
    append_range(codec.collect_action, codec.collect_action + 1)
    return groups


def action_branch(action: int, codec: PortableActionCodec) -> tuple[str, int]:
    decoded = codec.decode(action)
    if decoded.kind.name == "PLANT":
        return ("plant", decoded.slot)
    return (decoded.kind.name.lower(), 0)


def archive_branch_ids(
    archive: TrajectoryArchive,
    codec: PortableActionCodec,
    trace_window: int = 1,
) -> list[tuple[tuple[str, int], tuple[tuple[str, int], ...]]]:
    """Describe both the latest action and recent semantic behavior.

    Last-action rarity disappears immediately after a rare interaction.  A
    bounded trace signature keeps that interaction visible while exploring its
    consequences, without assigning meaning or value to any particular seed
    slot.  Including the latest branch separately preserves the original
    selector's distinction between immediate action families.
    """

    if trace_window < 1:
        raise ValueError("Branch trace window must be positive")

    def signature(actions: tuple[int, ...]) -> tuple[
        tuple[str, int], tuple[tuple[str, int], ...]
    ]:
        if not actions:
            initial = ("initial", 0)
            return initial, (initial,)
        latest = action_branch(actions[-1], codec)
        recent = tuple(
            sorted({action_branch(action, codec) for action in actions[-trace_window:]})
        )
        return latest, recent

    return [
        signature(cell.actions)
        for cell in archive.cells.values()
    ]


def random_legal_action(
    legal_mask: np.ndarray,
    codec: PortableActionCodec,
    generator: np.random.Generator,
    sampling: str,
) -> int:
    if sampling == "flat":
        legal = np.flatnonzero(legal_mask)
        if legal.size == 0:
            raise RuntimeError("Portable engine returned no legal actions")
        return int(generator.choice(legal))
    if sampling != "factored":
        raise ValueError(f"Unsupported random action sampling: {sampling}")
    groups = factored_random_action_groups(legal_mask, codec)
    if not groups:
        raise RuntimeError("Portable engine returned no legal actions")
    group = groups[int(generator.integers(len(groups)))]
    return int(generator.choice(group))


def resolve_restore_action(
    replay_action: int,
    raw_legal_mask: np.ndarray,
    codec: PortableActionCodec,
    tolerate_collect_timing: bool,
) -> tuple[int | None, bool]:
    """Resolve a recorded action under small collectible-timing variation.

    Collection is deterministic housekeeping in this project. If a recorded
    collect is unavailable on replay, WAIT advances the same number of engine
    ticks without inventing a game-specific control action. Other mismatches
    remain hard restore failures.
    """

    return resolve_archive_action(
        replay_action, raw_legal_mask, codec, tolerate_collect_timing
    )


def load_archive(
    path: Path, encoder: RandomProjectionCellEncoder
) -> TrajectoryArchive:
    if not path.exists():
        return TrajectoryArchive()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Archive checkpoint must contain an object")
    return TrajectoryArchive.from_payload(payload, encoder)


def save_archive(
    path: Path,
    archive: TrajectoryArchive,
    encoder: RandomProjectionCellEncoder,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(archive.payload(encoder), handle, separators=(",", ":"))
    temporary.replace(path)


def policy_action(
    model: torch.nn.Module,
    state: np.ndarray,
    legal_mask: np.ndarray,
    codec: PortableActionCodec,
    generator: np.random.Generator,
    epsilon: float,
    temperature: float,
    random_action_sampling: str,
    device: torch.device,
) -> int:
    legal = np.flatnonzero(legal_mask)
    if legal.size == 0:
        raise RuntimeError("Portable engine returned no legal actions")
    if generator.random() < epsilon:
        return random_legal_action(
            legal_mask, codec, generator, random_action_sampling
        )
    with torch.inference_mode():
        logits, _ = model(
            torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)
        )
        masked = logits.squeeze(0).masked_fill(
            ~torch.from_numpy(legal_mask).to(device), -torch.inf
        )
        probabilities = torch.softmax(masked / temperature, dim=-1).cpu().numpy()
    return int(generator.choice(probabilities.size, p=probabilities))


def record_trace(
    store: ReplayStore,
    seed: int,
    level: int,
    start_key: bytes,
    trace: list[Transition],
    *,
    won: bool,
    restored: bool,
    new_cells: int,
) -> None:
    with store.episode(
        "pvz_portable_go_explore",
        seed,
        {
            "policy": "go_explore_random_projection_v1",
            "portable_level": level,
            "level_id": f"{(level - 1) // 10 + 1}-{(level - 1) % 10 + 1}",
            "archive_start_key": start_key.hex(),
            "restored": restored,
            "new_cells": new_cells,
            "won": won,
            "action_semantics": "collect_sun_first_v1",
        },
    ) as writer:
        for transition in trace:
            writer.append(transition)


def make_config(args: argparse.Namespace, checkpoint: dict[str, object]) -> PortableProcessConfig:
    raw = checkpoint.get("portable_config")
    if not isinstance(raw, dict):
        raise SystemExit("Checkpoint has no portable engine configuration")
    config = portable_process_config_from_checkpoint(
        raw, state_dim=int(checkpoint["state_dim"])
    )
    return replace(
        config,
        binary=args.portable_binary,
        resources=args.portable_resources,
        level=args.level,
        max_steps=args.episode_steps,
        frame_width=0,
        frame_height=0,
        reward_mode="clipped-damage",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic action-prefix Go-Explore on the authentic portable engine"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--portable-binary", type=Path, required=True)
    parser.add_argument("--portable-resources", type=Path, required=True)
    parser.add_argument("--level", type=int, choices=range(1, 51), required=True)
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--initial-seeds", type=int, default=8)
    parser.add_argument("--explore-steps", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=3_600)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--random-action-sampling",
        choices=("flat", "factored"),
        default="flat",
        help="Sample epsilon actions flatly or by semantic family/seed slot",
    )
    parser.add_argument(
        "--tolerate-collect-timing",
        action="store_true",
        help="Replace unavailable archived COLLECT actions with WAIT during restore",
    )
    parser.add_argument("--cell-projections", type=int, default=20)
    parser.add_argument("--cell-width", type=float, default=0.02)
    parser.add_argument(
        "--selection",
        choices=("count", "frontier", "frontier-branch"),
        default="count",
        help="Archive return-cell selection strategy",
    )
    parser.add_argument(
        "--frontier-fraction",
        type=float,
        default=0.2,
        help="Deepest archive fraction prioritized by frontier selection",
    )
    parser.add_argument(
        "--branch-trace-window",
        type=int,
        default=1,
        help=(
            "Decisions of recent semantic action history retained by "
            "frontier-branch selection"
        ),
    )
    parser.add_argument("--seed", type=int, default=260_000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.initial_seeds < 1 or args.explore_steps < 1:
        raise SystemExit("Iteration, seed, and exploration counts must be positive")
    if not 0.0 <= args.epsilon <= 1.0 or args.temperature <= 0.0:
        raise SystemExit("Invalid epsilon or temperature")
    if not 0.0 < args.frontier_fraction <= 1.0:
        raise SystemExit("Frontier fraction must be in (0, 1]")
    if args.branch_trace_window < 1:
        raise SystemExit("Branch trace window must be positive")

    device = select_device(args.device)
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device.device, weights_only=True)
    config = make_config(args, checkpoint)
    model = build_actor_critic(
        int(checkpoint["state_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
        str(checkpoint.get("policy_architecture", "flat")),
    ).to(device.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    encoder = RandomProjectionCellEncoder(
        int(checkpoint["state_dim"]),
        args.cell_projections,
        args.seed,
        args.cell_width,
    )
    archive = load_archive(args.archive, encoder)

    if not archive.cells:
        for offset in range(args.initial_seeds):
            env = PortableEnv(config)
            try:
                observation, _ = env.reset(seed=args.seed + offset)
                state = observation.vector(config)
                archive.add(encoder.encode(state), args.seed + offset, [], 0.0)
            finally:
                env.close()

    wins = 0
    restore_failures = 0
    approximate_restores = 0
    collect_timing_substitutions = 0
    recorded_episodes = 0
    total_new_cells = 0
    best_return = -float("inf")
    best_actions: list[int] = []
    iterations_completed = 0
    with ReplayStore(args.database) as store:
        for iteration in range(args.iterations):
            iterations_completed = iteration + 1
            start = archive.select(
                generator,
                strategy=args.selection,
                frontier_fraction=args.frontier_fraction,
                branch_ids=(
                    archive_branch_ids(
                        archive,
                        PortableActionCodec(),
                        trace_window=args.branch_trace_window,
                    )
                    if args.selection == "frontier-branch"
                    else None
                ),
            )
            env = PortableEnv(config)
            trace: list[Transition] = []
            actions: list[int] = []
            episode_return = 0.0
            new_cells = 0
            restored = True
            restore_was_approximate = False
            terminated = False
            truncated = False
            try:
                observation, _ = env.reset(seed=start.seed)
                state = observation.vector(config)
                for replay_action in start.actions:
                    raw_legal_mask = env.legal_action_mask()
                    executed_action, substituted = resolve_restore_action(
                        replay_action,
                        raw_legal_mask,
                        env.codec,
                        args.tolerate_collect_timing,
                    )
                    if executed_action is None:
                        restored = False
                        restore_failures += 1
                        break
                    legal_mask = collect_sun_first_mask(env.codec, raw_legal_mask)
                    if substituted:
                        legal_mask = raw_legal_mask.copy()
                        collect_timing_substitutions += 1
                        restore_was_approximate = True
                    next_observation, reward, terminated, truncated, _ = env.step(
                        executed_action
                    )
                    next_state = next_observation.vector(config)
                    trace.append(
                        Transition(
                            state,
                            executed_action,
                            reward,
                            next_state,
                            terminated,
                            truncated,
                            legal_mask,
                        )
                    )
                    actions.append(executed_action)
                    episode_return += reward
                    observation = next_observation
                    state = next_state
                    if terminated or truncated:
                        restored = False
                        restore_failures += 1
                        break

                current_key = encoder.encode(state)
                if restored and current_key != start.key:
                    if args.tolerate_collect_timing and restore_was_approximate:
                        approximate_restores += 1
                        archive.add(
                            current_key,
                            start.seed,
                            actions,
                            episode_return,
                        )
                    else:
                        restored = False
                        restore_failures += 1

                if restored:
                    for _ in range(args.explore_steps):
                        legal_mask = collect_sun_first_mask(
                            env.codec, env.legal_action_mask()
                        )
                        action = policy_action(
                            model,
                            state,
                            legal_mask,
                            env.codec,
                            generator,
                            args.epsilon,
                            args.temperature,
                            args.random_action_sampling,
                            device.device,
                        )
                        next_observation, reward, terminated, truncated, _ = env.step(action)
                        next_state = next_observation.vector(config)
                        trace.append(
                            Transition(
                                state,
                                action,
                                reward,
                                next_state,
                                terminated,
                                truncated,
                                legal_mask,
                            )
                        )
                        actions.append(action)
                        episode_return += reward
                        observation = next_observation
                        state = next_state
                        if not (terminated or truncated):
                            key = encoder.encode(state)
                            update = archive.add(
                                key,
                                start.seed,
                                actions,
                                episode_return,
                                parent_key=current_key,
                            )
                            new_cells += int(update.inserted)
                            current_key = key
                        if terminated or truncated:
                            break

                won = bool(observation.won)
                wins += int(won)
                total_new_cells += new_cells
                if episode_return > best_return:
                    best_return = episode_return
                    best_actions = list(actions)
                if new_cells or won:
                    if trace and not (terminated or truncated):
                        trace[-1] = replace(trace[-1], truncated=True)
                    record_trace(
                        store,
                        start.seed,
                        args.level,
                        start.key,
                        trace,
                        won=won,
                        restored=restored,
                        new_cells=new_cells,
                    )
                    recorded_episodes += 1
                if won:
                    break
            finally:
                env.close()

            if (iteration + 1) % args.save_every == 0:
                save_archive(args.archive, archive, encoder)

    save_archive(args.archive, archive, encoder)
    result = {
        "method": "go_explore_random_projection_v1",
        "reference": "https://arxiv.org/abs/2004.12919",
        "checkpoint": str(args.checkpoint),
        "level": args.level,
        "iterations_requested": args.iterations,
        "iterations_completed": iterations_completed,
        "archive_cells": len(archive.cells),
        "new_cells": total_new_cells,
        "recorded_episodes": recorded_episodes,
        "restore_failures": restore_failures,
        "approximate_restores": approximate_restores,
        "collect_timing_substitutions": collect_timing_substitutions,
        "wins": wins,
        "best_return": best_return,
        "best_actions": best_actions,
        "epsilon": args.epsilon,
        "temperature": args.temperature,
        "random_action_sampling": args.random_action_sampling,
        "tolerate_collect_timing": args.tolerate_collect_timing,
        "cell_projections": args.cell_projections,
        "cell_width": args.cell_width,
        "selection": args.selection,
        "frontier_fraction": args.frontier_fraction,
        "branch_trace_window": args.branch_trace_window,
        "seed": args.seed,
        "database": str(args.database),
        "archive": str(args.archive),
        "action_semantics": "collect_sun_first_v1",
        "game_specific_archive_features": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

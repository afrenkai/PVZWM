from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pvzwm.agents.distillation import (
    balanced_level_batch_indices,
    masked_policy_distillation_loss,
)
from pvzwm.agents.ppo import PolicyModel, build_actor_critic
from pvzwm.device import select_device
from pvzwm.domain import GameConfig
from pvzwm.portable import (
    PortableEnv,
    collect_sun_first_mask,
    portable_process_config_from_checkpoint,
)
from pvzwm.portable_collection import parse_adventure_levels


@dataclass(slots=True)
class TeacherStates:
    level_id: str
    states: np.ndarray
    logits: np.ndarray
    legal_masks: np.ndarray
    episodes: int
    wins: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_actor(checkpoint: dict[str, object], device: torch.device) -> PolicyModel:
    actor = build_actor_critic(
        int(checkpoint["state_dim"]),
        int(checkpoint["action_dim"]),
        int(checkpoint["hidden_dim"]),
        str(checkpoint.get("policy_architecture", "flat")),
    ).to(device)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Policy checkpoint has no state dictionary")
    actor.load_state_dict(state_dict)
    return actor


def collect_teacher_states(
    level_id: str,
    entry: dict[str, object],
    *,
    episodes: int,
    seed: int,
    max_states: int,
    binary: Path | None,
    resources: Path | None,
    device: torch.device,
) -> TeacherStates:
    checkpoint_path = Path(str(entry["checkpoint"]))
    actual_hash = sha256(checkpoint_path)
    if actual_hash != str(entry["sha256"]):
        raise ValueError(f"Checkpoint hash mismatch for {level_id}: {actual_hash}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    raw_config = checkpoint.get("portable_config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"Teacher checkpoint for {level_id} has no portable config")
    config = portable_process_config_from_checkpoint(
        raw_config, state_dim=int(checkpoint["state_dim"])
    )
    config = replace(
        config,
        level=parse_adventure_levels(level_id)[0],
        max_steps=int(entry.get("episode_step_cap", 3600)),
        binary=config.binary if binary is None else binary,
        resources=config.resources if resources is None else resources,
        frame_width=0,
        frame_height=0,
    )
    game_config = GameConfig(**checkpoint["game_config"])
    teacher = load_actor(checkpoint, device).eval()
    states: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    wins = 0
    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        torch.manual_seed(episode_seed)
        environment = PortableEnv(config)
        try:
            observation, _ = environment.reset(seed=episode_seed)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                state = observation.vector(game_config)
                legal_mask = collect_sun_first_mask(
                    environment.codec, environment.legal_action_mask()
                )
                state_tensor = torch.from_numpy(state).to(device).unsqueeze(0)
                mask_tensor = torch.from_numpy(legal_mask).to(device)
                with torch.inference_mode():
                    teacher_logits, _ = teacher(state_tensor)
                    masked = teacher_logits.squeeze(0).masked_fill(
                        ~mask_tensor, -torch.inf
                    )
                    action = int(
                        torch.distributions.Categorical(logits=masked).sample().item()
                    )
                states.append(state)
                logits.append(teacher_logits.squeeze(0).float().cpu().numpy())
                legal_masks.append(legal_mask)
                observation, _, terminated, truncated, _ = environment.step(action)
            wins += int(observation.won)
        finally:
            environment.close()
    if len(states) > max_states:
        selection = np.random.default_rng(seed).choice(
            len(states), size=max_states, replace=False
        )
    else:
        selection = np.arange(len(states))
    return TeacherStates(
        level_id,
        np.stack(states)[selection].astype(np.float32, copy=False),
        np.stack(logits)[selection].astype(np.float16),
        np.stack(legal_masks)[selection].astype(np.bool_, copy=False),
        episodes,
        wins,
    )


def split_level_data(
    data: TeacherStates,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[TeacherStates, TeacherStates]:
    count = len(data.states)
    validation_count = max(1, min(count - 1, round(count * validation_fraction)))
    permutation = rng.permutation(count)
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]

    def subset(indices: np.ndarray) -> TeacherStates:
        return TeacherStates(
            data.level_id,
            data.states[indices],
            data.logits[indices],
            data.legal_masks[indices],
            data.episodes,
            data.wins,
        )

    return subset(training_indices), subset(validation_indices)


@torch.inference_mode()
def evaluate_distillation(
    student: PolicyModel,
    levels: list[TeacherStates],
    device: torch.device,
    temperature: float,
) -> dict[str, object]:
    per_level: dict[str, dict[str, float]] = {}
    for data in levels:
        losses: list[float] = []
        agreements: list[float] = []
        for start in range(0, len(data.states), 512):
            states = torch.from_numpy(data.states[start : start + 512]).to(device)
            teachers = torch.from_numpy(
                data.logits[start : start + 512].astype(np.float32)
            ).to(device)
            masks = torch.from_numpy(data.legal_masks[start : start + 512]).to(device)
            student_logits, _ = student(states)
            losses.append(
                float(
                    masked_policy_distillation_loss(
                        student_logits,
                        teachers,
                        masks,
                        temperature=temperature,
                    ).cpu()
                )
            )
            student_actions = student_logits.masked_fill(~masks, -torch.inf).argmax(1)
            teacher_actions = teachers.masked_fill(~masks, -torch.inf).argmax(1)
            agreements.append(float((student_actions == teacher_actions).float().mean().cpu()))
        per_level[data.level_id] = {
            "states": float(len(data.states)),
            "kl": float(np.mean(losses)),
            "argmax_agreement": float(np.mean(agreements)),
        }
    return {
        "levels": per_level,
        "mean_level_kl": float(np.mean([item["kl"] for item in per_level.values()])),
        "mean_level_argmax_agreement": float(
            np.mean([item["argmax_agreement"] for item in per_level.values()])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill frozen stage specialists into one level-conditioned actor"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/stage_policy_library.json")
    )
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--levels", nargs="*")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-states-per-level", type=int, default=4096)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=260_000)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_states_per_level < 2 or args.updates < 1:
        raise SystemExit("Episode, state, and update counts must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in (0, 1)")
    if args.learning_rate <= 0.0 or args.temperature <= 0.0:
        raise SystemExit("Learning rate and temperature must be positive")

    manifest = json.loads(args.manifest.read_text())
    policies = manifest.get("policies")
    if not isinstance(policies, dict):
        raise SystemExit("Policy-library manifest has no policies mapping")
    level_ids = args.levels or list(policies)
    if args.batch_size < len(level_ids):
        raise SystemExit("--batch-size must be at least the number of levels")
    device_info = select_device(args.device)
    rng = np.random.default_rng(args.seed)
    collected: list[TeacherStates] = []
    for level_index, level_id in enumerate(level_ids):
        entry = policies.get(level_id)
        if not isinstance(entry, dict):
            raise SystemExit(f"No teacher checkpoint for {level_id}")
        data = collect_teacher_states(
            level_id,
            entry,
            episodes=args.episodes,
            seed=args.seed + level_index * 10_000,
            max_states=args.max_states_per_level,
            binary=args.binary,
            resources=args.resources,
            device=device_info.device,
        )
        collected.append(data)
        print(
            json.dumps(
                {
                    "phase": "teacher_collection",
                    "level": level_id,
                    "states": len(data.states),
                    "episodes": data.episodes,
                    "wins": data.wins,
                }
            ),
            flush=True,
        )

    training: list[TeacherStates] = []
    validation: list[TeacherStates] = []
    for data in collected:
        train_data, validation_data = split_level_data(
            data, args.validation_fraction, rng
        )
        training.append(train_data)
        validation.append(validation_data)

    initializer = torch.load(
        args.initialize_from, map_location=device_info.device, weights_only=True
    )
    student = load_actor(initializer, device_info.device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    level_sizes = [len(data.states) for data in training]
    losses: list[float] = []
    for _ in range(args.updates):
        selections = balanced_level_batch_indices(level_sizes, args.batch_size, rng)
        states = torch.from_numpy(
            np.concatenate(
                [data.states[indices] for data, indices in zip(training, selections)]
            )
        ).to(device_info.device)
        teacher_logits = torch.from_numpy(
            np.concatenate(
                [
                    data.logits[indices].astype(np.float32)
                    for data, indices in zip(training, selections)
                ]
            )
        ).to(device_info.device)
        legal_masks = torch.from_numpy(
            np.concatenate(
                [data.legal_masks[indices] for data, indices in zip(training, selections)]
            )
        ).to(device_info.device)
        optimizer.zero_grad(set_to_none=True)
        student_logits, _ = student(states)
        loss = masked_policy_distillation_loss(
            student_logits,
            teacher_logits,
            legal_masks,
            temperature=args.temperature,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    student.eval()
    validation_metrics = evaluate_distillation(
        student, validation, device_info.device, args.temperature
    )
    collection_metrics = {
        data.level_id: {
            "states": len(data.states),
            "episodes": data.episodes,
            "wins": data.wins,
        }
        for data in collected
    }
    output_checkpoint = dict(initializer)
    output_checkpoint.update(
        {
            "agent": "ppo",
            "training_method": "multi_teacher_policy_distillation",
            "initialization": {"checkpoint": str(args.initialize_from)},
            "state_dict": student.state_dict(),
            "portable_action_semantics": "collect_sun_first_v1",
            "teacher_manifest": str(args.manifest),
            "teacher_levels": level_ids,
            "teacher_collection": collection_metrics,
            "distillation_config": {
                "updates": args.updates,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "temperature": args.temperature,
                "validation_fraction": args.validation_fraction,
                "level_balancing": "uniform_per_update",
                "objective": "masked_teacher_kl",
            },
            "distillation_validation": validation_metrics,
            "episode_levels": [],
            "episode_returns": [],
            "episode_wins": [],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    print(
        json.dumps(
            {
                "phase": "complete",
                "output": str(args.output),
                "device": device_info.backend,
                "last_loss": losses[-1],
                **validation_metrics,
            }
        )
    )


if __name__ == "__main__":
    main()

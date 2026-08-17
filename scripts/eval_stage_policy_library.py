from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from pvzwm.agents.ppo import build_actor_critic
from pvzwm.device import select_device
from pvzwm.domain import GameConfig
from pvzwm.portable import (
    PortableEnv,
    collect_sun_first_mask,
    portable_process_config_from_checkpoint,
)
from pvzwm.portable_collection import parse_adventure_levels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen stage-specialist policies without overwriting them"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/stage_policy_library.json")
    )
    parser.add_argument("--levels", nargs="*")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=120_000)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--until-win",
        action="store_true",
        help="Stop each stage after its first authentic terminal win",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Do not expose the next stage when the current stage has no win",
    )
    parser.add_argument(
        "--reseed-policy-per-episode",
        action="store_true",
        help="Seed stochastic policy sampling independently for every attempt",
    )
    parser.add_argument(
        "--require-all-stages-won",
        action="store_true",
        help="Exit nonzero unless every requested stage wins in order",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")

    manifest = json.loads(args.manifest.read_text())
    policies = manifest.get("policies")
    if not isinstance(policies, dict):
        raise SystemExit("Policy-library manifest has no policies mapping")
    levels = args.levels or [level for level, entry in policies.items() if entry]
    device = select_device(args.device)
    results: dict[str, object] = {}
    all_stages_won = True
    completed_through: str | None = None
    for level_index, level_id in enumerate(levels):
        entry = policies.get(level_id)
        if not isinstance(entry, dict):
            results[level_id] = {"status": "no_policy"}
            all_stages_won = False
            continue
        checkpoint_path = Path(str(entry["checkpoint"]))
        expected_hash = str(entry["sha256"])
        actual_hash = _sha256(checkpoint_path)
        if actual_hash != expected_hash:
            raise SystemExit(f"Checkpoint hash mismatch for {level_id}: {actual_hash}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device.device, weights_only=True
        )
        raw_config = checkpoint.get("portable_config")
        if not isinstance(raw_config, dict):
            raise SystemExit(f"Checkpoint for {level_id} has no portable config")
        config = portable_process_config_from_checkpoint(
            raw_config, state_dim=int(checkpoint["state_dim"])
        )
        episode_step_cap = int(entry.get("episode_step_cap", config.max_steps))
        if episode_step_cap < 1:
            raise SystemExit(f"Invalid episode step cap for {level_id}")
        config = replace(
            config,
            level=parse_adventure_levels(level_id)[0],
            max_steps=episode_step_cap,
            binary=config.binary if args.binary is None else args.binary,
            resources=config.resources if args.resources is None else args.resources,
            frame_width=0,
            frame_height=0,
        )
        game_config = GameConfig(**checkpoint["game_config"])
        actor = build_actor_critic(
            int(checkpoint["state_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint["hidden_dim"]),
            str(checkpoint.get("policy_architecture", "flat")),
        ).to(device.device)
        actor.load_state_dict(checkpoint["state_dict"])
        actor.eval()
        wins = 0
        truncations = 0
        winning_seed: int | None = None
        returns: list[float] = []
        lengths: list[int] = []
        torch.manual_seed(args.seed + level_index * 10_000)
        for episode_index in range(args.episodes):
            seed = args.seed + level_index * 10_000 + episode_index
            if args.reseed_policy_per_episode:
                torch.manual_seed(seed)
            environment = PortableEnv(config)
            try:
                observation, _ = environment.reset(seed=seed)
                episode_return = 0.0
                episode_length = 0
                done = False
                while not done:
                    state = torch.from_numpy(observation.vector(game_config)).to(
                        device.device
                    )
                    legal = torch.from_numpy(
                        collect_sun_first_mask(
                            environment.codec, environment.legal_action_mask()
                        )
                    ).to(device.device)
                    with torch.inference_mode():
                        logits, _ = actor(state.unsqueeze(0))
                        masked = logits.squeeze(0).masked_fill(~legal, -torch.inf)
                        if args.stochastic:
                            action = int(
                                torch.distributions.Categorical(logits=masked)
                                .sample()
                                .item()
                            )
                        else:
                            action = int(masked.argmax().item())
                    observation, reward, terminated, truncated, _ = environment.step(
                        action
                    )
                    episode_return += reward
                    episode_length += 1
                    done = terminated or truncated
                wins += int(observation.won)
                truncations += int(truncated and not terminated)
                returns.append(episode_return)
                lengths.append(episode_length)
                if observation.won:
                    if winning_seed is None:
                        winning_seed = seed
                    if args.until_win:
                        break
            finally:
                environment.close()
        all_stages_won &= wins > 0
        if wins > 0:
            completed_through = level_id
        episodes_attempted = len(returns)
        results[level_id] = {
            "checkpoint": str(checkpoint_path),
            "episode_budget": args.episodes,
            "episodes": episodes_attempted,
            "wins": wins,
            "win_rate": wins / episodes_attempted,
            "mean_return": sum(returns) / episodes_attempted,
            "mean_length": sum(lengths) / episodes_attempted,
            "truncation_rate": truncations / episodes_attempted,
            "episode_step_cap": config.max_steps,
            "attempts_until_first_win": len(returns) if wins > 0 else None,
            "winning_seed": winning_seed if wins > 0 else None,
        }
        print(
            json.dumps({"stage_result": level_id, **results[level_id]}),
            file=sys.stderr,
            flush=True,
        )
        if wins == 0 and args.stop_on_failure:
            break
    campaign_complete = all_stages_won and len(results) == len(levels)
    result = {
        "device": device.backend,
        "evaluation_mode": "stochastic" if args.stochastic else "argmax",
        "policy_reseeded_per_episode": args.reseed_policy_per_episode,
        "progression_signal": "ordered authentic terminal wins only",
        "requested_stages": levels,
        "completed_through": completed_through,
        "all_stages_won_at_least_once": campaign_complete,
        "levels": results,
    }
    serialized = json.dumps(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(serialized)
    if args.require_all_stages_won and not campaign_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

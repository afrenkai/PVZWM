from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one shared policy across an authentic Adventure prefix"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=280_000)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.episodes < 1 or args.episode_steps < 1:
        raise SystemExit("Episode counts and step caps must be positive")

    device = select_device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device.device, weights_only=True
    )
    raw_config = checkpoint.get("portable_config")
    if not isinstance(raw_config, dict):
        raise SystemExit("Shared policy checkpoint has no portable config")
    base_config = portable_process_config_from_checkpoint(
        raw_config, state_dim=int(checkpoint["state_dim"])
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

    results: dict[str, dict[str, float]] = {}
    for level_index, level_id in enumerate(args.levels):
        level = parse_adventure_levels(level_id)[0]
        config = replace(
            base_config,
            level=level,
            max_steps=args.episode_steps,
            binary=base_config.binary if args.binary is None else args.binary,
            resources=base_config.resources if args.resources is None else args.resources,
            frame_width=0,
            frame_height=0,
        )
        wins = 0
        returns: list[float] = []
        lengths: list[int] = []
        truncations = 0
        for episode_index in range(args.episodes):
            seed = args.seed + level_index * 10_000 + episode_index
            torch.manual_seed(seed)
            environment = PortableEnv(config)
            try:
                observation, _ = environment.reset(seed=seed)
                episode_return = 0.0
                length = 0
                terminated = False
                truncated = False
                while not (terminated or truncated):
                    state = torch.from_numpy(observation.vector(game_config)).to(
                        device.device
                    )
                    mask = torch.from_numpy(
                        collect_sun_first_mask(
                            environment.codec, environment.legal_action_mask()
                        )
                    ).to(device.device)
                    with torch.inference_mode():
                        logits, _ = actor(state.unsqueeze(0))
                        masked = logits.squeeze(0).masked_fill(~mask, -torch.inf)
                        action = int(
                            torch.distributions.Categorical(logits=masked).sample().item()
                            if args.stochastic
                            else masked.argmax().item()
                        )
                    observation, reward, terminated, truncated, _ = environment.step(
                        action
                    )
                    episode_return += reward
                    length += 1
                wins += int(observation.won)
                truncations += int(truncated and not terminated)
                returns.append(episode_return)
                lengths.append(length)
            finally:
                environment.close()
        results[level_id] = {
            "episodes": float(args.episodes),
            "wins": float(wins),
            "win_rate": wins / args.episodes,
            "mean_return": float(np.mean(returns)),
            "mean_length": float(np.mean(lengths)),
            "truncation_rate": truncations / args.episodes,
        }
        print(json.dumps({"level": level_id, **results[level_id]}), flush=True)
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "levels": results,
                "all_levels_have_a_win": all(item["wins"] > 0 for item in results.values()),
                "minimum_level_win_rate": min(
                    item["win_rate"] for item in results.values()
                ),
                "mean_level_win_rate": float(
                    np.mean([item["win_rate"] for item in results.values()])
                ),
                "action_semantics": "collect_sun_first_v1",
                "device": device.backend,
            }
        )
    )


if __name__ == "__main__":
    main()

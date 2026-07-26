from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pvzwm.data.store import ReplayStore, Transition
from pvzwm.portable import PortableEnv, PortableProcessConfig
from pvzwm.portable_collection import (
    PortableExplorationPolicy,
    encode_portable_observation,
    parse_adventure_levels,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect authentic PvZ engine transitions into DuckDB"
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--levels", default="1-1")
    parser.add_argument("--episodes-per-level", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--ticks-per-step", type=int, default=25)
    parser.add_argument("--observation", choices=("pixels", "vector"), default="pixels")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.episodes_per_level < 1 or args.max_steps < 1:
        raise SystemExit("Episode and step counts must be positive")

    levels = parse_adventure_levels(args.levels)
    config = PortableProcessConfig(
        binary=args.binary,
        resources=args.resources,
        ticks_per_step=args.ticks_per_step,
        frame_width=args.width if args.observation == "pixels" else 0,
        frame_height=args.height if args.observation == "pixels" else 0,
    )
    policy = PortableExplorationPolicy()
    environment = PortableEnv(config)
    try:
        with ReplayStore(args.database) as store:
            episode_index = 0
            for level in levels:
                for _ in range(args.episodes_per_level):
                    seed = args.seed + episode_index
                    episode_index += 1
                    observation, info = environment.reset(level=level, seed=seed)
                    encoded = encode_portable_observation(
                        observation,
                        args.observation,
                        environment.render() if args.observation == "pixels" else None,
                    )
                    rng = np.random.default_rng(seed)
                    with store.episode(
                        f"pvz_portable_{args.observation}",
                        seed,
                        {
                            "engine": "wszqkzqk/PvZ-Portable",
                            "level_number": level,
                            "level_id": info["level"],
                            "logical_resolution": [800, 600],
                            "observation": args.observation,
                            "observation_shape": list(encoded.shape),
                            "vector_version": (
                                observation.vector_version
                                if args.observation == "vector"
                                else None
                            ),
                            "frame_resolution": (
                                [args.width, args.height]
                                if args.observation == "pixels"
                                else None
                            ),
                            "ticks_per_step": args.ticks_per_step,
                            "action_count": environment.codec.size,
                            "policy": "collect_sun_then_uniform_legal_v2",
                            "protocol_version": observation.payload.get(
                                "protocol_version", 2
                            ),
                        },
                    ) as writer:
                        terminated = False
                        truncated = False
                        for step in range(args.max_steps):
                            legal_mask = environment.legal_action_mask()
                            action = policy.act(legal_mask, rng)
                            next_observation, reward, terminated, _, _ = environment.step(
                                action
                            )
                            next_encoded = encode_portable_observation(
                                next_observation,
                                args.observation,
                                environment.render()
                                if args.observation == "pixels"
                                else None,
                            )
                            truncated = not terminated and step + 1 == args.max_steps
                            writer.append(
                                Transition(
                                    observation=encoded,
                                    action=action,
                                    reward=reward,
                                    next_observation=next_encoded,
                                    terminated=terminated,
                                    truncated=truncated,
                                    legal_mask=legal_mask,
                                )
                            )
                            encoded = next_encoded
                            if terminated or truncated:
                                break
                    print(
                        json.dumps(
                            {
                                "episode_id": writer.episode_id,
                                "level": info["level"],
                                "seed": seed,
                                "length": writer.length,
                                "return": writer.total_return,
                                "terminated": terminated,
                                "truncated": truncated,
                                "database_transitions": store.count_transitions(),
                            }
                        )
                    )
    finally:
        environment.close()


if __name__ == "__main__":
    main()

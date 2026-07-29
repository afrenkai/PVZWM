from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pvzwm.portable import PortableActionCodec, PortableEnv, PortableProcessConfig
from pvzwm.portable_collection import PortableExplorationPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace authentic portable state progress")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--ticks-per-step", type=int, default=50)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gdb", action="store_true", help="Run the bridge under gdb")
    args = parser.parse_args()

    environment = PortableEnv(
        PortableProcessConfig(
            binary=args.binary,
            resources=args.resources,
            level=args.level,
            ticks_per_step=args.ticks_per_step,
            max_steps=args.steps,
            frame_width=0,
            frame_height=0,
            debugger=args.gdb,
        )
    )
    policy = PortableExplorationPolicy()
    rng = np.random.default_rng(args.seed)
    codec = PortableActionCodec()
    try:
        observation, _ = environment.reset(seed=args.seed)
        for step in range(args.steps):
            mask = environment.legal_action_mask()
            action = policy.act(mask, rng)
            observation, reward, terminated, truncated, _ = environment.step(action)
            if step == 0 or (step + 1) % args.every == 0 or terminated or truncated:
                decoded = codec.decode(action)
                print(
                    json.dumps(
                        {
                            "step": step + 1,
                            "action": action,
                            "kind": decoded.kind.name,
                            "slot": decoded.slot,
                            "row": decoded.row,
                            "col": decoded.col,
                            "reward": reward,
                            "sun": observation.payload.get("sun"),
                            "wave": observation.payload.get("wave"),
                            "plants": observation.payload.get("plants_count"),
                            "zombies": observation.payload.get("zombies_count"),
                            "coins": observation.payload.get("coins_count"),
                            "legal": int(mask.sum()),
                            "terminated": terminated,
                            "truncated": truncated,
                        }
                    )
                )
            if terminated or truncated:
                break
    finally:
        environment.close()


if __name__ == "__main__":
    main()

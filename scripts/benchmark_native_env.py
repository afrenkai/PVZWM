from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pvzwm.native import NativeBatchEnv, build_native_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the batched native PvZ core")
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--build-directory", type=Path, default=Path("native/build"))
    args = parser.parse_args()
    library = args.library or build_native_library(args.build_directory)
    actions = np.zeros(args.envs, dtype=np.int32)
    with NativeBatchEnv(args.envs, library=library) as env:
        env.reset(0)
        started = time.perf_counter()
        transitions = 0
        for step_index in range(args.steps):
            transition = env.step(actions)
            transitions += args.envs
            if (transition.terminals | transition.truncations).any():
                env.reset(step_index * args.envs)
        elapsed = time.perf_counter() - started
    print(
        f"envs={args.envs} transitions={transitions} seconds={elapsed:.3f} "
        f"steps_per_second={transitions / elapsed:,.0f}"
    )


if __name__ == "__main__":
    main()

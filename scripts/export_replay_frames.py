from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pvzwm.data.store import ReplayStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--decision-ms", type=int, default=250)
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Optional integer nearest-neighbor scale; defaults to native replay pixels",
    )
    args = parser.parse_args()
    if args.frames < 1 or args.scale < 1:
        raise SystemExit("--frames and --scale must be positive")

    with ReplayStore(args.database, read_only=True) as store:
        transitions = list(store.transitions(episode_ids=[args.episode]))
    if not transitions:
        raise SystemExit("Episode is empty or missing")

    args.output.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(transitions) - 1, args.frames, dtype=int)
    cumulative_returns = np.cumsum([item.reward for item in transitions])
    manifest: list[dict[str, int | float | str]] = []
    for frame_index, step in enumerate(indices):
        transition = transitions[int(step)]
        pixels = np.transpose(transition.observation, (1, 2, 0))
        destination = args.output / f"frame-{frame_index:02d}.jpg"
        image = Image.fromarray(pixels)
        if args.scale != 1:
            image = image.resize(
                (image.width * args.scale, image.height * args.scale),
                Image.Resampling.NEAREST,
            )
        image.save(destination, quality=90, optimize=True)
        manifest.append(
            {
                "file": destination.name,
                "step": int(step),
                "seconds": round(int(step) * args.decision_ms / 1_000.0, 2),
                "action": int(transition.action),
                "reward": round(float(transition.reward), 3),
                "return": round(float(cumulative_returns[int(step)]), 3),
            }
        )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

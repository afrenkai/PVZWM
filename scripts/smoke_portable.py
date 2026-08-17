from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from pvzwm.portable import PortableEnv, PortableProcessConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test authentic Adventure resets")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--levels", default="1,11,21,31,41,50")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--no-xvfb",
        action="store_true",
        help="Launch directly (required on macOS; the bridge still uses local pipes)",
    )
    args = parser.parse_args()

    save_directory = tempfile.TemporaryDirectory(prefix="pvzwm-smoke-")
    environment = PortableEnv(
        PortableProcessConfig(
            binary=args.binary,
            resources=args.resources,
            ticks_per_step=1,
            xvfb_run=None if args.no_xvfb else "xvfb-run",
            save_directory=Path(save_directory.name),
        )
    )
    try:
        for level in (int(value) for value in args.levels.split(",")):
            observation, info = environment.reset(level=level, seed=args.seed)
            frame = environment.render()
            print(
                json.dumps(
                    {
                        "level": info["level"],
                        "protocol_version": int(
                            observation.payload.get("protocol_version", 0)
                        ),
                        "vector_size": int(observation.vector().size),
                        "legal_actions": int(environment.legal_action_mask().sum()),
                        "plants": int(observation.payload["plants_count"]),
                        "zombies": int(observation.payload["zombies_count"]),
                        "projectiles": len(observation.payload.get("projectiles", [])),
                        "coins": len(observation.payload.get("coins", [])),
                        "grid_items": len(observation.payload.get("grid_items", [])),
                        "frame_shape": list(frame.shape),
                        "frame_min": int(frame.min()),
                        "frame_max": int(frame.max()),
                        "frame_unique_colors": len(
                            set(
                                map(
                                    tuple,
                                    frame.transpose(1, 2, 0).reshape(-1, 3),
                                )
                            )
                        ),
                    }
                )
            )
    finally:
        environment.close()
        save_directory.cleanup()


if __name__ == "__main__":
    main()

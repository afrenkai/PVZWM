from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np

from pvzwm.data.store import ReplayStore
from pvzwm.portable import PortableEnv, PortableProcessConfig


def _write_frame(
    stream: av.video.stream.VideoStream,
    container: av.container.OutputContainer,
    chw: np.ndarray,
) -> None:
    image = np.ascontiguousarray(np.transpose(chw, (1, 2, 0)))
    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay an exact portable action trajectory and render it as MP4"
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--ticks-per-step", type=int, default=25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max-waits-for-legal",
        type=int,
        default=0,
        help="Insert at most this many waits before a temporarily illegal source action",
    )
    parser.add_argument(
        "--reward-mode", choices=("bridge", "clipped-damage"), default="bridge"
    )
    args = parser.parse_args()
    if args.fps < 1 or args.frame_stride < 1 or args.max_waits_for_legal < 0:
        raise SystemExit("--fps/--frame-stride must be positive and waits nonnegative")

    with ReplayStore(args.database, read_only=True) as store:
        episode = store.connection.execute(
            "SELECT seed, metadata FROM episodes WHERE episode_id = ?",
            [args.episode],
        ).fetchone()
        transitions = list(store.transitions(episode_ids=[args.episode]))
    if episode is None or not transitions:
        raise SystemExit("Episode is empty or missing")
    seed, raw_metadata = episode
    metadata = json.loads(raw_metadata)
    level_id = str(metadata.get("level_id", ""))
    if not level_id.startswith("1-"):
        raise SystemExit(f"Replay requires an Adventure 1-x level id, found {level_id!r}")
    level = int(level_id.split("-", maxsplit=1)[1])

    config = PortableProcessConfig(
        binary=args.binary,
        resources=args.resources,
        level=level,
        ticks_per_step=args.ticks_per_step,
        max_steps=max(1_200, len(transitions) * 3 + 1),
        frame_width=args.width,
        frame_height=args.height,
        vector_version=int(metadata.get("vector_version", 2)),
        reward_mode=args.reward_mode,
    )
    environment = PortableEnv(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    delayed_actions: list[dict[str, int]] = []
    skipped_illegal_steps: list[int] = []
    inserted_waits = 0
    executed_source_actions = 0
    game_steps = 0
    total_return = 0.0
    try:
        observation, _ = environment.reset(seed=int(seed))
        terminated = False
        truncated = False
        with av.open(str(args.output), mode="w") as container:
            stream = container.add_stream("libx264", rate=args.fps)
            stream.width = args.width
            stream.height = args.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}
            _write_frame(stream, container, environment.render())
            frames += 1
            for step, transition in enumerate(transitions):
                waits = 0
                while (
                    not environment.legal_action_mask()[transition.action]
                    and waits < args.max_waits_for_legal
                    and not (observation.won or observation.lost)
                ):
                    observation, reward, terminated, truncated, _ = environment.step(0)
                    total_return += reward
                    waits += 1
                    inserted_waits += 1
                    game_steps += 1
                    if game_steps % args.frame_stride == 0 or terminated or truncated:
                        _write_frame(stream, container, environment.render())
                        frames += 1
                    if terminated or truncated:
                        break
                if waits:
                    delayed_actions.append({"source_step": step, "waits": waits})
                if terminated or truncated:
                    break
                if not environment.legal_action_mask()[transition.action]:
                    skipped_illegal_steps.append(step)
                    continue
                observation, reward, terminated, truncated, _ = environment.step(
                    transition.action
                )
                total_return += reward
                executed_source_actions += 1
                game_steps += 1
                if game_steps % args.frame_stride == 0 or terminated or truncated:
                    _write_frame(stream, container, environment.render())
                    frames += 1
                if terminated or truncated:
                    break
            for packet in stream.encode():
                container.mux(packet)
    finally:
        environment.close()

    print(
        json.dumps(
            {
                "episode": args.episode,
                "level": level_id,
                "seed": seed,
                "source_actions": len(transitions),
                "executed_source_actions": executed_source_actions,
                "inserted_waits": inserted_waits,
                "frames": frames,
                "fps": args.fps,
                "resolution": [args.width, args.height],
                "delayed_actions": delayed_actions,
                "skipped_illegal_steps": skipped_illegal_steps,
                "return": total_return,
                "won": bool(observation.won),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()

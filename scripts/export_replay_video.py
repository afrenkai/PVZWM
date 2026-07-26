from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np

from pvzwm.data.store import ReplayStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a DuckDB replay episode as MP4")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--width", type=int, help="Optional integer-upscaled output width"
    )
    parser.add_argument(
        "--height", type=int, help="Optional integer-upscaled output height"
    )
    args = parser.parse_args()
    if args.fps < 1:
        raise SystemExit("Video rate must be positive")
    if (args.width is None) != (args.height is None):
        raise SystemExit("--width and --height must be supplied together")
    if args.width is not None and (args.width < 1 or args.height < 1):
        raise SystemExit("Video dimensions must be positive")

    with ReplayStore(args.database, read_only=True) as store:
        transitions = list(store.transitions(episode_ids=[args.episode]))
    if not transitions:
        raise SystemExit("Episode is empty or missing")

    source_shape = transitions[0].observation.shape
    if len(source_shape) != 3 or source_shape[0] != 3:
        raise SystemExit("Replay video export requires channel-first RGB observations")
    source_height, source_width = source_shape[1:]
    output_width = args.width if args.width is not None else source_width
    output_height = args.height if args.height is not None else source_height
    if output_width % source_width or output_height % source_height:
        raise SystemExit("Output dimensions must be integer multiples of the replay frame")
    repeat_x = output_width // source_width
    repeat_y = output_height // source_height

    frames = [item.observation for item in transitions]
    frames.append(transitions[-1].next_observation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(args.output), mode="w") as container:
        stream = container.add_stream("libx264", rate=args.fps)
        stream.width = output_width
        stream.height = output_height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for chw in frames:
            if chw.shape != source_shape or chw.dtype != np.uint8:
                raise SystemExit("Replay contains inconsistent RGB frames")
            image = np.transpose(chw, (1, 2, 0))
            image = np.repeat(np.repeat(image, repeat_y, axis=0), repeat_x, axis=1)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    print(
        {
            "episode": args.episode,
            "frames": len(frames),
            "fps": args.fps,
            "resolution": [output_width, output_height],
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()

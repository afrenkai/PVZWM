from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from pvzwm.portable import (
    PortableBridgeClient,
    PortableObservation,
    PortableProcessConfig,
)
from pvzwm.portable_trace import canonical_snapshot, first_snapshot_difference
from pvzwm.renderer_diff import frame_metrics


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def payload_frame(payload: dict[str, Any]) -> np.ndarray:
    frame = PortableObservation(payload).frame()
    if frame is None:
        raise RuntimeError("renderer comparison response did not contain a frame")
    return frame


def compare_response_pair(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[str | None, dict[str, Any], np.ndarray, np.ndarray]:
    state_difference = first_snapshot_difference(
        canonical_snapshot(reference),
        canonical_snapshot(candidate),
    )
    reference_frame = payload_frame(reference)
    candidate_frame = payload_frame(candidate)
    return (
        state_difference,
        frame_metrics(reference_frame, candidate_frame),
        reference_frame,
        candidate_frame,
    )


def write_ppm(path: Path, frame: np.ndarray) -> None:
    channels, height, width = frame.shape
    if channels != 3 or frame.dtype != np.uint8:
        raise ValueError("PPM output requires a channel-first uint8 RGB frame")
    header = f"P6\n{width} {height}\n255\n".encode()
    path.write_bytes(header + frame.transpose(1, 2, 0).tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deterministic GOTY state and pixels between the OpenGL "
            "oracle bridge and the macOS Metal renderer"
        )
    )
    parser.add_argument("--opengl-binary", type=Path, required=True)
    parser.add_argument("--metal-binary", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument(
        "--levels",
        type=parse_int_list,
        default=parse_int_list("1,11,21,31,41,50"),
    )
    parser.add_argument(
        "--checkpoints",
        type=parse_int_list,
        default=parse_int_list("0,500,2000"),
        help="Cumulative engine ticks after reset at which to compare",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Write PPM reference, Metal, and amplified-difference frames",
    )
    parser.add_argument("--max-mean-absolute-error", type=float)
    parser.add_argument("--max-different-pixel-fraction", type=float)
    args = parser.parse_args()

    if tuple(sorted(set(args.checkpoints))) != args.checkpoints:
        raise ValueError("checkpoints must be unique and increasing")
    if args.checkpoints[0] != 0:
        raise ValueError("checkpoints must start at zero")

    reference_save = tempfile.TemporaryDirectory(prefix="pvzwm-renderer-opengl-")
    candidate_save = tempfile.TemporaryDirectory(prefix="pvzwm-renderer-metal-")
    common = {
        "resources": args.resources,
        "xvfb_run": None,
        "frame_width": 0,
        "frame_height": 0,
    }
    reference = PortableBridgeClient(
        PortableProcessConfig(
            binary=args.opengl_binary,
            save_directory=Path(reference_save.name),
            **common,
        )
    )
    candidate = PortableBridgeClient(
        PortableProcessConfig(
            binary=args.metal_binary,
            save_directory=Path(candidate_save.name),
            **common,
        )
    )

    comparisons: list[dict[str, Any]] = []
    try:
        for level in args.levels:
            previous_tick = 0
            for checkpoint in args.checkpoints:
                if checkpoint == 0:
                    command = {
                        "cmd": "reset",
                        "level": level,
                        "seed": args.seed,
                        "frame_width": args.width,
                        "frame_height": args.height,
                    }
                else:
                    command = {
                        "cmd": "step",
                        "action": 0,
                        "ticks": checkpoint - previous_tick,
                        "frame_width": args.width,
                        "frame_height": args.height,
                    }
                reference_payload = reference.request(command)
                candidate_payload = candidate.request(command)
                (
                    state_difference,
                    pixels,
                    reference_frame,
                    candidate_frame,
                ) = compare_response_pair(reference_payload, candidate_payload)
                record = {
                    "level": f"{(level - 1) // 10 + 1}-{(level - 1) % 10 + 1}",
                    "checkpoint": checkpoint,
                    "state_identical": state_difference is None,
                    "state_difference": state_difference,
                    "pixels": pixels,
                }
                comparisons.append(record)
                print(json.dumps(record, allow_nan=False))
                if args.artifact_dir is not None and not pixels["exact"]:
                    args.artifact_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"level-{level:02d}-tick-{checkpoint:05d}"
                    write_ppm(
                        args.artifact_dir / f"{stem}-opengl.ppm",
                        reference_frame,
                    )
                    write_ppm(
                        args.artifact_dir / f"{stem}-metal.ppm",
                        candidate_frame,
                    )
                    amplified_difference = np.clip(
                        np.abs(
                            reference_frame.astype(np.int16)
                            - candidate_frame.astype(np.int16)
                        )
                        * 8,
                        0,
                        255,
                    ).astype(np.uint8)
                    write_ppm(
                        args.artifact_dir / f"{stem}-difference-x8.ppm",
                        amplified_difference,
                    )
                if state_difference is not None:
                    raise RuntimeError(
                        f"gameplay state diverged at level {level}, "
                        f"tick {checkpoint}: {state_difference}"
                    )
                if (
                    args.max_mean_absolute_error is not None
                    and pixels["mean_absolute_error"]
                    > args.max_mean_absolute_error
                ):
                    raise RuntimeError(
                        f"pixel MAE exceeded threshold at level {level}, "
                        f"tick {checkpoint}: {pixels['mean_absolute_error']}"
                    )
                if (
                    args.max_different_pixel_fraction is not None
                    and pixels["different_pixel_fraction"]
                    > args.max_different_pixel_fraction
                ):
                    raise RuntimeError(
                        f"different-pixel fraction exceeded threshold at "
                        f"level {level}, tick {checkpoint}: "
                        f"{pixels['different_pixel_fraction']}"
                    )
                previous_tick = checkpoint
    finally:
        candidate.close()
        reference.close()
        candidate_save.cleanup()
        reference_save.cleanup()

    print(
        json.dumps(
            {
                "comparisons": len(comparisons),
                "state_identical": all(
                    record["state_identical"] for record in comparisons
                ),
                "exact_frames": sum(
                    bool(record["pixels"]["exact"]) for record in comparisons
                ),
                "maximum_mean_absolute_error": max(
                    float(record["pixels"]["mean_absolute_error"])
                    for record in comparisons
                ),
                "maximum_different_pixel_fraction": max(
                    float(record["pixels"]["different_pixel_fraction"])
                    for record in comparisons
                ),
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

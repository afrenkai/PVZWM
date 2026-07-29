from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from pvzwm.portable_collection import parse_adventure_levels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect resized authentic pixels from frozen stage policies"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/stage_policy_library.json")
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--levels", nargs="*")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=230_000)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--uv",
        default=shutil.which("uv") or "uv",
        help="uv executable used for isolated recorder subprocesses",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.width < 1 or args.height < 1:
        raise SystemExit("Episode count and frame dimensions must be positive")

    manifest = json.loads(args.manifest.read_text())
    policies = manifest.get("policies")
    if not isinstance(policies, dict):
        raise SystemExit("Policy-library manifest has no policies mapping")
    levels = args.levels or list(policies)
    for index, stage in enumerate(levels):
        entry = policies.get(stage)
        if not isinstance(entry, dict):
            raise SystemExit(f"No frozen policy for {stage}")
        checkpoint = Path(str(entry["checkpoint"]))
        actual_hash = sha256(checkpoint)
        if actual_hash != str(entry["sha256"]):
            raise SystemExit(f"Checkpoint hash mismatch for {stage}: {actual_hash}")
        command = [
            args.uv,
            "run",
            "scripts/record_portable_agent.py",
            "--checkpoint",
            str(checkpoint),
            "--database",
            str(args.database),
            "--episodes",
            str(args.episodes),
            "--level",
            str(parse_adventure_levels(stage)[0]),
            "--episode-steps",
            str(entry.get("episode_step_cap", 3600)),
            "--observation",
            "pixels",
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--seed",
            str(args.seed + index * 10_000),
            "--stochastic",
            "--device",
            args.device,
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

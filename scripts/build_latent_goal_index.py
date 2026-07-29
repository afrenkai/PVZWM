from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pvzwm.agents.goal_iql import load_goal_episodes
from pvzwm.agents.latent_goals import (
    encode_vector_states,
    fit_cosine_kmeans,
    select_cluster_prototypes,
)
from pvzwm.device import select_device
from pvzwm.evaluation import build_world_model
from pvzwm.models.jepa import ActionConditionedJEPA


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a VISReg-JEPA cosine-cluster goal index from vector replay"
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=373_100)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    sources = args.source or [None]
    episodes = []
    for source in sources:
        episodes.extend(load_goal_episodes(args.database, source=source)[0])
    states = np.concatenate([episode.states for episode in episodes]).astype(
        np.float32, copy=False
    )
    depths = np.concatenate(
        [np.arange(len(episode.states), dtype=np.int64) for episode in episodes]
    )
    device = select_device(args.device)
    raw_checkpoint = torch.load(
        args.jepa_checkpoint, map_location="cpu", weights_only=True
    )
    rebuilt = build_world_model(raw_checkpoint).to(device.device).eval()
    if not isinstance(rebuilt, ActionConditionedJEPA):
        raise SystemExit("Latent goal indexing requires an action-conditioned JEPA")
    latents = encode_vector_states(
        rebuilt, states, device=device.device, batch_size=args.batch_size
    )
    centers, assignments = fit_cosine_kmeans(
        latents,
        args.clusters,
        iterations=args.iterations,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device.device,
    )
    prototypes = select_cluster_prototypes(
        states, depths, latents, centers, assignments
    )
    if len(prototypes.states) != args.clusters:
        raise RuntimeError("Spherical k-means produced empty final clusters")
    payload = {
        "kind": "jepa-cosine-cluster-goals-v1",
        "centers": centers,
        "prototype_states": prototypes.states,
        "prototype_depths": prototypes.depths,
        "cluster_counts": prototypes.counts,
        "prototype_similarities": prototypes.similarities,
        "database": str(args.database),
        "sources": sources,
        "jepa_checkpoint": str(args.jepa_checkpoint),
        "jepa_checkpoint_sha256": file_sha256(args.jepa_checkpoint),
        "clusters": args.clusters,
        "iterations": args.iterations,
        "seed": args.seed,
        "episodes": len(episodes),
        "states": len(states),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary = {
        "output": str(args.output),
        "device": device.backend,
        "episodes": len(episodes),
        "states": len(states),
        "clusters": args.clusters,
        "minimum_cluster_size": int(prototypes.counts.min()),
        "median_cluster_size": float(prototypes.counts.float().median()),
        "maximum_cluster_size": int(prototypes.counts.max()),
        "mean_prototype_cosine_similarity": float(prototypes.similarities.mean()),
        "minimum_prototype_cosine_similarity": float(prototypes.similarities.min()),
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

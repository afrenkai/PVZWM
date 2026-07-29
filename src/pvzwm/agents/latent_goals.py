from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pvzwm.agents.goal_selector import FrontierGoal
from pvzwm.models.jepa import ActionConditionedJEPA


def normalize_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 2:
        raise ValueError("Latent states must be a matrix")
    return torch.nn.functional.normalize(latents.float(), dim=-1, eps=1e-8)


@torch.inference_mode()
def encode_vector_states(
    model: ActionConditionedJEPA,
    states: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    if states.ndim != 2 or len(states) < 1:
        raise ValueError("JEPA goal encoding requires a nonempty state matrix")
    if batch_size < 1:
        raise ValueError("JEPA encoding batch size must be positive")
    encoded: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        encoded.append(model.encode(batch).cpu())
    return normalize_latents(torch.cat(encoded, dim=0))


def nearest_cluster(
    latents: torch.Tensor, centers: torch.Tensor, *, batch_size: int = 4096
) -> torch.Tensor:
    latents = normalize_latents(latents)
    centers = normalize_latents(centers).to(latents.device)
    if batch_size < 1:
        raise ValueError("Cluster assignment batch size must be positive")
    assignments: list[torch.Tensor] = []
    for start in range(0, len(latents), batch_size):
        assignments.append((latents[start : start + batch_size] @ centers.T).argmax(1))
    return torch.cat(assignments).cpu()


def fit_cosine_kmeans(
    latents: torch.Tensor,
    clusters: int,
    *,
    iterations: int = 20,
    seed: int = 0,
    batch_size: int = 4096,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit deterministic spherical k-means without an extra sklearn dependency."""
    if clusters < 2 or clusters > len(latents):
        raise ValueError("Cluster count must be between two and the sample count")
    if iterations < 1:
        raise ValueError("K-means iterations must be positive")
    target = device or latents.device
    normalized = normalize_latents(latents).to(target)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(len(normalized), (), generator=generator))
    selected = torch.zeros(len(normalized), dtype=torch.bool, device=target)
    selected[first] = True
    initial_indices = [first]
    best_similarity = normalized @ normalized[first]
    for _ in range(1, clusters):
        distance = 1.0 - best_similarity
        distance[selected] = -torch.inf
        next_index = int(distance.argmax())
        selected[next_index] = True
        initial_indices.append(next_index)
        best_similarity = torch.maximum(
            best_similarity, normalized @ normalized[next_index]
        )
    centers = normalized[torch.tensor(initial_indices, device=target)].clone()
    for _ in range(iterations):
        assignments = nearest_cluster(
            normalized, centers, batch_size=batch_size
        ).to(target)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, normalized)
        counts = torch.bincount(assignments, minlength=clusters)
        occupied = counts > 0
        updated = centers.clone()
        updated[occupied] = sums[occupied] / counts[occupied, None]
        centers = normalize_latents(updated)
    assignments = nearest_cluster(normalized, centers, batch_size=batch_size)
    return centers.cpu(), assignments


@dataclass(frozen=True, slots=True)
class LatentGoalPrototypes:
    states: torch.Tensor
    depths: torch.Tensor
    counts: torch.Tensor
    similarities: torch.Tensor


def select_cluster_prototypes(
    states: np.ndarray,
    depths: np.ndarray,
    latents: torch.Tensor,
    centers: torch.Tensor,
    assignments: torch.Tensor,
) -> LatentGoalPrototypes:
    if states.ndim != 2 or depths.shape != (len(states),):
        raise ValueError("Prototype states and depths are misaligned")
    if latents.shape[0] != len(states) or assignments.shape != (len(states),):
        raise ValueError("Prototype latent assignments are misaligned")
    normalized = normalize_latents(latents).cpu()
    centers = normalize_latents(centers).cpu()
    assignments = assignments.long().cpu()
    clusters = len(centers)
    similarities = (normalized * centers[assignments]).sum(-1)
    best_indices = torch.full((clusters,), -1, dtype=torch.long)
    best_similarities = torch.full((clusters,), -torch.inf)
    counts = torch.bincount(assignments, minlength=clusters)
    for index, cluster in enumerate(assignments.tolist()):
        if similarities[index] > best_similarities[cluster]:
            best_similarities[cluster] = similarities[index]
            best_indices[cluster] = index
    occupied = best_indices >= 0
    selected_indices = best_indices[occupied]
    return LatentGoalPrototypes(
        torch.from_numpy(states[selected_indices.numpy()].copy()),
        torch.from_numpy(depths[selected_indices.numpy()].copy()),
        counts[occupied],
        best_similarities[occupied],
    )


def cluster_key(cluster: int) -> bytes:
    if cluster < 0:
        raise ValueError("Cluster ids cannot be negative")
    return int(cluster).to_bytes(4, "little", signed=False)


class JEPALatentGoalEncoder:
    def __init__(
        self,
        model: ActionConditionedJEPA,
        centers: torch.Tensor,
        prototypes: torch.Tensor,
        device: torch.device,
    ) -> None:
        if prototypes.ndim != 2 or len(prototypes) != len(centers):
            raise ValueError("Latent centers and prototypes must align")
        self.model = model
        self.centers = normalize_latents(centers).to(device)
        self.prototypes = prototypes.cpu()
        self.device = device

    @torch.inference_mode()
    def cluster(self, state: np.ndarray) -> int:
        tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
        latent = normalize_latents(self.model.encode(tensor))
        return int((latent @ self.centers.T).argmax(1).item())

    def encode(self, state: np.ndarray) -> bytes:
        return cluster_key(self.cluster(state))

    def goal_state(self, state: np.ndarray) -> np.ndarray:
        return self.prototypes[self.cluster(state)].numpy().astype(
            np.float32, copy=False
        )


def frontier_goals_from_checkpoint(
    checkpoint: dict[str, object],
) -> tuple[list[FrontierGoal], torch.Tensor]:
    centers = checkpoint.get("centers")
    states = checkpoint.get("prototype_states")
    depths = checkpoint.get("prototype_depths")
    if not isinstance(centers, torch.Tensor):
        raise ValueError("Latent goal index has no centers")
    if not isinstance(states, torch.Tensor) or not isinstance(depths, torch.Tensor):
        raise ValueError("Latent goal index has no prototypes")
    if states.ndim != 2 or depths.shape != (len(states),):
        raise ValueError("Latent goal prototypes are malformed")
    if len(states) != len(centers):
        raise ValueError("Every latent center requires one prototype")
    goals = [
        FrontierGoal(
            states[index].numpy().astype(np.float32, copy=False),
            cluster_key(index),
            int(depths[index]),
        )
        for index in range(len(states))
    ]
    return goals, normalize_latents(centers)

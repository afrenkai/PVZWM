import numpy as np
import torch

from pvzwm.agents.latent_goals import (
    cluster_key,
    fit_cosine_kmeans,
    frontier_goals_from_checkpoint,
    nearest_cluster,
    select_cluster_prototypes,
)


def test_cosine_kmeans_separates_obvious_directions() -> None:
    latents = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]
    )
    centers, assignments = fit_cosine_kmeans(
        latents, 2, iterations=5, seed=3, device=torch.device("cpu")
    )
    assert assignments[0] == assignments[1]
    assert assignments[2] == assignments[3]
    assert assignments[0] != assignments[2]
    assert torch.equal(nearest_cluster(latents, centers), assignments)


def test_cluster_prototypes_keep_the_medoid_state_depth() -> None:
    states = np.arange(20, dtype=np.float32).reshape(4, 5)
    depths = np.asarray([1, 7, 2, 9], dtype=np.int64)
    latents = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]
    )
    centers = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    assignments = torch.tensor([0, 0, 1, 1])
    prototypes = select_cluster_prototypes(
        states, depths, latents, centers, assignments
    )
    assert prototypes.states.shape == (2, 5)
    assert prototypes.depths.tolist() == [1, 2]
    assert prototypes.counts.tolist() == [2, 2]
    assert any(np.array_equal(prototypes.states[0].numpy(), row) for row in states)


def test_frontier_goal_checkpoint_round_trip() -> None:
    goals, centers = frontier_goals_from_checkpoint(
        {
            "centers": torch.eye(2),
            "prototype_states": torch.arange(10, dtype=torch.float32).reshape(2, 5),
            "prototype_depths": torch.tensor([4, 8]),
        }
    )
    assert [goal.depth for goal in goals] == [4, 8]
    assert goals[1].key == cluster_key(1)
    assert centers.shape == (2, 2)

import numpy as np
import torch

from pvzwm.domain import ActionKind, GameConfig, PVZAction
from pvzwm.env import PVZEnv
from pvzwm.perception import (
    BoardPerceptionModel,
    ObservationVectorLayout,
    SyntheticPerceptionGenerator,
    legal_mask_from_vector,
    legal_masks_from_vectors,
)


def test_perception_shapes_and_vector_legal_mask() -> None:
    config = GameConfig(max_time=5.0, spawn_interval=100.0)
    generator = SyntheticPerceptionGenerator(config, seed=0, cell_size=8)
    frames, states = generator.batch(3)
    layout = ObservationVectorLayout(config)
    assert frames.shape == (3, 3, config.rows * 8, config.cols * 8)
    assert states.shape == (3, layout.size)
    model = BoardPerceptionModel(layout.size, latent_dim=16)
    loss = model.loss(frames, states)
    loss.backward()
    assert torch.isfinite(loss)

    env = PVZEnv(config, seed=0)
    observation, _ = env.reset()
    predicted_mask = legal_mask_from_vector(observation.vector(config), config)
    np.testing.assert_array_equal(predicted_mask, env.legal_action_mask())
    observation, *_ = env.step(PVZAction(ActionKind.PLANT_SUNFLOWER, 0, 0))
    predicted_mask = legal_mask_from_vector(observation.vector(config), config)
    np.testing.assert_array_equal(predicted_mask, env.legal_action_mask())
    batch = np.stack([observation.vector(config), observation.vector(config)])
    predicted_batch = legal_masks_from_vectors(batch, config)
    assert predicted_batch.shape == (2, env.codec.size)
    np.testing.assert_array_equal(predicted_batch[0], env.legal_action_mask())

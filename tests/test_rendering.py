import numpy as np

from pvzwm.domain import ActionKind, GameConfig, PVZAction
from pvzwm.env import PVZEnv
from pvzwm.rendering import SymbolicRenderer


def test_symbolic_renderer_produces_chw_float_pixels() -> None:
    config = GameConfig(max_time=10.0, spawn_interval=100.0)
    env = PVZEnv(config, seed=0)
    observation, _ = env.reset()
    renderer = SymbolicRenderer(config, cell_size=8)
    before = renderer.render(observation)
    assert before.shape == (3, config.rows * 8, config.cols * 8)
    assert before.dtype == np.float32
    assert 0.0 <= before.min() <= before.max() <= 1.0

    observation, *_ = env.step(PVZAction(ActionKind.PLANT_SUNFLOWER, 0, 0))
    after = renderer.render(observation)
    assert not np.array_equal(before, after)

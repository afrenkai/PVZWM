from __future__ import annotations

import numpy as np

from pvzwm.domain import GameConfig, Observation, PlantKind


class SymbolicRenderer:
    """Clean-room rasterizer for exercising pixel world models without game assets."""

    plant_colors = {
        PlantKind.SUNFLOWER: np.asarray([0.96, 0.78, 0.12], dtype=np.float32),
        PlantKind.PEASHOOTER: np.asarray([0.18, 0.72, 0.22], dtype=np.float32),
        PlantKind.WALLNUT: np.asarray([0.55, 0.31, 0.13], dtype=np.float32),
    }

    def __init__(self, config: GameConfig, cell_size: int = 12) -> None:
        if cell_size < 4:
            raise ValueError("cell_size must be at least 4")
        self.config = config
        self.cell_size = cell_size
        self.height = config.rows * cell_size
        self.width = config.cols * cell_size

    def render(self, observation: Observation) -> np.ndarray:
        """Return channel-first float32 RGB in [0, 1]."""
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        lawn_a = np.asarray([0.30, 0.60, 0.20], dtype=np.float32)
        lawn_b = np.asarray([0.35, 0.66, 0.24], dtype=np.float32)
        for row in range(self.config.rows):
            for col in range(self.config.cols):
                top = row * self.cell_size
                left = col * self.cell_size
                image[top : top + self.cell_size, left : left + self.cell_size] = (
                    lawn_a if (row + col) % 2 == 0 else lawn_b
                )
                plant = PlantKind(int(observation.plants[row, col]))
                if plant != PlantKind.EMPTY:
                    margin = max(self.cell_size // 4, 1)
                    color = self.plant_colors[plant]
                    hp = float(
                        np.clip(observation.plant_hp[row, col] / 4000.0, 0.15, 1.0)
                    )
                    image[
                        top + margin : top + self.cell_size - margin,
                        left + margin : left + self.cell_size - margin,
                    ] = color * (0.55 + 0.45 * hp)
                zombie_count = int(observation.zombies[row, col])
                if zombie_count:
                    width = min(self.cell_size - 2, 2 + 2 * zombie_count)
                    image[top + 1 : top + self.cell_size - 1, left + 1 : left + 1 + width] = (
                        0.48,
                        0.42,
                        0.46,
                    )
                    hp_width = int(
                        (self.cell_size - 2)
                        * np.clip(observation.zombie_hp[row, col] / 500.0, 0.0, 1.0)
                    )
                    image[top : top + 1, left + 1 : left + 1 + hp_width] = (0.8, 0.05, 0.05)

        # Global resource/time bars provide the information present in vector state.
        sun_width = int(self.width * np.clip(observation.sun / 1000.0, 0.0, 1.0))
        time_width = int(
            self.width * np.clip(observation.time / self.config.max_time, 0.0, 1.0)
        )
        image[0:2, :sun_width] = (1.0, 0.86, 0.1)
        image[-2:, :time_width] = (0.15, 0.3, 0.95)
        for row, available in enumerate(observation.mowers):
            if available:
                top = row * self.cell_size + self.cell_size // 2
                image[top : top + 2, :3] = (0.9, 0.1, 0.1)
        return np.transpose(image, (2, 0, 1)).copy()

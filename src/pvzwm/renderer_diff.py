from __future__ import annotations

import math
from typing import Any

import numpy as np


def frame_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Return deterministic RGB error metrics for two channel-first frames."""

    if reference.shape != candidate.shape:
        raise ValueError(
            f"frame shapes differ: {reference.shape!r} != {candidate.shape!r}"
        )
    delta = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
    squared = delta.astype(np.float64) ** 2
    mean_squared_error = float(squared.mean())
    different_pixels = np.any(delta != 0, axis=0)
    y_indices, x_indices = np.nonzero(different_pixels)
    if x_indices.size:
        difference_bounds: dict[str, int] | None = {
            "x": int(x_indices.min()),
            "y": int(y_indices.min()),
            "width": int(x_indices.max() - x_indices.min() + 1),
            "height": int(y_indices.max() - y_indices.min() + 1),
        }
        sample_differences = [
            {
                "x": int(x),
                "y": int(y),
                "reference": reference[:, y, x].tolist(),
                "candidate": candidate[:, y, x].tolist(),
            }
            for y, x in zip(y_indices[:8], x_indices[:8], strict=True)
        ]
    else:
        difference_bounds = None
        sample_differences = []
    return {
        "exact": bool(np.array_equal(reference, candidate)),
        "different_pixel_count": int(different_pixels.sum()),
        "different_pixel_fraction": float(different_pixels.mean()),
        "difference_bounds": difference_bounds,
        "sample_differences": sample_differences,
        "mean_absolute_error": float(delta.mean()),
        "root_mean_squared_error": math.sqrt(mean_squared_error),
        "peak_signal_to_noise_db": (
            None
            if mean_squared_error == 0.0
            else 20.0 * math.log10(255.0 / math.sqrt(mean_squared_error))
        ),
        "maximum_channel_error": int(delta.max()),
        "within_one_channel_fraction": float((delta <= 1).mean()),
    }

from __future__ import annotations

import numpy as np
import pytest

from pvzwm.renderer_diff import frame_metrics


def test_frame_metrics_exact() -> None:
    frame = np.arange(18, dtype=np.uint8).reshape(3, 2, 3)

    metrics = frame_metrics(frame, frame.copy())

    assert metrics == {
        "exact": True,
        "different_pixel_count": 0,
        "different_pixel_fraction": 0.0,
        "difference_bounds": None,
        "sample_differences": [],
        "mean_absolute_error": 0.0,
        "root_mean_squared_error": 0.0,
        "peak_signal_to_noise_db": None,
        "maximum_channel_error": 0,
        "within_one_channel_fraction": 1.0,
    }


def test_frame_metrics_reports_channel_and_pixel_error() -> None:
    reference = np.zeros((3, 2, 2), dtype=np.uint8)
    candidate = reference.copy()
    candidate[0, 0, 1] = 2

    metrics = frame_metrics(reference, candidate)

    assert metrics["exact"] is False
    assert metrics["different_pixel_count"] == 1
    assert metrics["different_pixel_fraction"] == 0.25
    assert metrics["difference_bounds"] == {"x": 1, "y": 0, "width": 1, "height": 1}
    assert metrics["sample_differences"] == [
        {"x": 1, "y": 0, "reference": [0, 0, 0], "candidate": [2, 0, 0]}
    ]
    assert metrics["mean_absolute_error"] == pytest.approx(1.0 / 6.0)
    assert metrics["root_mean_squared_error"] == pytest.approx((1.0 / 3.0) ** 0.5)
    assert metrics["maximum_channel_error"] == 2
    assert metrics["within_one_channel_fraction"] == pytest.approx(11.0 / 12.0)

from argparse import Namespace
import json
from pathlib import Path

import torch

from pvzwm.cli import _checkpoint_arguments, build_parser


def test_checkpoint_arguments_convert_paths_recursively(tmp_path: Path) -> None:
    arguments = _checkpoint_arguments(
        Namespace(
            database=tmp_path / "replay.duckdb",
            nested={"checkpoints": [tmp_path / "encoder.pt"]},
            levels=(1, 2),
            function=lambda _: None,
        )
    )

    assert arguments == {
        "database": str(tmp_path / "replay.duckdb"),
        "nested": {"checkpoints": [str(tmp_path / "encoder.pt")]},
        "levels": [1, 2],
    }
    json.dumps(arguments)
    checkpoint = tmp_path / "safe.pt"
    torch.save({"args": arguments}, checkpoint)
    assert torch.load(checkpoint, weights_only=True)["args"] == arguments


def test_world_model_training_defaults_to_bounded_cpu_threads() -> None:
    arguments = build_parser().parse_args(["train-wm"])

    assert arguments.cpu_threads == 8


def test_world_model_training_accepts_explicit_cpu_thread_limit() -> None:
    arguments = build_parser().parse_args(["train-wm", "--cpu-threads", "3"])

    assert arguments.cpu_threads == 3


def test_agent_training_accepts_adaptive_intrinsic_weighting() -> None:
    arguments = build_parser().parse_args(
        [
            "train-agent",
            "--agent",
            "ppo",
            "--adaptive-intrinsic-weight",
            "--intrinsic-lagrange-rate",
            "2.5",
            "--intrinsic-performance-smoothing",
            "0.2",
        ]
    )

    assert arguments.adaptive_intrinsic_weight is True
    assert arguments.intrinsic_lagrange_rate == 2.5
    assert arguments.intrinsic_performance_smoothing == 0.2


def test_agent_training_accepts_vdn_autoregressive_credit_assignment() -> None:
    arguments = build_parser().parse_args(
        [
            "train-agent",
            "--agent",
            "ppo",
            "--policy-architecture",
            "portable-vdn-autoregressive",
            "--vdn-importance-anneal-fraction",
            "0.5",
        ]
    )

    assert arguments.policy_architecture == "portable-vdn-autoregressive"
    assert arguments.vdn_importance_anneal_fraction == 0.5

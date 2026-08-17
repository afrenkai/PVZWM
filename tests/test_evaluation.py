from pathlib import Path

import numpy as np
import torch

from pvzwm.data.store import ReplayStore, Transition
from pvzwm.evaluation import build_world_model, evaluate_world_model
from pvzwm.models.delta import PixelDeltaWorldModel
from pvzwm.models.dynamics import DeterministicDynamics, DynamicsEnsemble
from pvzwm.models.flow import RepresentationFlowWorldModel
from pvzwm.models.jepa import ActionConditionedJEPA, ConvEncoder, VectorEncoder
from pvzwm.models.lewm import ActionConditionedLeWorldModel
from pvzwm.models.token import CausalTokenWorldModel


def test_open_loop_evaluation_and_checkpoint_rebuild(tmp_path: Path) -> None:
    database = tmp_path / "evaluation.duckdb"
    state_dim = 6
    action_dim = 4
    with ReplayStore(database) as store:
        with store.episode("python_port", 0) as episode:
            for step in range(12):
                observation = np.full(state_dim, step / 12, dtype=np.float32)
                next_observation = np.full(state_dim, (step + 1) / 12, dtype=np.float32)
                episode.append(
                    Transition(
                        observation,
                        step % action_dim,
                        float(step == 11),
                        next_observation,
                        step == 11,
                        legal_mask=np.ones(action_dim, dtype=np.bool_),
                    )
                )
        episode_ids = store.episode_ids("python_port")

    model = DeterministicDynamics(state_dim, action_dim, hidden_dim=16)
    metrics = evaluate_world_model(
        model,
        "mlp",
        database,
        episode_ids,
        torch.device("cpu"),
        horizon=3,
        max_sequences=4,
        batch_size=2,
    )
    assert metrics["evaluated_sequences"] == 4
    assert metrics["open_loop_state_mse_h1"] >= 0
    assert metrics["open_loop_state_mse_h3"] >= 0
    assert metrics["open_loop_state_persistence_mse_mean"] > 0
    assert np.isfinite(metrics["open_loop_state_skill_vs_persistence"])
    assert metrics["open_loop_reward_mae_mean"] >= 0

    checkpoint = {
        "model": "mlp",
        "state_dict": model.state_dict(),
        "state_dim": state_dim,
        "observation_shape": (state_dim,),
        "action_dim": action_dim,
        "args": {"hidden_dim": 16},
    }
    rebuilt = build_world_model(checkpoint)
    for expected, actual in zip(model.parameters(), rebuilt.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)

    ensemble = DynamicsEnsemble(3, state_dim, action_dim, hidden_dim=16)
    ensemble_metrics = evaluate_world_model(
        ensemble,
        "ensemble",
        database,
        episode_ids,
        torch.device("cpu"),
        horizon=3,
        max_sequences=4,
        batch_size=2,
    )
    assert ensemble_metrics["open_loop_epistemic_variance_mean"] >= 0
    rebuilt_ensemble = build_world_model(
        {
            "model": "ensemble",
            "state_dict": ensemble.state_dict(),
            "state_dim": state_dim,
            "observation_shape": (state_dim,),
            "action_dim": action_dim,
            "args": {"hidden_dim": 16, "ensemble_size": 3},
        }
    )
    assert isinstance(rebuilt_ensemble, DynamicsEnsemble)

    jepa = ActionConditionedJEPA(
        VectorEncoder(state_dim, latent_dim=5, hidden_dim=12),
        action_dim,
        latent_dim=5,
        hidden_dim=12,
    )
    jepa_metrics = evaluate_world_model(
        jepa,
        "jepa",
        database,
        episode_ids,
        torch.device("cpu"),
        horizon=3,
        max_sequences=4,
        batch_size=2,
    )
    assert jepa_metrics["representation_feature_std_mean"] >= 0
    assert 1 <= jepa_metrics["representation_effective_rank"] <= 5
    assert 0 <= jepa_metrics["open_loop_latent_cosine_distance_mean"] <= 2
    assert np.isfinite(jepa_metrics["open_loop_latent_skill_vs_persistence"])
    assert np.isfinite(
        jepa_metrics["open_loop_action_counterfactual_margin_h1"]
    )
    assert 0 <= jepa_metrics["open_loop_action_counterfactual_preference_h1"] <= 1


def test_lewm_rebuild_preserves_projection_head_compatibility() -> None:
    observation_shape = (3, 16, 24)
    common_args = {
        "latent_dim": 8,
        "hidden_dim": 16,
        "conv_pool_height": 1,
        "conv_pool_width": 1,
        "transformer_layers": 1,
        "transformer_heads": 2,
        "transformer_dropout": 0.0,
        "sequence_length": 3,
    }
    for projection_head, prediction_projection_head in (
        (False, False),
        (True, False),
        (True, True),
    ):
        model = ActionConditionedLeWorldModel(
            ConvEncoder(3, latent_dim=8, pool_height=1, pool_width=1),
            action_dim=4,
            latent_dim=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_history=3,
            projection_head=projection_head,
            prediction_projection_head=prediction_projection_head,
            projection_hidden_dim=12,
        )
        args = dict(common_args)
        if projection_head:
            args["lewm_projection_head"] = True
            args["lewm_projection_hidden_dim"] = 12
        if prediction_projection_head:
            args["lewm_prediction_projection_head"] = True
        rebuilt = build_world_model(
            {
                "model": "lewm",
                "state_dict": model.state_dict(),
                "state_dim": int(np.prod(observation_shape)),
                "observation_shape": observation_shape,
                "action_dim": 4,
                "args": args,
            }
        )
        assert isinstance(rebuilt, ActionConditionedLeWorldModel)
        assert (rebuilt.projector is not None) is projection_head
        assert (
            rebuilt.prediction_projector is not None
        ) is prediction_projection_head

    residual = ActionConditionedLeWorldModel(
        ConvEncoder(3, latent_dim=8, pool_height=1, pool_width=1),
        action_dim=4,
        latent_dim=8,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        max_history=3,
        residual_prediction=True,
    )
    residual_args = dict(common_args)
    residual_args["lewm_residual_prediction"] = True
    rebuilt_residual = build_world_model(
        {
            "model": "lewm",
            "state_dict": residual.state_dict(),
            "state_dim": int(np.prod(observation_shape)),
            "observation_shape": observation_shape,
            "action_dim": 4,
            "args": residual_args,
        }
    )
    assert isinstance(rebuilt_residual, ActionConditionedLeWorldModel)
    assert rebuilt_residual.residual_prediction

    legacy_projector = ActionConditionedLeWorldModel(
        ConvEncoder(3, latent_dim=8, pool_height=1, pool_width=1),
        action_dim=4,
        latent_dim=8,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        max_history=3,
        projection_head=True,
        projection_depth=3,
    )
    legacy_args = dict(common_args)
    legacy_args["lewm_projection_head"] = True
    rebuilt_legacy = build_world_model(
        {
            "model": "lewm",
            "state_dict": legacy_projector.state_dict(),
            "state_dim": int(np.prod(observation_shape)),
            "observation_shape": observation_shape,
            "action_dim": 4,
            "args": legacy_args,
        }
    )
    assert isinstance(rebuilt_legacy, ActionConditionedLeWorldModel)
    assert rebuilt_legacy.projector is not None


def test_token_world_model_open_loop_evaluation_and_rebuild(tmp_path: Path) -> None:
    database = tmp_path / "token_evaluation.duckdb"
    observation_shape = (3, 16, 24)
    action_dim = 4
    with ReplayStore(database) as store:
        with store.episode("pixels", 0) as episode:
            for step in range(6):
                observation = np.full(observation_shape, step * 10, dtype=np.uint8)
                next_observation = np.full(observation_shape, (step + 1) * 10, dtype=np.uint8)
                episode.append(
                    Transition(
                        observation,
                        step % action_dim,
                        float(step == 5),
                            next_observation,
                            step == 5,
                            legal_mask=np.ones(action_dim, dtype=np.bool_),
                            next_legal_mask=np.ones(action_dim, dtype=np.bool_),
                        )
                )
        episode_ids = store.episode_ids("pixels")

    model = CausalTokenWorldModel(
        observation_shape,
        action_dim,
        vocab_size=8,
        code_dim=4,
        token_height=2,
        token_width=2,
        tokenizer_hidden_dim=16,
        model_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_transitions=3,
    )
    metrics = evaluate_world_model(
        model,
        "token",
        database,
        episode_ids,
        torch.device("cpu"),
        horizon=2,
        max_sequences=2,
        batch_size=1,
    )
    assert metrics["open_loop_pixel_mse_mean"] >= 0
    assert metrics["open_loop_pixel_persistence_mse_mean"] > 0
    assert 0 <= metrics["token_codebook_utilization"] <= 1
    checkpoint = {
        "model": "token",
        "state_dict": model.state_dict(),
        "state_dim": int(np.prod(observation_shape)),
        "observation_shape": observation_shape,
        "action_dim": action_dim,
        "args": {
            "token_vocab_size": 8,
            "token_code_dim": 4,
            "token_height": 2,
            "token_width": 2,
            "tokenizer_hidden_dim": 16,
            "token_model_dim": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "transformer_dropout": 0.0,
            "sequence_length": 3,
        },
    }
    rebuilt = build_world_model(checkpoint)
    assert isinstance(rebuilt, CausalTokenWorldModel)

    flow = RepresentationFlowWorldModel(
        observation_shape,
        action_dim,
        code_dim=4,
        latent_height=2,
        latent_width=2,
        autoencoder_hidden_dim=16,
        model_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    flow_metrics = evaluate_world_model(
        flow,
        "flow",
        database,
        episode_ids,
        torch.device("cpu"),
        horizon=2,
        max_sequences=2,
        batch_size=1,
    )
    assert flow_metrics["open_loop_pixel_mse_mean"] >= 0
    assert "open_loop_action_counterfactual_margin_h1" in flow_metrics
    assert 0.0 <= flow_metrics["open_loop_action_counterfactual_preference_h1"] <= 1.0
    rebuilt_flow = build_world_model(
        {
            "model": "flow",
            "state_dict": flow.state_dict(),
            "state_dim": int(np.prod(observation_shape)),
            "observation_shape": observation_shape,
            "action_dim": action_dim,
            "args": {
                "flow_code_dim": 4,
                "flow_latent_height": 2,
                "flow_latent_width": 2,
                "flow_autoencoder_hidden_dim": 16,
                "flow_model_dim": 16,
                "flow_layers": 1,
                "flow_heads": 4,
                "flow_dropout": 0.0,
            },
        }
    )
    assert isinstance(rebuilt_flow, RepresentationFlowWorldModel)

    delta = PixelDeltaWorldModel(
        observation_shape,
        action_dim,
        hidden_channels=16,
        residual_blocks=1,
    )
    delta_metrics = evaluate_world_model(
        delta,
        "delta",
        database,
        episode_ids,
        torch.device("cpu"),
        horizon=2,
        max_sequences=2,
        batch_size=1,
    )
    assert delta_metrics["open_loop_pixel_mse_mean"] >= 0
    torch.testing.assert_close(
        torch.tensor(delta_metrics["open_loop_pixel_mse_mean"]),
        torch.tensor(delta_metrics["open_loop_pixel_persistence_mse_mean"]),
    )
    rebuilt_delta = build_world_model(
        {
            "model": "delta",
            "state_dict": delta.state_dict(),
            "state_dim": int(np.prod(observation_shape)),
            "observation_shape": observation_shape,
            "action_dim": action_dim,
            "args": {
                "delta_hidden_channels": 16,
                "delta_residual_blocks": 1,
            },
        }
    )
    assert isinstance(rebuilt_delta, PixelDeltaWorldModel)

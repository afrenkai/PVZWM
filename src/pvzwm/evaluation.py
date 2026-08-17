from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pvzwm.data.store import ReplayStore, Transition
from pvzwm.models.delta import PixelDeltaWorldModel
from pvzwm.models.dynamics import DeterministicDynamics, DynamicsEnsemble
from pvzwm.models.flow import RepresentationFlowWorldModel
from pvzwm.models.jepa import ActionConditionedJEPA, ConvEncoder, VectorEncoder
from pvzwm.models.lewm import ActionConditionedLeWorldModel
from pvzwm.models.rssm import RecurrentStateSpaceModel
from pvzwm.models.token import CausalTokenWorldModel

WorldModel = (
    DeterministicDynamics
    | DynamicsEnsemble
    | ActionConditionedJEPA
    | ActionConditionedLeWorldModel
    | RecurrentStateSpaceModel
    | CausalTokenWorldModel
    | RepresentationFlowWorldModel
    | PixelDeltaWorldModel
)


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    if values.dtype == np.uint8:
        tensor = tensor.div(255.0)
    return tensor


def _mean_over_features(value: torch.Tensor) -> torch.Tensor:
    return value.flatten(1).mean(dim=1)


def _accumulate(
    totals: dict[str, float], counts: dict[str, int], name: str, values: torch.Tensor
) -> None:
    totals[name] = totals.get(name, 0.0) + float(values.sum().detach().cpu())
    counts[name] = counts.get(name, 0) + int(values.numel())


def _accumulate_legality(
    totals: dict[str, float],
    counts: dict[str, int],
    logits: torch.Tensor,
    target: torch.Tensor,
    step: int,
) -> None:
    predicted = logits.sigmoid() >= 0.5
    true_positive = (predicted & target).sum(dim=1).float()
    values = {
        "bce": F.binary_cross_entropy_with_logits(
            logits, target.to(logits.dtype), reduction="none"
        ).mean(dim=1),
        "accuracy": (predicted == target).float().mean(dim=1),
        "recall": true_positive / target.sum(dim=1).clamp_min(1),
        "precision": true_positive / predicted.sum(dim=1).clamp_min(1),
        "predicted_fraction": predicted.float().mean(dim=1),
        "true_fraction": target.float().mean(dim=1),
    }
    for name, value in values.items():
        _accumulate(totals, counts, f"open_loop_legal_{name}_h{step}", value)


def _alternative_legal_actions(
    actions: torch.Tensor, legal_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    alternatives = legal_mask.clone()
    alternatives.scatter_(1, actions[:, None], False)
    valid = alternatives.any(dim=1)
    selected = alternatives.to(torch.int64).argmax(dim=1)
    selected = torch.where(valid, selected, actions)
    return selected, valid


@torch.inference_mode()
def evaluate_world_model(
    model: WorldModel,
    model_kind: str,
    database: str | Path,
    episode_ids: list[str] | None,
    device: torch.device,
    *,
    horizon: int = 8,
    max_sequences: int = 128,
    batch_size: int = 16,
) -> dict[str, float | bool]:
    """Evaluate open-loop predictions on episode-safe trajectory windows."""
    if horizon < 1 or max_sequences < 1 or batch_size < 1:
        raise ValueError("horizon, max_sequences, and batch_size must be positive")
    with ReplayStore(database, read_only=True) as store:
        sequences: list[list[Transition]] = []
        for sequence in store.transition_sequences(
            horizon, stride=horizon, episode_ids=episode_ids
        ):
            sequences.append(sequence)
            if len(sequences) >= max_sequences:
                break
    if not sequences:
        raise ValueError("No selected episodes are long enough for the evaluation horizon")

    model = model.to(device)
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    representation_samples: list[torch.Tensor] = []
    target_name = (
        "latent"
        if model_kind in {"jepa", "lewm"}
        else "pixel"
        if model_kind in {"token", "flow", "delta"}
        or (model_kind == "rssm" and model.pixel_observations)
        else "state"
    )
    token_samples: list[torch.Tensor] = []

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        observations = _tensor(
            np.stack([[step.observation for step in sequence] for sequence in batch]),
            device,
        )
        next_observations = _tensor(
            np.stack([[step.next_observation for step in sequence] for sequence in batch]),
            device,
        )
        actions = torch.tensor(
            [[step.action for step in sequence] for sequence in batch],
            dtype=torch.long,
            device=device,
        )
        rewards = torch.tensor(
            [[step.reward for step in sequence] for sequence in batch],
            dtype=torch.float32,
            device=device,
        )
        dones = torch.tensor(
            [
                [float(step.terminated or step.truncated) for step in sequence]
                for sequence in batch
            ],
            dtype=torch.float32,
            device=device,
        )
        legal_masks = None
        if all(
            step.legal_mask is not None for sequence in batch for step in sequence
        ):
            legal_masks = torch.from_numpy(
                np.stack(
                    [[step.legal_mask for step in sequence] for sequence in batch]
                )
            ).to(device=device, dtype=torch.bool)

        current: torch.Tensor | None = None
        rssm_state = None
        token_context = None
        flow_latent = None
        lewm_embeddings = None
        lewm_actions = None
        persistence_target: torch.Tensor
        if model_kind in {"mlp", "ensemble"}:
            current = observations[:, 0]
            persistence_target = current
        elif model_kind == "jepa":
            assert isinstance(model, ActionConditionedJEPA)
            current = model.encode(observations[:, 0])
            persistence_target = current
        elif model_kind == "lewm":
            assert isinstance(model, ActionConditionedLeWorldModel)
            current = model.encode(observations[:, 0])
            persistence_target = current
            lewm_embeddings = current.unsqueeze(1)
            lewm_actions = torch.empty(
                len(batch), 0, dtype=torch.long, device=device
            )
        elif model_kind == "rssm":
            assert isinstance(model, RecurrentStateSpaceModel)
            persistence_target = observations[:, 0]
            rssm_state = model.initial(len(batch), device)
            bootstrap = torch.zeros(len(batch), dtype=torch.long, device=device)
            rssm_state, _ = model.observe(
                rssm_state, bootstrap, observations[:, 0], sample=False
            )
        elif model_kind == "token":
            assert isinstance(model, CausalTokenWorldModel)
            persistence_target = observations[:, 0]
            token_context = model.initial_context(observations[:, 0])
        elif model_kind == "flow":
            assert isinstance(model, RepresentationFlowWorldModel)
            persistence_target = observations[:, 0]
            flow_latent = model.encode(observations[:, 0])
        elif model_kind == "delta":
            assert isinstance(model, PixelDeltaWorldModel)
            current = observations[:, 0]
            persistence_target = current
        else:
            raise ValueError(f"Unsupported model kind: {model_kind}")

        for index in range(horizon):
            action_counterfactual_target: torch.Tensor | None = None
            action_counterfactual_valid: torch.Tensor | None = None
            if model_kind in {"mlp", "ensemble"}:
                assert isinstance(model, (DeterministicDynamics, DynamicsEnsemble))
                assert current is not None
                if legal_masks is not None and model.predict_legality:
                    _accumulate_legality(
                        totals,
                        counts,
                        model.legal_logits(current),
                        legal_masks[:, index],
                        index + 1,
                    )
                prediction = model(current, actions[:, index])
                predicted_target = prediction.next_state
                true_target = next_observations[:, index]
                predicted_reward = prediction.reward
                predicted_done = prediction.done_logit
                current = predicted_target
                if prediction.variance is not None:
                    _accumulate(
                        totals,
                        counts,
                        f"open_loop_epistemic_variance_h{index + 1}",
                        _mean_over_features(prediction.variance),
                    )
            elif model_kind == "jepa":
                assert isinstance(model, ActionConditionedJEPA)
                assert current is not None
                if index == 0 and legal_masks is not None:
                    alternative_action, action_counterfactual_valid = (
                        _alternative_legal_actions(
                            actions[:, index], legal_masks[:, index]
                        )
                    )
                    action_counterfactual_target = model.predict(
                        current, alternative_action
                    ).embedding
                prediction = model.predict(current, actions[:, index])
                predicted_target = prediction.embedding
                true_target = model.encode(next_observations[:, index])
                representation_samples.append(true_target.float().cpu())
                predicted_reward = prediction.reward
                predicted_done = prediction.done_logit
                current = predicted_target
            elif model_kind == "lewm":
                assert isinstance(model, ActionConditionedLeWorldModel)
                assert lewm_embeddings is not None and lewm_actions is not None
                lewm_actions = torch.cat(
                    (lewm_actions, actions[:, index : index + 1]), dim=1
                )
                prediction = model.next(lewm_embeddings, lewm_actions)
                predicted_target = prediction.embedding
                true_target = model.encode(next_observations[:, index])
                representation_samples.append(true_target.float().cpu())
                predicted_reward = prediction.reward
                predicted_done = prediction.done_logit
                lewm_embeddings = torch.cat(
                    (lewm_embeddings, predicted_target.unsqueeze(1)), dim=1
                )
            elif model_kind == "rssm":
                assert isinstance(model, RecurrentStateSpaceModel)
                assert rssm_state is not None
                rssm_state = model.imagine(rssm_state, actions[:, index], sample=False)
                predicted_target, predicted_reward, predicted_done = model.decode(rssm_state)
                true_target = next_observations[:, index]
            elif model_kind == "token":
                assert isinstance(model, CausalTokenWorldModel)
                assert token_context is not None
                token_step = model.generate_next(token_context, actions[:, index])
                token_context = token_step.context
                predicted_target = token_step.observation
                predicted_reward = token_step.reward
                predicted_done = token_step.done_logit
                true_target = next_observations[:, index]
                token_samples.append(model.encode_tokens(true_target).cpu())
            elif model_kind == "delta":
                assert isinstance(model, PixelDeltaWorldModel)
                assert current is not None
                if legal_masks is not None:
                    _accumulate_legality(
                        totals,
                        counts,
                        model.legal_logits(current),
                        legal_masks[:, index],
                        index + 1,
                    )
                delta_step = model(current, actions[:, index])
                predicted_target = delta_step.observation
                predicted_reward = delta_step.reward
                predicted_done = delta_step.done_logit
                true_target = next_observations[:, index]
                current = predicted_target
            else:
                assert isinstance(model, RepresentationFlowWorldModel)
                assert flow_latent is not None
                predicted_legal = model.legal_logits(flow_latent)
                if index == 0 and legal_masks is not None:
                    alternative_action, action_counterfactual_valid = (
                        _alternative_legal_actions(
                            actions[:, index], legal_masks[:, index]
                        )
                    )
                    action_counterfactual_target = model.imagine_latent(
                        flow_latent, alternative_action
                    ).observation
                flow_step = model.imagine_latent(flow_latent, actions[:, index])
                predicted_target = flow_step.observation
                predicted_reward = flow_step.reward
                predicted_done = flow_step.done_logit
                true_target = next_observations[:, index]
                flow_latent = flow_step.latent
                if legal_masks is not None:
                    true_legal = legal_masks[:, index]
                    legal_bce = F.binary_cross_entropy_with_logits(
                        predicted_legal, true_legal.to(predicted_legal.dtype), reduction="none"
                    ).mean(dim=1)
                    legal_accuracy = (
                        (predicted_legal.sigmoid() >= 0.5) == true_legal
                    ).float().mean(dim=1)
                    predicted_legal_mask = predicted_legal.sigmoid() >= 0.5
                    true_positive = (predicted_legal_mask & true_legal).sum(dim=1).float()
                    legal_recall = true_positive / true_legal.sum(dim=1).clamp_min(1)
                    legal_precision = true_positive / predicted_legal_mask.sum(
                        dim=1
                    ).clamp_min(1)
                    predicted_legal_fraction = predicted_legal_mask.float().mean(dim=1)
                    true_legal_fraction = true_legal.float().mean(dim=1)
                    _accumulate(
                        totals, counts, f"open_loop_legal_bce_h{index + 1}", legal_bce
                    )
                    _accumulate(
                        totals,
                        counts,
                        f"open_loop_legal_accuracy_h{index + 1}",
                        legal_accuracy,
                    )
                    for metric_name, metric_value in (
                        ("recall", legal_recall),
                        ("precision", legal_precision),
                        ("predicted_fraction", predicted_legal_fraction),
                        ("true_fraction", true_legal_fraction),
                    ):
                        _accumulate(
                            totals,
                            counts,
                            f"open_loop_legal_{metric_name}_h{index + 1}",
                            metric_value,
                        )

            step = index + 1
            target_error = _mean_over_features((predicted_target - true_target).square())
            if (
                action_counterfactual_target is not None
                and action_counterfactual_valid is not None
                and action_counterfactual_valid.any()
            ):
                counterfactual_error = _mean_over_features(
                    (action_counterfactual_target - true_target).square()
                )
                valid = action_counterfactual_valid
                _accumulate(
                    totals,
                    counts,
                    "open_loop_action_counterfactual_margin_h1",
                    counterfactual_error[valid] - target_error[valid],
                )
                _accumulate(
                    totals,
                    counts,
                    "open_loop_action_counterfactual_preference_h1",
                    (target_error[valid] < counterfactual_error[valid]).float(),
                )
            persistence_error = _mean_over_features(
                (persistence_target - true_target).square()
            )
            reward_error = (predicted_reward - rewards[:, index]).abs()
            done_error = (predicted_done.sigmoid() - dones[:, index]).square()
            _accumulate(
                totals, counts, f"open_loop_{target_name}_mse_h{step}", target_error
            )
            _accumulate(
                totals,
                counts,
                f"open_loop_{target_name}_persistence_mse_h{step}",
                persistence_error,
            )
            if model_kind in {"jepa", "lewm"}:
                cosine_distance = 1.0 - F.cosine_similarity(
                    predicted_target.float(), true_target.float(), dim=-1
                )
                _accumulate(
                    totals,
                    counts,
                    f"open_loop_latent_cosine_distance_h{step}",
                    cosine_distance,
                )
            _accumulate(totals, counts, f"open_loop_reward_mae_h{step}", reward_error)
            _accumulate(totals, counts, f"open_loop_done_brier_h{step}", done_error)

    metrics: dict[str, float | bool] = {
        name: totals[name] / counts[name] for name in totals
    }
    metrics.update(
        {
            "evaluated_sequences": float(len(sequences)),
            "evaluation_horizon": float(horizon),
            f"open_loop_{target_name}_mse_mean": float(
                np.mean(
                    [metrics[f"open_loop_{target_name}_mse_h{step}"] for step in range(1, horizon + 1)]
                )
            ),
            f"open_loop_{target_name}_persistence_mse_mean": float(
                np.mean(
                    [
                        metrics[f"open_loop_{target_name}_persistence_mse_h{step}"]
                        for step in range(1, horizon + 1)
                    ]
                )
            ),
            "open_loop_reward_mae_mean": float(
                np.mean(
                    [metrics[f"open_loop_reward_mae_h{step}"] for step in range(1, horizon + 1)]
                )
            ),
            "open_loop_done_brier_mean": float(
                np.mean(
                    [metrics[f"open_loop_done_brier_h{step}"] for step in range(1, horizon + 1)]
                )
            ),
        }
    )
    model_mse = float(metrics[f"open_loop_{target_name}_mse_mean"])
    persistence_mse = float(
        metrics[f"open_loop_{target_name}_persistence_mse_mean"]
    )
    metrics[f"open_loop_{target_name}_skill_vs_persistence"] = float(
        1.0 - model_mse / max(persistence_mse, 1e-12)
    )
    if model_kind in {"jepa", "lewm"}:
        metrics["open_loop_latent_cosine_distance_mean"] = float(
            np.mean(
                [
                    metrics[f"open_loop_latent_cosine_distance_h{step}"]
                    for step in range(1, horizon + 1)
                ]
            )
        )
    if model_kind == "ensemble":
        metrics["open_loop_epistemic_variance_mean"] = float(
            np.mean(
                [
                    metrics[f"open_loop_epistemic_variance_h{step}"]
                    for step in range(1, horizon + 1)
                ]
            )
        )
    if "open_loop_legal_bce_h1" in metrics:
        metrics["open_loop_legal_bce_mean"] = float(
            np.mean(
                [metrics[f"open_loop_legal_bce_h{step}"] for step in range(1, horizon + 1)]
            )
        )
        metrics["open_loop_legal_accuracy_mean"] = float(
            np.mean(
                [
                    metrics[f"open_loop_legal_accuracy_h{step}"]
                    for step in range(1, horizon + 1)
                ]
            )
        )
        for metric_name in (
            "recall",
            "precision",
            "predicted_fraction",
            "true_fraction",
        ):
            metrics[f"open_loop_legal_{metric_name}_mean"] = float(
                np.mean(
                    [
                        metrics[f"open_loop_legal_{metric_name}_h{step}"]
                        for step in range(1, horizon + 1)
                    ]
                )
            )
    if representation_samples:
        embeddings = torch.cat(representation_samples, dim=0).float()
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        feature_std = centered.std(dim=0, unbiased=False)
        covariance = centered.T @ centered / max(len(embeddings) - 1, 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        metrics.update(
            {
                "representation_feature_std_mean": float(feature_std.mean()),
                "representation_feature_std_min": float(feature_std.min()),
                "representation_active_fraction": float((feature_std > 0.01).float().mean()),
                "representation_effective_rank": float(entropy.exp()),
                "representation_top_eigenvalue_fraction": float(probabilities.max()),
            }
        )
    if token_samples:
        assert isinstance(model, CausalTokenWorldModel)
        tokens = torch.cat(token_samples, dim=0).flatten()
        counts = torch.bincount(tokens, minlength=model.vocab_size).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        metrics["token_codebook_utilization"] = float((counts > 0).float().mean())
        metrics["token_perplexity"] = float(
            torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        )
    return metrics


def build_world_model(checkpoint: dict[str, object]) -> WorldModel:
    model_kind = str(checkpoint["model"])
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    observation_shape = tuple(int(value) for value in checkpoint["observation_shape"])
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        args = {}
    hidden_dim = int(args.get("hidden_dim", 256))
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint state_dict is invalid")
    predict_legality = bool(args.get("predict_legality", False)) or any(
        "legal_head." in str(key) for key in state_dict
    )
    if model_kind == "mlp":
        model: WorldModel = DeterministicDynamics(
            state_dim,
            action_dim,
            hidden_dim,
            predict_legality=predict_legality,
        )
    elif model_kind == "ensemble":
        model = DynamicsEnsemble(
            int(args.get("ensemble_size", 5)),
            state_dim,
            action_dim,
            hidden_dim,
            predict_legality=predict_legality,
        )
    elif model_kind == "jepa":
        latent_dim = int(args.get("latent_dim", 128))
        encoder: nn.Module = (
            VectorEncoder(state_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
            if len(observation_shape) == 1
            else ConvEncoder(
                channels=observation_shape[0],
                latent_dim=latent_dim,
                pool_height=int(args.get("conv_pool_height", 1)),
                pool_width=int(args.get("conv_pool_width", 1)),
            )
        )
        model = ActionConditionedJEPA(
            encoder,
            action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
    elif model_kind == "lewm":
        if len(observation_shape) != 3:
            raise ValueError("LeWM checkpoints require channel-first pixel observations")
        latent_dim = int(args.get("latent_dim", 128))
        model = ActionConditionedLeWorldModel(
            ConvEncoder(
                channels=observation_shape[0],
                latent_dim=latent_dim,
                pool_height=int(args.get("conv_pool_height", 3)),
                pool_width=int(args.get("conv_pool_width", 5)),
            ),
            action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=int(args.get("transformer_layers", 4)),
            num_heads=int(args.get("transformer_heads", 4)),
            dropout=float(args.get("transformer_dropout", 0.1)),
            max_history=int(args.get("sequence_length", 16)),
            projection_head=bool(args.get("lewm_projection_head", False)),
            prediction_projection_head=bool(
                args.get("lewm_prediction_projection_head", False)
            ),
            projection_hidden_dim=(
                int(args["lewm_projection_hidden_dim"])
                if "lewm_projection_hidden_dim" in args
                else None
            ),
            projection_depth=int(
                args.get(
                    "lewm_projection_depth",
                    2 if "lewm_projection_hidden_dim" in args else 3,
                )
            ),
            residual_prediction=bool(args.get("lewm_residual_prediction", False)),
        )
    elif model_kind == "rssm":
        model = RecurrentStateSpaceModel(
            observation_shape if len(observation_shape) == 3 else state_dim,
            action_dim,
            deterministic_dim=int(args.get("deterministic_dim", 128)),
            stochastic_dim=int(args.get("stochastic_dim", 32)),
            hidden_dim=hidden_dim,
        )
    elif model_kind == "token":
        if len(observation_shape) != 3:
            raise ValueError("Token checkpoints require channel-first pixel observations")
        model = CausalTokenWorldModel(
            observation_shape,
            action_dim,
            vocab_size=int(args.get("token_vocab_size", 256)),
            code_dim=int(args.get("token_code_dim", 64)),
            quantizer=str(args.get("token_quantizer", "vq")),
            fsq_levels=tuple(int(value) for value in args.get("fsq_levels", (8, 8, 4))),
            token_height=int(args.get("token_height", 3)),
            token_width=int(args.get("token_width", 5)),
            tokenizer_hidden_dim=int(args.get("tokenizer_hidden_dim", 64)),
            model_dim=int(args.get("token_model_dim", 128)),
            num_layers=int(args.get("transformer_layers", 4)),
            num_heads=int(args.get("transformer_heads", 4)),
            dropout=float(args.get("transformer_dropout", 0.1)),
            max_transitions=int(args.get("sequence_length", 16)),
        )
    elif model_kind == "flow":
        if len(observation_shape) != 3:
            raise ValueError("Flow checkpoints require channel-first pixel observations")
        model = RepresentationFlowWorldModel(
            observation_shape,
            action_dim,
            code_dim=int(args.get("flow_code_dim", 32)),
            latent_height=int(args.get("flow_latent_height", 3)),
            latent_width=int(args.get("flow_latent_width", 5)),
            autoencoder_hidden_dim=int(args.get("flow_autoencoder_hidden_dim", 64)),
            model_dim=int(args.get("flow_model_dim", 128)),
            num_layers=int(args.get("flow_layers", 4)),
            num_heads=int(args.get("flow_heads", 4)),
            dropout=float(args.get("flow_dropout", 0.1)),
        )
    elif model_kind == "delta":
        if len(observation_shape) != 3:
            raise ValueError("Delta checkpoints require channel-first pixel observations")
        model = PixelDeltaWorldModel(
            observation_shape,
            action_dim,
            hidden_channels=int(args.get("delta_hidden_channels", 64)),
            residual_blocks=int(args.get("delta_residual_blocks", 3)),
        )
    else:
        raise ValueError(f"Unsupported checkpoint model: {model_kind}")
    if model_kind == "flow":
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key for key in incompatible.missing_keys if not key.startswith("legal_head.")
        ]
        if missing or unexpected:
            raise ValueError(
                f"Flow checkpoint is incompatible; missing={missing}, unexpected={unexpected}"
            )
    else:
        model.load_state_dict(state_dict)
    return model

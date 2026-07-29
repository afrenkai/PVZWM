from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import torch

from pvzwm.assets import sync_wiki_assets
from pvzwm.campaign import campaign_status, supported_campaign_stage_ids
from pvzwm.agents.policies import HeuristicPolicy, RandomPolicy
from pvzwm.agents.dqn import MaskedDoubleDQN
from pvzwm.agents.ppo import (
    ActorCritic,
    AutoregressiveSpatialConvActorCritic,
    POLICY_ARCHITECTURES,
    PolicyModel,
    SpatialActorCritic,
    SpatialConvActorCritic,
    VDNAutoregressiveSpatialConvActorCritic,
    build_actor_critic,
)
from pvzwm.agents.offline import (
    BehaviorCloneConfig,
    DiscreteIQLConfig,
    train_behavior_clone,
    train_discrete_iql,
)
from pvzwm.agents.planning import CategoricalCEMPlanner
from pvzwm.agents.imagination import FlowPPOTrainConfig, train_flow_ppo
from pvzwm.agents.trainers import (
    AgentTrainResult,
    DQNTrainConfig,
    DynaDQNTrainConfig,
    GroupPolicyTrainConfig,
    PPOTrainConfig,
    train_dqn,
    train_dyna_dqn,
    train_group_policy,
    train_ppo,
)
from pvzwm.data.store import ReplayStore, Transition
from pvzwm.data.audit import audit_data
from pvzwm.device import select_device
from pvzwm.distributed import initialize_distributed
from pvzwm.domain import ActionCodec, GameConfig
from pvzwm.env import PVZEnv
from pvzwm.evaluation import build_world_model, evaluate_world_model
from pvzwm.models.dynamics import DeterministicDynamics, DynamicsEnsemble
from pvzwm.models.delta import PixelDeltaWorldModel
from pvzwm.models.flow import RepresentationFlowWorldModel
from pvzwm.models.jepa import ActionConditionedJEPA, ConvEncoder, VectorEncoder
from pvzwm.models.lewm import ActionConditionedLeWorldModel
from pvzwm.models.rssm import RecurrentStateSpaceModel
from pvzwm.models.token import CausalTokenWorldModel
from pvzwm.rendering import SymbolicRenderer
from pvzwm.perception import (
    BoardPerceptionModel,
    ObservationVectorLayout,
    PerceptionTrainConfig,
    legal_mask_from_vector,
    train_perception,
)
from pvzwm.python_port import PythonPortConfig, PythonPortEnv, UPSTREAM_URL
from pvzwm.portable import (
    PortableEnv,
    PortableLoadoutCurriculum,
    PortableProcessConfig,
    PortableSequentialCurriculum,
    collect_sun_first_mask,
    portable_process_config_from_checkpoint,
)
from pvzwm.portable_archive import PortableArchiveCurriculum
from pvzwm.live import (
    LiveGameAdapter,
    NullPointerBackend,
    PillowScreenBackend,
    QuartzPointerBackend,
    ScreenGeometry,
    parse_actions,
    record_live_actions,
)
from pvzwm.training import TrainConfig, train_sequence_model, train_transition_model
from pvzwm.wiki_levels import sync_level_wiki


def _parse_fsq_levels(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("FSQ levels must be comma-separated integers") from error
    if not levels or any(level < 2 for level in levels):
        raise argparse.ArgumentTypeError("Every FSQ level must be at least two")
    return levels


def _parse_seed_types(value: str) -> tuple[int, ...]:
    try:
        seed_types = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Seed types must be comma-separated integer enum values"
        ) from error
    if not 1 <= len(seed_types) <= 10:
        raise argparse.ArgumentTypeError("A seed loadout must contain 1 to 10 types")
    if len(set(seed_types)) != len(seed_types):
        raise argparse.ArgumentTypeError("A seed loadout cannot contain duplicates")
    if any(not 0 <= seed_type < 48 for seed_type in seed_types):
        raise argparse.ArgumentTypeError("Seed types must be in [0, 48)")
    return seed_types


def _load_loadout_archive(path: Path) -> tuple[tuple[int, ...], ...]:
    payload = json.loads(path.read_text())
    raw_archive = payload.get("archive")
    if not isinstance(raw_archive, list) or not raw_archive:
        raise SystemExit("Portable loadout archive has no MAP-Elites entries")
    loadouts: list[tuple[int, ...]] = []
    for entry in raw_archive:
        raw_loadout = entry.get("loadout") if isinstance(entry, dict) else None
        if not isinstance(raw_loadout, list):
            raise SystemExit("Portable loadout archive contains an invalid entry")
        loadouts.append(tuple(int(item) for item in raw_loadout))
    return tuple(dict.fromkeys(loadouts))


def _checkpoint_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Return CLI arguments safe for JSON and weights-only checkpoints."""

    def convert(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return {
        key: convert(value)
        for key, value in vars(args).items()
        if key != "function"
    }


def collect(args: argparse.Namespace) -> None:
    config = GameConfig(max_time=args.max_time)
    renderer = SymbolicRenderer(config, cell_size=args.cell_size)
    policy_type = HeuristicPolicy if args.policy == "heuristic" else RandomPolicy
    with ReplayStore(args.database) as store:
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            env = PVZEnv(config, seed=seed)
            policy = policy_type() if args.policy == "heuristic" else policy_type(seed)
            observation, _ = env.reset(seed=seed)
            with store.episode(
                "symbolic", seed, {"policy": args.policy, "observation": args.observation}
            ) as writer:
                done = False
                while not done:
                    encoded_observation = (
                        observation.vector(config)
                        if args.observation == "vector"
                        else renderer.render(observation)
                    )
                    legal_mask = env.legal_action_mask()
                    action = policy.act(observation, env)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    writer.append(
                        Transition(
                            encoded_observation,
                            action,
                            reward,
                            next_observation.vector(config)
                            if args.observation == "vector"
                            else renderer.render(next_observation),
                            terminated,
                            truncated,
                            legal_mask,
                        )
                    )
                    observation = next_observation
                    done = terminated or truncated
            print(
                json.dumps(
                    {
                        "episode": episode_index + 1,
                        "return": writer.total_return,
                        "length": writer.length,
                        "won": observation.won,
                    }
                )
            )


def collect_python_port(args: argparse.Namespace) -> None:
    settings = PythonPortConfig(
        level=args.level,
        decision_ms=args.decision_ms,
        frame_ms=args.frame_ms,
        max_steps=args.max_steps,
        output_height=args.height,
        output_width=args.width,
        auto_collect_sun=not args.manual_sun,
        headless=not args.window,
        campaign_stage=args.campaign_stage,
    )
    policy_type = HeuristicPolicy if args.policy == "heuristic" else RandomPolicy
    env = PythonPortEnv(settings, seed=args.seed)
    try:
        with ReplayStore(args.database) as store:
            for episode_index in range(args.episodes):
                seed = args.seed + episode_index
                policy = policy_type() if args.policy == "heuristic" else policy_type(seed)
                observation, _ = env.reset(seed=seed)
                with store.episode(
                    "python_port",
                    seed,
                    {
                        "game_id": "marblexu_python_port",
                        "upstream": UPSTREAM_URL,
                        "package": "plants-vs-zombies==0.0.5",
                        "project_license": "GPL-3.0-only",
                        "upstream_asset_notice": "personal learning and noncommercial use",
                        "level": args.level,
                        "scenario_id": settings.scenario_id,
                        "upstream_level_file": (
                            None if args.campaign_stage else f"level_{args.level}.json"
                        ),
                        "campaign_stage": args.campaign_stage,
                        "policy": args.policy,
                        "observation": args.observation,
                        "decision_ms": args.decision_ms,
                        "frame_ms": args.frame_ms,
                        "auto_collect_sun": not args.manual_sun,
                    },
                ) as writer:
                    done = False
                    while not done:
                        encoded_observation = (
                            observation.vector(env.config)
                            if args.observation == "vector"
                            else env.render()
                        )
                        legal_mask = env.legal_action_mask()
                        action = policy.act(observation, env)
                        next_observation, reward, terminated, truncated, info = env.step(action)
                        writer.append(
                            Transition(
                                encoded_observation,
                                action,
                                reward,
                                next_observation.vector(env.config)
                                if args.observation == "vector"
                                else env.render(),
                                terminated,
                                truncated,
                                legal_mask,
                            )
                        )
                        observation = next_observation
                        done = terminated or truncated
                print(
                    json.dumps(
                        {
                            "episode": episode_index + 1,
                            "source": "python_port",
                            "level": args.level,
                            "scenario_id": settings.scenario_id,
                            "return": writer.total_return,
                            "length": writer.length,
                            "won": observation.won,
                            "kills": info["kills"],
                            "database": args.database,
                        }
                    )
                )
    finally:
        env.close()


def train_world_model(args: argparse.Namespace) -> None:
    if args.cpu_threads < 1:
        raise SystemExit("CPU thread count must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.dynamics_rollout_length < 1 or args.dynamics_rollout_stride < 1:
        raise SystemExit("Dynamics rollout length and stride must be positive")
    if args.regularization_weight is None:
        if args.model == "lewm" and args.lewm_loss_mixing == "additive":
            # Released LeWM uses prediction + .09 * Gaussian regularizer.
            args.regularization_weight = 0.09
        else:
            # Released VISReg image pretraining uses the convex lambda=.9.
            # Keep the historical .05 default for other regularizers, whose
            # scales are not interchangeable.
            args.regularization_weight = (
                0.9 if args.regularizer == "visreg" else 0.05
            )
    distributed = initialize_distributed(args.device)
    device = distributed.device_info
    torch.manual_seed(args.seed)
    with ReplayStore(args.database, read_only=True) as store:
        if args.split_by_metadata is None:
            training_episode_ids, validation_episode_ids = store.split_episode_ids(
                args.source, args.validation_fraction, args.split_seed
            )
        else:
            training_episode_ids, validation_episode_ids = (
                store.split_episode_ids_grouped(
                    args.split_by_metadata,
                    args.source,
                    args.validation_fraction,
                    args.split_seed,
                )
            )
        first = next(
            store.transitions(limit=1, episode_ids=training_episode_ids), None
        )
        if first is None:
            source_note = "" if args.source is None else f" for source {args.source!r}"
            raise SystemExit(
                "Replay database has no training transitions"
                f"{source_note}; collect episodes first"
            )
        action_dim = store.infer_action_dim(
            training_episode_ids,
            fallback=ActionCodec(GameConfig()).size,
        )
    observation_shape = tuple(int(value) for value in first.observation.shape)
    state_dim = int(first.observation.size)
    if args.model == "mlp":
        if first.observation.ndim != 1:
            raise SystemExit("MLP dynamics currently require vector observations")
        model: torch.nn.Module = DeterministicDynamics(
            state_dim,
            action_dim,
            hidden_dim=args.hidden_dim,
            predict_legality=args.predict_legality,
        )
        loss_options: dict[str, object] = {
            "state_weight": args.state_weight,
            "state_change_weight": args.state_change_weight,
            "reward_weight": args.reward_weight,
            "done_weight": args.done_weight,
            "legal_weight": args.legal_weight,
            "action_dropout_probability": args.flow_action_dropout,
        }
    elif args.model == "ensemble":
        if first.observation.ndim != 1:
            raise SystemExit("Ensemble dynamics currently require vector observations")
        model = DynamicsEnsemble(
            args.ensemble_size,
            state_dim,
            action_dim,
            hidden_dim=args.hidden_dim,
            predict_legality=args.predict_legality,
        )
        loss_options = {
            "state_weight": args.state_weight,
            "state_change_weight": args.state_change_weight,
            "reward_weight": args.reward_weight,
            "done_weight": args.done_weight,
            "legal_weight": args.legal_weight,
        }
    elif args.model == "jepa":
        encoder = (
            VectorEncoder(state_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
            if first.observation.ndim == 1
            else ConvEncoder(
                channels=observation_shape[0],
                latent_dim=args.latent_dim,
                pool_height=args.conv_pool_height,
                pool_width=args.conv_pool_width,
            )
        )
        model = ActionConditionedJEPA(
            encoder,
            action_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        )
        loss_options = {
            "regularizer": args.regularizer,
            "regularization_weight": args.regularization_weight,
            "slices": args.slices,
            "reward_weight": args.reward_weight,
            "done_weight": args.done_weight,
        }
    elif args.model == "lewm":
        if first.observation.ndim != 3:
            raise SystemExit("LeWorldModel requires channel-first pixel observations")
        if args.regularizer not in {"visreg", "sigreg"}:
            raise SystemExit("LeWorldModel supports VISReg or SIGReg")
        model = ActionConditionedLeWorldModel(
            ConvEncoder(
                channels=observation_shape[0],
                latent_dim=args.latent_dim,
                pool_height=args.conv_pool_height,
                pool_width=args.conv_pool_width,
            ),
            action_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.transformer_layers,
            num_heads=args.transformer_heads,
            dropout=args.transformer_dropout,
            max_history=args.sequence_length,
            projection_head=args.lewm_projection_head,
            prediction_projection_head=args.lewm_prediction_projection_head,
            projection_hidden_dim=args.lewm_projection_hidden_dim,
            projection_depth=args.lewm_projection_depth,
            residual_prediction=args.lewm_residual_prediction,
        )
        if args.initialize_representation_from is not None:
            representation_checkpoint = torch.load(
                args.initialize_representation_from,
                map_location="cpu",
                weights_only=True,
            )
            if representation_checkpoint.get("model") != "lewm":
                raise SystemExit("Representation initialization requires a LeWM checkpoint")
            representation_state = representation_checkpoint.get("state_dict")
            if not isinstance(representation_state, dict):
                raise SystemExit("Representation checkpoint has no state dictionary")
            model.load_representation_state(representation_state)
        if args.freeze_representation:
            model.freeze_representation()
        loss_options = {
            "regularizer": args.regularizer,
            "regularization_weight": args.regularization_weight,
            "slices": args.slices,
            "reward_weight": args.reward_weight,
            "done_weight": args.done_weight,
            "loss_mixing": args.lewm_loss_mixing,
        }
    elif args.model == "token":
        if first.observation.ndim != 3:
            raise SystemExit("Token dynamics require channel-first pixel observations")
        model = CausalTokenWorldModel(
            observation_shape,
            action_dim,
            vocab_size=args.token_vocab_size,
            code_dim=args.token_code_dim,
            quantizer=args.token_quantizer,
            fsq_levels=args.fsq_levels,
            token_height=args.token_height,
            token_width=args.token_width,
            tokenizer_hidden_dim=args.tokenizer_hidden_dim,
            model_dim=args.token_model_dim,
            num_layers=args.transformer_layers,
            num_heads=args.transformer_heads,
            dropout=args.transformer_dropout,
            max_transitions=args.sequence_length,
        )
        loss_options = {
            "tokenizer_weight": args.tokenizer_weight,
            "codebook_weight": args.codebook_weight,
            "commitment_beta": args.commitment_beta,
            "reward_weight": args.reward_weight,
            "done_weight": args.done_weight,
        }
    elif args.model == "flow":
        if first.observation.ndim != 3:
            raise SystemExit("Representation-flow dynamics require channel-first pixels")
        model = RepresentationFlowWorldModel(
            observation_shape,
            action_dim,
            code_dim=args.flow_code_dim,
            latent_height=args.flow_latent_height,
            latent_width=args.flow_latent_width,
            autoencoder_hidden_dim=args.flow_autoencoder_hidden_dim,
            model_dim=args.flow_model_dim,
            num_layers=args.flow_layers,
            num_heads=args.flow_heads,
            dropout=args.flow_dropout,
        )
        loss_options = {
            "reconstruction_weight": args.reconstruction_weight,
            "flow_weight": args.flow_weight,
            "visreg_weight": args.flow_visreg_weight,
            "visreg_slices": args.slices,
            "reward_weight": args.reward_weight,
            "done_weight": args.done_weight,
            "legal_weight": args.legal_weight,
        }
    elif args.model == "delta":
        if first.observation.ndim != 3:
            raise SystemExit("Pixel-delta dynamics require channel-first pixels")
        model = PixelDeltaWorldModel(
            observation_shape,
            action_dim,
            hidden_channels=args.delta_hidden_channels,
            residual_blocks=args.delta_residual_blocks,
        )
        loss_options = {
            "pixel_weight": args.delta_pixel_weight,
            "changed_pixel_weight": args.delta_changed_pixel_weight,
            "change_threshold": args.delta_change_threshold,
            "reward_weight": args.reward_weight,
            "done_weight": args.done_weight,
            "legal_weight": args.legal_weight,
        }
    else:
        model = RecurrentStateSpaceModel(
            observation_shape if first.observation.ndim == 3 else state_dim,
            action_dim,
            deterministic_dim=args.deterministic_dim,
            stochastic_dim=args.stochastic_dim,
            hidden_dim=args.hidden_dim,
        )
        loss_options = {
            "kl_weight": args.kl_weight,
            "free_nats": args.free_nats,
            "reward_weight": args.reward_weight,
            "done_weight": args.done_weight,
        }
    training_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        warmup_epochs=args.warmup_epochs,
        final_learning_rate_divisor=args.final_learning_rate_divisor,
        num_workers=args.num_workers,
    )
    dynamics_rollout = (
        args.model in {"mlp", "ensemble"} and args.dynamics_rollout_length > 1
    )
    if args.model in {"rssm", "token", "lewm"} or dynamics_rollout:
        history = train_sequence_model(
            model,
            args.database,
            device,
            training_config,
            sequence_length=(
                args.dynamics_rollout_length
                if dynamics_rollout
                else args.sequence_length
            ),
            sequence_stride=(
                args.dynamics_rollout_stride
                if dynamics_rollout
                else args.sequence_stride
            ),
            episode_ids=training_episode_ids,
            distributed=distributed,
            include_legal_mask=(dynamics_rollout and args.predict_legality),
            **loss_options,
        )
    else:
        history = train_transition_model(
            model,
            args.database,
            device,
            training_config,
            episode_ids=training_episode_ids,
            distributed=distributed,
            include_legal_mask=(
                args.model in {"flow", "delta"}
                or (args.model in {"mlp", "ensemble"} and args.predict_legality)
            ),
            **loss_options,
        )
    distributed.barrier()
    if not distributed.is_main:
        distributed.close()
        return
    evaluation_episode_ids = validation_episode_ids or training_episode_ids
    evaluation_metrics = evaluate_world_model(
        model,
        args.model,
        args.database,
        evaluation_episode_ids,
        device.device,
        horizon=args.eval_horizon,
        max_sequences=args.eval_sequences,
        batch_size=args.eval_batch_size,
    )
    evaluation_metrics["evaluation_is_held_out"] = bool(validation_episode_ids)
    evaluation_metrics["training_episodes"] = float(len(training_episode_ids))
    evaluation_metrics["validation_episodes"] = float(len(validation_episode_ids))
    evaluation_metrics["world_size"] = float(distributed.world_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_arguments = _checkpoint_arguments(args)
    torch.save(
        {
            "model": args.model,
            "state_dict": model.state_dict(),
            "state_dim": state_dim,
            "observation_shape": observation_shape,
            "action_dim": action_dim,
            "history": history,
            "training_episode_ids": training_episode_ids,
            "validation_episode_ids": validation_episode_ids,
            "evaluation": evaluation_metrics,
            "args": checkpoint_arguments,
        },
        output,
    )
    metrics = {
        "device": device.backend,
        "output": str(output),
        **history[-1],
        **evaluation_metrics,
    }
    with ReplayStore(args.experiment_database) as store:
        experiment_id = store.record_experiment(
            args.model,
            checkpoint_arguments,
            metrics,
            seed=args.seed,
            device=device.backend,
        )
    print(json.dumps({"experiment_id": experiment_id, **metrics}))
    distributed.close()


def evaluate_world_model_checkpoint(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_world_model(checkpoint)
    if args.split == "validation":
        episode_ids = checkpoint.get("validation_episode_ids")
    elif args.split == "training":
        episode_ids = checkpoint.get("training_episode_ids")
    else:
        episode_ids = None
    if episode_ids is not None and not isinstance(episode_ids, list):
        raise SystemExit("Checkpoint episode split is invalid")
    if args.split != "all" and not episode_ids:
        raise SystemExit(f"Checkpoint has no {args.split} episodes")
    metrics = evaluate_world_model(
        model,
        str(checkpoint["model"]),
        args.database,
        episode_ids,
        device.device,
        horizon=args.horizon,
        max_sequences=args.sequences,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "database": None if args.database is None else str(args.database),
                "model": checkpoint["model"],
                "device": device.backend,
                "split": args.split,
                **metrics,
            }
        )
    )


def train_agent(args: argparse.Namespace) -> None:
    if args.agent != "grpo" and args.parameter_noise_std != 0.0:
        raise SystemExit("Parameter-space noise requires grouped GRPO")
    port_settings = None
    portable_settings = None
    env_factory = None
    if args.environment == "python-port":
        port_settings = PythonPortConfig(
            level=args.port_level,
            decision_ms=args.port_decision_ms,
            frame_ms=args.port_frame_ms,
            max_steps=args.port_episode_steps,
            campaign_stage=args.port_campaign_stage,
        )
        game_config = GameConfig(
            tick_seconds=port_settings.decision_ms / 1_000.0,
            max_time=port_settings.max_steps * port_settings.decision_ms / 1_000.0,
        )
        env_factory = lambda seed: PythonPortEnv(port_settings, seed=seed)
    elif args.environment == "portable":
        if args.portable_binary is None or args.portable_resources is None:
            raise SystemExit(
                "The portable environment requires --portable-binary and "
                "--portable-resources"
            )
        portable_settings = PortableProcessConfig(
            binary=args.portable_binary,
            resources=args.portable_resources,
            level=args.portable_level,
            ticks_per_step=args.portable_ticks_per_step,
            max_steps=args.portable_episode_steps,
            frame_width=0,
            frame_height=0,
            reward_mode=args.portable_reward_mode,
            disable_mowers=args.portable_disable_mowers,
            seed_types=args.portable_seed_types,
        )
        if (
            args.portable_curriculum_end is not None
            and args.portable_curriculum_end < args.portable_level
        ):
            raise SystemExit("Portable curriculum end cannot precede its start level")
        if not 0.0 <= args.portable_rehearsal_probability <= 1.0:
            raise SystemExit("Portable rehearsal probability must be in [0, 1]")
        if (
            args.portable_curriculum_end is None
            and args.portable_rehearsal_probability != 0.0
        ):
            raise SystemExit("Portable rehearsal requires --portable-curriculum-end")
        if args.portable_completed_through >= args.portable_level:
            raise SystemExit("Completed portable levels must precede the start level")
        if (
            args.portable_curriculum_end is None
            and args.portable_completed_through != 0
        ):
            raise SystemExit("Completed portable levels require --portable-curriculum-end")
        if (
            args.portable_archive is not None
            and args.portable_curriculum_end is not None
        ):
            raise SystemExit(
                "Archive starts and sequential stage curricula cannot be combined"
            )
        if args.portable_loadout_archive is not None and (
            args.portable_archive is not None
            or args.portable_curriculum_end is not None
            or args.portable_seed_types is not None
        ):
            raise SystemExit(
                "Loadout curricula cannot be combined with a fixed loadout, "
                "archive starts, or a stage curriculum"
            )
        if not 0.0 <= args.portable_archive_start_probability <= 1.0:
            raise SystemExit("Portable archive start probability must be in [0, 1]")
        game_config = GameConfig(
            tick_seconds=args.portable_ticks_per_step / 100.0,
            max_time=(
                args.portable_episode_steps * args.portable_ticks_per_step / 100.0
            ),
        )
        if args.portable_loadout_archive is not None:
            loadout_curriculum = _load_loadout_archive(
                args.portable_loadout_archive
            )
            env_factory = lambda seed: PortableLoadoutCurriculum(
                portable_settings, loadout_curriculum
            )
        elif args.portable_archive is not None:
            env_factory = lambda seed: PortableArchiveCurriculum(
                portable_settings,
                args.portable_archive,
                start_probability=args.portable_archive_start_probability,
                frontier_fraction=args.portable_archive_frontier_fraction,
                minimum_depth=args.portable_archive_minimum_depth,
                maximum_depth=args.portable_archive_maximum_depth,
                branch_trace_window=args.portable_archive_trace_window,
            )
        else:
            env_factory = (
                (lambda seed: PortableEnv(portable_settings))
                if args.portable_curriculum_end is None
                else (
                    lambda seed: PortableSequentialCurriculum(
                        portable_settings,
                        args.portable_curriculum_end,
                        rehearsal_probability=args.portable_rehearsal_probability,
                        completed_levels=range(
                            1, args.portable_completed_through + 1
                        ),
                    )
                )
            )
    else:
        game_config = GameConfig(max_time=args.max_time, final_wave_time=args.final_wave_time)
    device = select_device(args.device)
    world_model_name = "none"
    world_model_metadata: dict[str, object] | None = None
    embedded_world_checkpoint: dict[str, object] | None = None
    agent_state_dim: int | None = None
    initialization_metadata: dict[str, object] | None = None
    if args.agent == "bc":
        if args.database is None:
            raise SystemExit("Behavior cloning requires --database")
        with ReplayStore(args.database, read_only=True) as store:
            if args.bc_top_fraction is not None:
                episode_ids = store.top_episode_ids(
                    args.bc_top_fraction, args.replay_source
                )
            else:
                episode_ids = (
                    store.successful_episode_ids(args.replay_source)
                    if args.bc_success_only
                    else store.episode_ids(args.replay_source)
                )
        if not episode_ids:
            raise SystemExit("Behavior-cloning replay selection has no episodes")
        config = BehaviorCloneConfig(
            epochs=args.bc_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            informative_fraction=args.bc_informative_fraction,
            sampling=args.bc_sampling,
            policy_architecture=args.policy_architecture,
        )
        agent, clone_result = train_behavior_clone(
            args.database,
            device,
            config,
            seed=args.seed,
            episode_ids=episode_ids,
        )
        result = AgentTrainResult([], [], [clone_result.loss])
        state_dict = agent.state_dict()
        agent_state_dim = agent.state_dim
        initialization_metadata = {
            "database": args.database,
            "replay_source": args.replay_source,
            "selection": (
                "top_return"
                if args.bc_top_fraction is not None
                else ("success" if args.bc_success_only else "all")
            ),
            "success_only": args.bc_success_only and args.bc_top_fraction is None,
            "top_fraction": args.bc_top_fraction,
            "episodes": clone_result.episodes,
            "transitions": clone_result.transitions,
            "accuracy": clone_result.accuracy,
            "non_wait_accuracy": clone_result.non_wait_accuracy,
        }
    elif args.agent == "iql":
        if args.database is None:
            raise SystemExit("Discrete IQL requires --database")
        initial_actor_state_dict = None
        initializer = None
        if args.initialize_from is not None:
            initial_checkpoint = torch.load(
                args.initialize_from, map_location="cpu", weights_only=True
            )
            if initial_checkpoint.get("agent") != "ppo":
                raise SystemExit("IQL actor initialization requires a PPO/BC/IQL actor")
            if int(initial_checkpoint.get("hidden_dim", -1)) != args.hidden_dim:
                raise SystemExit("IQL initializer hidden dimension does not match")
            if str(initial_checkpoint.get("policy_architecture", "flat")) != (
                args.policy_architecture
            ):
                raise SystemExit("IQL initializer policy architecture does not match")
            initial_actor_state_dict = initial_checkpoint.get("state_dict")
            if not isinstance(initial_actor_state_dict, dict):
                raise SystemExit("IQL initializer has no actor state dictionary")
            initializer = str(args.initialize_from)
        config = DiscreteIQLConfig(
            epochs=args.offline_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            expectile=args.iql_expectile,
            advantage_temperature=args.iql_advantage_temperature,
            max_advantage_weight=args.iql_max_advantage_weight,
            reward_scale=args.iql_reward_scale,
            target_tau=args.iql_target_tau,
            success_fraction=args.iql_success_fraction,
            informative_fraction=args.iql_informative_fraction,
            policy_architecture=args.policy_architecture,
        )
        agent, result = train_discrete_iql(
            args.database,
            device,
            config,
            seed=args.seed,
            source=args.replay_source,
            initial_actor_state_dict=initial_actor_state_dict,
        )
        state_dict = agent.state_dict()
        agent_state_dim = agent.state_dim
        initialization_metadata = {
            "database": args.database,
            "replay_source": args.replay_source,
            "actor_checkpoint": initializer,
            "episodes": result.episodes,
            "successful_episodes": result.successful_episodes,
            "transitions": result.transitions,
        }
    elif args.agent == "dqn":
        config: DQNTrainConfig | DynaDQNTrainConfig | PPOTrainConfig | GroupPolicyTrainConfig = DQNTrainConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            target_interval=args.target_interval,
            hidden_dim=args.hidden_dim,
        )
        agent, result = train_dqn(
            game_config, device, config, seed=args.seed, env_factory=env_factory
        )
        state_dict = agent.online.state_dict()
    elif args.agent == "flow-ppo":
        if args.database is None or args.world_model is None:
            raise SystemExit("Flow PPO requires --database and --world-model")
        if args.environment not in {"python-port", "portable"}:
            raise SystemExit(
                "Flow PPO real evaluation requires --environment python-port or portable"
            )
        world_checkpoint = torch.load(args.world_model, map_location="cpu", weights_only=True)
        world_model_name = str(world_checkpoint.get("model"))
        if world_model_name != "flow":
            raise SystemExit("Flow PPO requires a representation-flow checkpoint")
        raw_world_state = world_checkpoint.get("state_dict")
        if not isinstance(raw_world_state, dict) or not any(
            str(key).startswith("legal_head.") for key in raw_world_state
        ):
            raise SystemExit(
                "Flow PPO requires a flow checkpoint trained with the legality head"
            )
        world_model = build_world_model(world_checkpoint)
        assert isinstance(world_model, RepresentationFlowWorldModel)
        observation_shape = tuple(world_checkpoint["observation_shape"])
        if len(observation_shape) != 3 or observation_shape[0] != 3:
            raise SystemExit(
                "Flow PPO requires a three-channel pixel world-model checkpoint"
            )
        if args.environment == "python-port":
            assert port_settings is not None
            port_settings = replace(
                port_settings,
                output_height=int(observation_shape[1]),
                output_width=int(observation_shape[2]),
            )
            env_factory = lambda seed: PythonPortEnv(port_settings, seed=seed)
        else:
            assert portable_settings is not None
            portable_settings = replace(
                portable_settings,
                frame_height=int(observation_shape[1]),
                frame_width=int(observation_shape[2]),
            )
            env_factory = lambda seed: PortableEnv(portable_settings)
        if args.replay_source is None:
            episode_ids = world_checkpoint.get("training_episode_ids")
            if episode_ids is not None and not isinstance(episode_ids, list):
                raise SystemExit("World-model checkpoint training episode split is invalid")
        else:
            with ReplayStore(args.database, read_only=True) as store:
                episode_ids = store.episode_ids(args.replay_source)
        transitions_per_update = args.parallel_envs * args.imagination_horizon
        config = FlowPPOTrainConfig(
            updates=max(1, (args.steps + transitions_per_update - 1) // transitions_per_update),
            parallel_envs=args.parallel_envs,
            imagination_horizon=args.imagination_horizon,
            update_epochs=args.update_epochs,
            minibatch_size=args.batch_size,
            learning_rate=args.learning_rate,
            discount=args.discount,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.ppo_clip_ratio,
            value_weight=args.value_weight,
            entropy_weight=args.entropy_weight,
            hidden_dim=args.hidden_dim,
            flow_steps=args.flow_steps,
            stochastic_dynamics=not args.deterministic_dynamics,
            legality_threshold=args.legality_threshold,
            done_threshold=args.done_threshold,
            max_start_states=args.max_replay_transitions,
        )
        agent, result = train_flow_ppo(
            device,
            args.database,
            world_model,
            config,
            seed=args.seed,
            episode_ids=episode_ids,
        )
        state_dict = agent.state_dict()
        agent_state_dim = (
            world_model.code_dim
            * world_model.latent_height
            * world_model.latent_width
        )
        embedded_world_checkpoint = world_checkpoint
        world_model_metadata = {
            "checkpoint": args.world_model,
            "model": world_model_name,
            "database": args.database,
            "replay_source": args.replay_source,
            "episode_count": None if episode_ids is None else len(episode_ids),
            "embedded": True,
        }
    elif args.agent == "dyna-dqn":
        if args.database is None or args.world_model is None:
            raise SystemExit("Dyna DQN requires --database and --world-model")
        world_checkpoint = torch.load(args.world_model, map_location="cpu", weights_only=True)
        world_model_name = str(world_checkpoint.get("model"))
        if world_model_name not in {"mlp", "ensemble"}:
            raise SystemExit("Dyna DQN requires an MLP or ensemble dynamics checkpoint")
        world_model = build_world_model(world_checkpoint)
        assert isinstance(world_model, (DeterministicDynamics, DynamicsEnsemble))
        if args.replay_source is None:
            episode_ids = world_checkpoint.get("training_episode_ids")
            if episode_ids is not None and not isinstance(episode_ids, list):
                raise SystemExit("World-model checkpoint training episode split is invalid")
        else:
            with ReplayStore(args.database, read_only=True) as store:
                episode_ids = store.episode_ids(args.replay_source)
        config = DynaDQNTrainConfig(
            gradient_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            target_interval=args.target_interval,
            hidden_dim=args.hidden_dim,
            model_ratio=args.model_ratio,
            rollout_horizon=args.model_rollout_horizon,
            model_epsilon=args.model_epsilon,
            uncertainty_penalty=args.uncertainty_penalty,
            max_uncertainty=args.max_uncertainty,
            max_replay_transitions=args.max_replay_transitions,
            informative_replay_fraction=args.informative_replay_fraction,
        )
        agent, result = train_dyna_dqn(
            game_config,
            device,
            args.database,
            world_model,
            config,
            seed=args.seed,
            episode_ids=episode_ids,
        )
        state_dict = agent.online.state_dict()
        world_model_metadata = {
            "checkpoint": args.world_model,
            "model": world_model_name,
            "database": args.database,
            "replay_source": args.replay_source,
            "episode_count": None if episode_ids is None else len(episode_ids),
        }
    elif args.agent in {"reinforce", "grpo"}:
        if args.portable_curriculum_end is not None:
            raise SystemExit(
                "Grouped policy training requires a fixed portable level; advance stages between runs"
            )
        initial_state_dict = None
        if args.initialize_from is not None:
            initial_checkpoint = torch.load(
                args.initialize_from, map_location="cpu", weights_only=True
            )
            if initial_checkpoint.get("agent") != "ppo":
                raise SystemExit("Grouped policy initialization requires a PPO actor")
            if int(initial_checkpoint.get("hidden_dim", -1)) != args.hidden_dim:
                raise SystemExit("Grouped policy initializer hidden dimension does not match")
            if str(initial_checkpoint.get("policy_architecture", "flat")) != (
                args.policy_architecture
            ):
                raise SystemExit("Grouped policy initializer architecture does not match")
            initial_state_dict = initial_checkpoint.get("state_dict")
            if not isinstance(initial_state_dict, dict):
                raise SystemExit("Grouped policy initializer has no actor state dictionary")
            initialization_metadata = {"checkpoint": str(args.initialize_from)}
        config = GroupPolicyTrainConfig(
            algorithm=args.agent,
            steps=args.steps,
            group_size=args.group_size,
            update_epochs=args.update_epochs,
            minibatch_size=args.batch_size,
            learning_rate=args.learning_rate,
            discount=args.discount,
            clip_ratio=args.ppo_clip_ratio,
            entropy_weight=args.entropy_weight,
            reference_kl_weight=args.reference_kl_weight,
            parameter_noise_std=args.parameter_noise_std,
            importance_ratio_cap=args.importance_ratio_cap,
            win_bonus=args.group_win_bonus,
            hidden_dim=args.hidden_dim,
            policy_architecture=args.policy_architecture,
        )
        agent, result = train_group_policy(
            game_config,
            device,
            config,
            seed=args.seed,
            env_factory=env_factory,
            initial_state_dict=initial_state_dict,
        )
        state_dict = agent.state_dict()
    else:
        config = PPOTrainConfig(
            steps=args.steps,
            rollout_steps=args.rollout_steps,
            update_epochs=args.update_epochs,
            minibatch_size=args.batch_size,
            learning_rate=args.learning_rate,
            discount=args.discount,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.ppo_clip_ratio,
            value_weight=args.value_weight,
            entropy_weight=args.entropy_weight,
            hidden_dim=args.hidden_dim,
            intrinsic_method=args.intrinsic_method,
            intrinsic_reward_weight=args.intrinsic_reward_weight,
            intrinsic_reward_clip=args.intrinsic_reward_clip,
            adaptive_intrinsic_weight=args.adaptive_intrinsic_weight,
            intrinsic_lagrange_rate=args.intrinsic_lagrange_rate,
            intrinsic_performance_smoothing=args.intrinsic_performance_smoothing,
            rnd_hidden_dim=args.rnd_hidden_dim,
            rnd_feature_dim=args.rnd_feature_dim,
            rnd_learning_rate=args.rnd_learning_rate,
            rnd_update_epochs=args.rnd_update_epochs,
            policy_architecture=args.policy_architecture,
            vdn_importance_anneal_fraction=args.vdn_importance_anneal_fraction,
        )
        initial_state_dict = None
        initial_rnd_checkpoint = None
        if args.initialize_from is not None:
            initial_checkpoint = torch.load(
                args.initialize_from, map_location="cpu", weights_only=True
            )
            if initial_checkpoint.get("agent") not in {"ppo", "bc"}:
                raise SystemExit("PPO initialization requires a PPO/BC checkpoint")
            if int(initial_checkpoint.get("hidden_dim", -1)) != args.hidden_dim:
                raise SystemExit("PPO initializer hidden dimension does not match")
            initial_architecture = str(
                initial_checkpoint.get("policy_architecture", "flat")
            )
            compatible_vdn_initializer = (
                args.policy_architecture == "portable-vdn-autoregressive"
                and initial_architecture == "portable-autoregressive"
            )
            if (
                initial_architecture != args.policy_architecture
                and not compatible_vdn_initializer
            ):
                raise SystemExit("PPO initializer policy architecture does not match")
            initial_state_dict = initial_checkpoint.get("state_dict")
            if not isinstance(initial_state_dict, dict):
                raise SystemExit("PPO initializer has no state dictionary")
            initialization_metadata = {"checkpoint": str(args.initialize_from)}
            raw_auxiliary = initial_checkpoint.get("training_auxiliary")
            if isinstance(raw_auxiliary, dict) and raw_auxiliary.get(
                "algorithm"
            ) in {"rnd", "ride"}:
                initial_rnd_checkpoint = raw_auxiliary
        agent, result = train_ppo(
            game_config,
            device,
            config,
            seed=args.seed,
            env_factory=env_factory,
            initial_state_dict=initial_state_dict,
            initial_rnd_checkpoint=initial_rnd_checkpoint,
        )
        state_dict = agent.state_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = PVZEnv(game_config, seed=args.seed) if env_factory is None else env_factory(args.seed)
    observation, _ = env.reset(seed=args.seed)
    if args.agent in {"bc", "iql"} and (
        agent_state_dim != int(observation.vector(game_config).size)
        or agent.action_dim != env.codec.size
    ):
        env.close()
        raise SystemExit("Offline replay vocabulary does not match environment")
    if args.agent == "flow-ppo" and env.codec.size != world_model.action_dim:
        env.close()
        raise SystemExit(
            "Flow world-model action vocabulary does not match the Python port"
        )
    checkpoint = {
        "agent": (
            "ppo"
            if args.agent in {"bc", "iql", "reinforce", "grpo"}
            else args.agent
        ),
        "training_method": args.agent,
        "initialization": initialization_metadata,
        "environment": args.environment,
        "port_config": None if port_settings is None else asdict(port_settings),
        "portable_config": (
            None if portable_settings is None else portable_settings.checkpoint_dict()
        ),
        "portable_curriculum_end": args.portable_curriculum_end,
        "portable_rehearsal_probability": args.portable_rehearsal_probability,
        "portable_completed_through": args.portable_completed_through,
        "portable_archive": (
            None if args.portable_archive is None else str(args.portable_archive)
        ),
        "portable_loadout_archive": (
            None
            if args.portable_loadout_archive is None
            else str(args.portable_loadout_archive)
        ),
        "portable_archive_start_probability": args.portable_archive_start_probability,
        "portable_archive_frontier_fraction": args.portable_archive_frontier_fraction,
        "portable_archive_minimum_depth": args.portable_archive_minimum_depth,
        "portable_archive_maximum_depth": args.portable_archive_maximum_depth,
        "portable_archive_trace_window": args.portable_archive_trace_window,
        "portable_action_semantics": (
            "collect_sun_first_v1" if args.environment == "portable" else None
        ),
        "state_dict": state_dict,
        "state_dim": (
            agent_state_dim
            if agent_state_dim is not None
            else int(observation.vector(game_config).size)
        ),
        "action_dim": env.codec.size,
        "hidden_dim": args.hidden_dim,
        "policy_architecture": (
            args.policy_architecture
            if args.agent in {"ppo", "bc", "iql", "reinforce", "grpo"}
            else "flat"
        ),
        "game_config": asdict(game_config),
        "train_config": asdict(config),
        "summary": result.summary(),
        "episode_levels": result.episode_levels,
        "episode_returns": result.episode_returns,
        "episode_wins": result.episode_wins,
        "world_model": world_model_metadata,
        "world_model_checkpoint": embedded_world_checkpoint,
        "training_auxiliary": result.auxiliary_checkpoint,
    }
    env.close()
    torch.save(checkpoint, output)
    metrics: dict[str, object] = {"output": str(output), **result.summary()}
    if initialization_metadata is not None:
        metrics["initialization"] = initialization_metadata
    with ReplayStore(args.experiment_database) as store:
        experiment_id = store.record_experiment(
            world_model_name,
            asdict(config),
            metrics,
            agent=args.agent,
            seed=args.seed,
            device=device.backend,
        )
    print(json.dumps({"experiment_id": experiment_id, "device": device.backend, **metrics}))


def evaluate_agent(args: argparse.Namespace) -> None:
    if not 0.0 <= args.epsilon <= 1.0:
        raise SystemExit("--epsilon must be in [0, 1]")
    if args.portable_episode_steps is not None and args.portable_episode_steps < 1:
        raise SystemExit("--portable-episode-steps must be positive")
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device.device, weights_only=True)
    game_config = GameConfig(**checkpoint["game_config"])
    environment_name = checkpoint.get("environment", "clean-room")
    if environment_name == "python-port":
        raw_port_config = checkpoint.get("port_config")
        if not isinstance(raw_port_config, dict):
            raise SystemExit("Python-port checkpoint has no port configuration")
        port_settings = PythonPortConfig(**raw_port_config)
        env_factory = lambda seed: PythonPortEnv(port_settings, seed=seed)
    elif environment_name == "portable":
        raw_portable_config = checkpoint.get("portable_config")
        if not isinstance(raw_portable_config, dict):
            raise SystemExit("Portable checkpoint has no engine configuration")
        portable_settings = portable_process_config_from_checkpoint(
            raw_portable_config, state_dim=int(checkpoint["state_dim"])
        )
        if args.portable_level is not None:
            portable_settings = replace(portable_settings, level=args.portable_level)
        if args.portable_episode_steps is not None:
            portable_settings = replace(
                portable_settings, max_steps=args.portable_episode_steps
            )
        if args.portable_disable_mowers:
            portable_settings = replace(portable_settings, disable_mowers=True)
        if args.portable_seed_types is not None:
            portable_settings = replace(
                portable_settings, seed_types=args.portable_seed_types
            )
        env_factory = lambda seed: PortableEnv(portable_settings)
    else:
        if (
            args.portable_level is not None
            or args.portable_episode_steps is not None
            or args.portable_disable_mowers
            or args.portable_seed_types is not None
        ):
            raise SystemExit(
                "Portable evaluation overrides require a portable checkpoint"
            )
        env_factory = lambda seed: PVZEnv(game_config, seed=seed)
    flow_world_model: RepresentationFlowWorldModel | None = None
    if checkpoint["agent"] in {"dqn", "dyna-dqn"}:
        learner = MaskedDoubleDQN(
            checkpoint["state_dim"],
            checkpoint["action_dim"],
            checkpoint["hidden_dim"],
            device=device.device,
        )
        learner.online.load_state_dict(checkpoint["state_dict"])
        policy_model: MaskedDoubleDQN | PolicyModel = learner
    else:
        actor_critic = build_actor_critic(
            checkpoint["state_dim"],
            checkpoint["action_dim"],
            checkpoint["hidden_dim"],
            str(checkpoint.get("policy_architecture", "flat")),
        ).to(device.device)
        actor_critic.load_state_dict(checkpoint["state_dict"])
        policy_model = actor_critic
        if checkpoint["agent"] == "flow-ppo":
            raw_world_checkpoint = checkpoint.get("world_model_checkpoint")
            if not isinstance(raw_world_checkpoint, dict):
                raise SystemExit("Flow PPO checkpoint has no embedded world model")
            rebuilt = build_world_model(raw_world_checkpoint).to(device.device).eval()
            if not isinstance(rebuilt, RepresentationFlowWorldModel):
                raise SystemExit("Flow PPO checkpoint embeds the wrong world-model kind")
            flow_world_model = rebuilt

    returns: list[float] = []
    episode_lengths: list[int] = []
    wins = 0
    truncations = 0
    mower_trigger_counts: list[int] = []
    wins_without_mower_trigger = 0
    for episode_index in range(args.episodes):
        env = env_factory(args.seed + episode_index)
        observation, _ = env.reset(seed=args.seed + episode_index)
        episode_return = 0.0
        episode_length = 0
        done = False
        while not done:
            mask = torch.from_numpy(
                collect_sun_first_mask(env.codec, env.legal_action_mask())
            ).to(device.device)
            if checkpoint["agent"] == "flow-ppo":
                assert flow_world_model is not None
                if not isinstance(env, (PythonPortEnv, PortableEnv)):
                    raise SystemExit(
                        "Flow PPO evaluation requires a Python-port or portable renderer"
                    )
                frame = torch.from_numpy(env.render()).to(
                    device=device.device, dtype=torch.float32
                ).div(255.0)
                with torch.inference_mode():
                    state = flow_world_model.encode(frame.unsqueeze(0)).flatten(1)
                    assert isinstance(
                        policy_model,
                        (
                            ActorCritic,
                            SpatialActorCritic,
                            SpatialConvActorCritic,
                            AutoregressiveSpatialConvActorCritic,
                            VDNAutoregressiveSpatialConvActorCritic,
                        ),
                    )
                    logits, _ = policy_model(state)
                    masked_logits = logits.squeeze(0).masked_fill(~mask, -torch.inf)
                    action = int(
                        torch.distributions.Categorical(logits=masked_logits).sample().item()
                        if args.stochastic
                        else masked_logits.argmax().item()
                    )
            elif checkpoint["agent"] in {"dqn", "dyna-dqn"}:
                state = torch.from_numpy(observation.vector(game_config)).to(device.device)
                assert isinstance(policy_model, MaskedDoubleDQN)
                action = policy_model.act(state, mask, epsilon=args.epsilon)
            else:
                state = torch.from_numpy(observation.vector(game_config)).to(device.device)
                assert isinstance(
                    policy_model,
                    (
                        ActorCritic,
                        SpatialActorCritic,
                        SpatialConvActorCritic,
                        AutoregressiveSpatialConvActorCritic,
                        VDNAutoregressiveSpatialConvActorCritic,
                    ),
                )
                with torch.inference_mode():
                    logits, _ = policy_model(state.unsqueeze(0))
                    masked_logits = logits.squeeze(0).masked_fill(~mask, -torch.inf)
                    action = int(
                        torch.distributions.Categorical(logits=masked_logits).sample().item()
                        if args.stochastic
                        else masked_logits.argmax().item()
                    )
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated
        returns.append(episode_return)
        episode_lengths.append(episode_length)
        wins += int(observation.won)
        if environment_name == "portable":
            mower_trigger_counts.append(observation.mowers_triggered)
            wins_without_mower_trigger += int(
                observation.won and observation.mowers_triggered == 0
            )
        truncations += int(truncated and not terminated)
        env.close()
    print(
        json.dumps(
            {
                "agent": checkpoint["agent"],
                "environment": environment_name,
                "device": device.backend,
                "episodes": args.episodes,
                "portable_level": (
                    portable_settings.level
                    if environment_name == "portable"
                    else None
                ),
                "mean_return": sum(returns) / len(returns),
                "mean_length": sum(episode_lengths) / len(episode_lengths),
                "win_rate": wins / args.episodes,
                "truncation_rate": truncations / args.episodes,
                "portable_episode_steps": (
                    portable_settings.max_steps
                    if environment_name == "portable"
                    else None
                ),
                "portable_action_semantics": (
                    "collect_sun_first_v1"
                    if environment_name == "portable"
                    else None
                ),
                "portable_mowers_disabled": (
                    portable_settings.disable_mowers
                    if environment_name == "portable"
                    else None
                ),
                "mean_mowers_triggered": (
                    sum(mower_trigger_counts) / len(mower_trigger_counts)
                    if mower_trigger_counts
                    else None
                ),
                "episodes_using_mowers": sum(
                    count > 0 for count in mower_trigger_counts
                ),
                "wins_without_mower_trigger": wins_without_mower_trigger,
                "evaluation_mode": (
                    f"epsilon-{args.epsilon:g}"
                    if checkpoint["agent"] in {"dqn", "dyna-dqn"}
                    else "stochastic" if args.stochastic else "argmax"
                ),
            }
        )
    )


def evaluate_planner(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model") not in {"mlp", "ensemble"}:
        raise SystemExit("CEM planning requires an MLP or ensemble dynamics checkpoint")
    model = build_world_model(checkpoint).to(device.device).eval()
    assert isinstance(model, (DeterministicDynamics, DynamicsEnsemble))
    game_config = GameConfig(max_time=args.max_time)
    planner = CategoricalCEMPlanner(
        model,
        int(checkpoint["action_dim"]),
        horizon=args.horizon,
        samples=args.samples,
        elites=args.elites,
        iterations=args.iterations,
        uncertainty_penalty=args.uncertainty_penalty,
        illegal_action_penalty=args.illegal_action_penalty,
        elite_prior_strength=args.elite_prior_strength,
        config=game_config,
        device=device.device,
    )
    if args.environment == "python-port":
        port_settings = PythonPortConfig(
            level=args.port_level,
            campaign_stage=args.port_campaign_stage,
            decision_ms=args.port_decision_ms,
            frame_ms=args.port_frame_ms,
            max_steps=args.port_episode_steps,
        )
        env_factory = lambda seed: PythonPortEnv(port_settings, seed=seed)
    elif args.environment == "portable":
        if args.portable_binary is None or args.portable_resources is None:
            raise SystemExit(
                "The portable planner requires --portable-binary and "
                "--portable-resources"
            )
        portable_settings = PortableProcessConfig(
            binary=args.portable_binary,
            resources=args.portable_resources,
            level=args.portable_level,
            ticks_per_step=args.portable_ticks_per_step,
            max_steps=args.portable_episode_steps,
            frame_width=0,
            frame_height=0,
        )
        env_factory = lambda seed: PortableEnv(portable_settings)
    else:
        env_factory = lambda seed: PVZEnv(game_config, seed=seed)

    returns: list[float] = []
    scores: list[float] = []
    selected_actions: list[int] = []
    episode_lengths: list[int] = []
    wins = 0
    store = None if args.database is None else ReplayStore(args.database)
    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            env = env_factory(episode_seed)
            observation, reset_info = env.reset(seed=episode_seed)
            source = args.record_source
            if source is None:
                source = {
                    "portable": "pvz_portable_vector",
                    "python-port": "python_port",
                    "clean-room": "symbolic",
                }[args.environment]
            writer_context = (
                nullcontext(None)
                if store is None
                else store.episode(
                    source,
                    episode_seed,
                    {
                        "policy": "categorical_cem",
                        "checkpoint": str(args.checkpoint),
                        "environment": args.environment,
                        "level_id": reset_info.get("level"),
                        "portable_level": (
                            args.portable_level
                            if args.environment == "portable"
                            else None
                        ),
                        "horizon": args.horizon,
                        "samples": args.samples,
                        "elites": args.elites,
                        "iterations": args.iterations,
                        "uncertainty_penalty": args.uncertainty_penalty,
                        "illegal_action_penalty": args.illegal_action_penalty,
                        "elite_prior_strength": args.elite_prior_strength,
                    },
                )
            )
            try:
                with writer_context as writer:
                    episode_return = 0.0
                    episode_steps = 0
                    done = False
                    while not done:
                        state = observation.vector(env.config)
                        legal_mask = collect_sun_first_mask(
                            env.codec, env.legal_action_mask()
                        )
                        result = planner.plan(
                            state,
                            legal_mask,
                            env.action_support_mask(),
                        )
                        scores.append(result.score)
                        selected_actions.append(result.action)
                        next_observation, reward, terminated, truncated, _ = env.step(
                            result.action
                        )
                        if writer is not None:
                            writer.append(
                                Transition(
                                    state,
                                    result.action,
                                    reward,
                                    next_observation.vector(env.config),
                                    terminated,
                                    truncated,
                                    legal_mask,
                                )
                            )
                        observation = next_observation
                        episode_return += reward
                        episode_steps += 1
                        done = terminated or truncated
                    if writer is not None:
                        writer.metadata.update(
                            {
                                "won": bool(observation.won),
                                "mean_predicted_score": (
                                    sum(scores[-episode_steps:]) / episode_steps
                                ),
                            }
                        )
                returns.append(episode_return)
                episode_lengths.append(episode_steps)
                wins += int(observation.won)
            finally:
                env.close()
    finally:
        if store is not None:
            store.close()
    print(
        json.dumps(
            {
                "planner": "categorical_cem",
                "checkpoint": args.checkpoint,
                "model": checkpoint["model"],
                "environment": args.environment,
                "device": device.backend,
                "episodes": args.episodes,
                "mean_return": sum(returns) / len(returns),
                "mean_episode_steps": sum(episode_lengths) / len(episode_lengths),
                "win_rate": wins / args.episodes,
                "mean_predicted_score": sum(scores) / len(scores),
                "wait_fraction": selected_actions.count(0) / len(selected_actions),
                "distinct_selected_actions": len(set(selected_actions)),
                "uncertainty_penalty": args.uncertainty_penalty,
                "illegal_action_penalty": args.illegal_action_penalty,
                "elite_prior_strength": args.elite_prior_strength,
                "database": None if args.database is None else str(args.database),
            }
        )
    )


def record_live(args: argparse.Namespace) -> None:
    geometry = ScreenGeometry.load(args.geometry)
    pointer = QuartzPointerBackend() if args.execute else NullPointerBackend()
    adapter = LiveGameAdapter(
        geometry,
        PillowScreenBackend(),
        pointer,
        output_height=args.height,
        output_width=args.width,
        settle_seconds=args.settle_seconds,
        execute=args.execute,
    )
    actions = parse_actions(Path(args.actions).read_text().splitlines(), adapter.codec)
    if not actions:
        raise SystemExit("Action file contains no actions")
    with ReplayStore(args.database) as store:
        episode_id, count = record_live_actions(
            adapter,
            store,
            actions,
            seed=args.seed,
            source=args.source,
            game_id=args.game_id,
        )
    print(
        json.dumps(
            {
                "episode_id": episode_id,
                "transitions": count,
                "database": args.database,
                "execute": args.execute,
                "reward_labels": False,
                "game_id": args.game_id,
            }
        )
    )


def sync_assets(args: argparse.Namespace) -> None:
    manifest = sync_wiki_assets(args.output, download=args.download)
    print(
        json.dumps(
            {
                "output": args.output,
                "downloaded": args.download,
                "assets": len(manifest["assets"]),
                "manifest": str(Path(args.output) / "manifest.json"),
            }
        )
    )


def sync_campaign_wiki(args: argparse.Namespace) -> None:
    print(json.dumps(sync_level_wiki(args.database), indent=2 if args.pretty else None))


def data_status(args: argparse.Namespace) -> None:
    status = audit_data(args.data_root)
    print(json.dumps(status, indent=2 if args.pretty else None))


def merge_replays(args: argparse.Namespace) -> None:
    with ReplayStore(args.database) as store:
        result = store.merge_from(args.sources)
    print(json.dumps({"database": args.database, **result}))


def show_campaign_status(args: argparse.Namespace) -> None:
    print(json.dumps(campaign_status(), indent=2 if args.pretty else None))


def train_perception_model(args: argparse.Namespace) -> None:
    game_config = GameConfig(max_time=args.max_time, final_wave_time=args.final_wave_time)
    device = select_device(args.device)
    config = PerceptionTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        cell_size=args.cell_size,
    )
    model, losses = train_perception(game_config, device, config, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": "board_perception",
        "state_dict": model.state_dict(),
        "state_dim": model.head[-1].out_features,
        "channels": 3,
        "latent_dim": args.latent_dim,
        "game_config": asdict(game_config),
        "train_config": asdict(config),
        "last_loss": losses[-1],
    }
    torch.save(checkpoint, output)
    metrics: dict[str, object] = {
        "output": str(output),
        "steps": args.steps,
        "last_loss": losses[-1],
    }
    with ReplayStore(args.experiment_database) as store:
        experiment_id = store.record_experiment(
            "board_perception",
            asdict(config),
            metrics,
            seed=args.seed,
            device=device.backend,
        )
    print(json.dumps({"experiment_id": experiment_id, "device": device.backend, **metrics}))


def play_live(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    perception_checkpoint = torch.load(
        args.perception, map_location=device.device, weights_only=True
    )
    agent_checkpoint = torch.load(args.agent, map_location=device.device, weights_only=True)
    game_config = GameConfig(**perception_checkpoint["game_config"])
    layout = ObservationVectorLayout(game_config)
    if perception_checkpoint["state_dim"] != layout.size:
        raise SystemExit("Perception checkpoint has an incompatible state layout")
    if agent_checkpoint["state_dim"] != layout.size:
        raise SystemExit("Agent and perception checkpoints use different state dimensions")

    perception = BoardPerceptionModel(
        perception_checkpoint["state_dim"],
        perception_checkpoint["channels"],
        perception_checkpoint["latent_dim"],
    ).to(device.device)
    perception.load_state_dict(perception_checkpoint["state_dict"])
    perception.eval()
    if agent_checkpoint["agent"] in {"dqn", "dyna-dqn"}:
        dqn = MaskedDoubleDQN(
            agent_checkpoint["state_dim"],
            agent_checkpoint["action_dim"],
            agent_checkpoint["hidden_dim"],
            device=device.device,
        )
        dqn.online.load_state_dict(agent_checkpoint["state_dict"])
        dqn.online.eval()
        policy: MaskedDoubleDQN | PolicyModel = dqn
    else:
        actor_critic = build_actor_critic(
            agent_checkpoint["state_dim"],
            agent_checkpoint["action_dim"],
            agent_checkpoint["hidden_dim"],
            str(agent_checkpoint.get("policy_architecture", "flat")),
        ).to(device.device)
        actor_critic.load_state_dict(agent_checkpoint["state_dict"])
        actor_critic.eval()
        policy = actor_critic

    geometry = ScreenGeometry.load(args.geometry)
    pointer = QuartzPointerBackend() if args.execute else NullPointerBackend()
    adapter = LiveGameAdapter(
        geometry,
        PillowScreenBackend(),
        pointer,
        output_height=args.height,
        output_width=args.width,
        settle_seconds=args.settle_seconds,
        execute=args.execute,
    )
    if adapter.codec.size != agent_checkpoint["action_dim"]:
        raise SystemExit("Screen geometry and agent action dimensions differ")

    for step in range(args.steps):
        frame = torch.from_numpy(adapter.observe()).to(device.device).unsqueeze(0)
        with torch.inference_mode():
            state = perception(frame).squeeze(0)
        state_numpy = state.float().cpu().numpy()
        mask_numpy = legal_mask_from_vector(state_numpy, game_config)
        mask = torch.from_numpy(mask_numpy).to(device.device)
        if agent_checkpoint["agent"] in {"dqn", "dyna-dqn"}:
            assert isinstance(policy, MaskedDoubleDQN)
            action = policy.act(state, mask)
        else:
            assert isinstance(
                policy,
                (
                    ActorCritic,
                    SpatialActorCritic,
                    SpatialConvActorCritic,
                    AutoregressiveSpatialConvActorCritic,
                    VDNAutoregressiveSpatialConvActorCritic,
                ),
            )
            with torch.inference_mode():
                logits, _ = policy(state.unsqueeze(0))
                action = int(logits.squeeze(0).masked_fill(~mask, -torch.inf).argmax().item())
        structured = adapter.codec.decode(action)
        plan = adapter.perform(action)
        print(
            json.dumps(
                {
                    "step": step,
                    "action_id": action,
                    "kind": structured.kind.name,
                    "row": structured.row,
                    "col": structured.col,
                    "clicks": [[point.x, point.y] for point in plan],
                    "execute": args.execute,
                    "estimated_sun": float(state_numpy[layout.globals.start] * 1000.0),
                    "legal_actions": int(mask_numpy.sum()),
                }
            )
        )


def evaluate(args: argparse.Namespace) -> None:
    config = GameConfig(max_time=args.max_time)
    policy_type = HeuristicPolicy if args.policy == "heuristic" else RandomPolicy
    returns: list[float] = []
    wins = 0
    for episode_index in range(args.episodes):
        seed = args.seed + episode_index
        env = PVZEnv(config, seed=seed)
        policy = policy_type() if args.policy == "heuristic" else policy_type(seed)
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        done = False
        while not done:
            action = policy.act(observation, env)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
        wins += int(observation.won)
    print(
        json.dumps(
            {
                "policy": args.policy,
                "episodes": args.episodes,
                "mean_return": sum(returns) / len(returns),
                "win_rate": wins / args.episodes,
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pvzwm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collection = subparsers.add_parser("collect", help="Collect symbolic simulator episodes")
    collection.add_argument("--database", default="data/replay.duckdb")
    collection.add_argument("--episodes", type=int, default=10)
    collection.add_argument("--policy", choices=("random", "heuristic"), default="random")
    collection.add_argument("--seed", type=int, default=0)
    collection.add_argument("--max-time", type=float, default=180.0)
    collection.add_argument("--observation", choices=("vector", "pixels"), default="vector")
    collection.add_argument("--cell-size", type=int, default=12)
    collection.set_defaults(function=collect)

    port_collection = subparsers.add_parser(
        "collect-port", help="Collect from the GPL-selected Python/Pygame PvZ port"
    )
    port_collection.add_argument("--database", default="data/port_replay.duckdb")
    port_collection.add_argument("--episodes", type=int, default=10)
    port_collection.add_argument(
        "--policy", choices=("random", "heuristic"), default="heuristic"
    )
    port_collection.add_argument("--seed", type=int, default=0)
    port_collection.add_argument("--level", type=int, choices=range(0, 4), default=1)
    port_collection.add_argument(
        "--campaign-stage", choices=supported_campaign_stage_ids(), default=None
    )
    port_collection.add_argument("--max-steps", type=int, default=1_200)
    port_collection.add_argument("--decision-ms", type=int, default=250)
    port_collection.add_argument("--frame-ms", type=int, default=25)
    port_collection.add_argument("--observation", choices=("vector", "pixels"), default="pixels")
    port_collection.add_argument("--height", type=int, default=120)
    port_collection.add_argument("--width", type=int, default=160)
    port_collection.add_argument(
        "--manual-sun",
        action="store_true",
        help="Disable automatic sun pickup (the current action space has no pickup action)",
    )
    port_collection.add_argument(
        "--window", action="store_true", help="Render through a visible SDL window"
    )
    port_collection.set_defaults(function=collect_python_port)

    training = subparsers.add_parser("train-wm", help="Train a baseline world model")
    training.add_argument("--database", default="data/replay.duckdb")
    training.add_argument(
        "--source",
        default=None,
        help="Train only on episodes with this exact source label",
    )
    training.add_argument("--validation-fraction", type=float, default=0.2)
    training.add_argument("--split-seed", type=int, default=0)
    training.add_argument(
        "--split-by-metadata",
        help="Hold out whole metadata groups, e.g. level_id for unseen-stage evaluation",
    )
    training.add_argument(
        "--model",
        choices=("mlp", "ensemble", "jepa", "lewm", "rssm", "token", "flow", "delta"),
        default="mlp",
    )
    training.add_argument("--ensemble-size", type=int, default=5)
    training.add_argument(
        "--regularizer",
        choices=("visreg", "sigreg", "vicreg", "none"),
        default="visreg",
    )
    training.add_argument(
        "--regularization-weight",
        type=float,
        default=None,
        help="Regularizer weight; defaults to .09 for additive LeWM, .9 for convex VISReg, or .05 otherwise",
    )
    training.add_argument("--slices", type=int, default=64)
    training.add_argument("--latent-dim", type=int, default=128)
    training.add_argument("--conv-pool-height", type=int, default=3)
    training.add_argument("--conv-pool-width", type=int, default=5)
    training.add_argument("--deterministic-dim", type=int, default=128)
    training.add_argument("--stochastic-dim", type=int, default=32)
    training.add_argument("--sequence-length", type=int, default=16)
    training.add_argument("--sequence-stride", type=int, default=4)
    training.add_argument(
        "--dynamics-rollout-length",
        type=int,
        default=1,
        help="Open-loop training horizon for MLP and ensemble vector dynamics",
    )
    training.add_argument(
        "--dynamics-rollout-stride",
        type=int,
        default=4,
        help="Replay-window stride for multi-step vector dynamics training",
    )
    training.add_argument("--kl-weight", type=float, default=0.1)
    training.add_argument("--free-nats", type=float, default=1.0)
    training.add_argument("--token-vocab-size", type=int, default=256)
    training.add_argument("--token-code-dim", type=int, default=64)
    training.add_argument("--token-quantizer", choices=("vq", "fsq"), default="vq")
    training.add_argument("--fsq-levels", type=_parse_fsq_levels, default=(8, 8, 4))
    training.add_argument("--token-height", type=int, default=3)
    training.add_argument("--token-width", type=int, default=5)
    training.add_argument("--tokenizer-hidden-dim", type=int, default=64)
    training.add_argument("--token-model-dim", type=int, default=128)
    training.add_argument("--transformer-layers", type=int, default=4)
    training.add_argument("--transformer-heads", type=int, default=4)
    training.add_argument("--transformer-dropout", type=float, default=0.1)
    training.add_argument(
        "--lewm-projection-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the released VISReg-style BatchNorm projection MLP for new LeWM runs",
    )
    training.add_argument(
        "--lewm-prediction-projection-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project predicted LeWM latents through a separate BatchNorm MLP",
    )
    training.add_argument("--lewm-projection-hidden-dim", type=int, default=2048)
    training.add_argument(
        "--lewm-projection-depth", type=int, choices=(2, 3), default=2
    )
    training.add_argument(
        "--lewm-loss-mixing",
        choices=("additive", "convex"),
        default="additive",
        help="Use LeWM prediction + lambda*regularizer or the image-SSL convex mix",
    )
    training.add_argument(
        "--lewm-residual-prediction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict a latent delta initialized exactly at the persistence baseline",
    )
    training.add_argument("--initialize-representation-from", type=Path)
    training.add_argument(
        "--freeze-representation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    training.add_argument("--tokenizer-weight", type=float, default=1.0)
    training.add_argument("--codebook-weight", type=float, default=1.0)
    training.add_argument("--commitment-beta", type=float, default=1.0)
    training.add_argument("--flow-code-dim", type=int, default=32)
    training.add_argument("--flow-latent-height", type=int, default=3)
    training.add_argument("--flow-latent-width", type=int, default=5)
    training.add_argument("--flow-autoencoder-hidden-dim", type=int, default=64)
    training.add_argument("--flow-model-dim", type=int, default=128)
    training.add_argument("--flow-layers", type=int, default=4)
    training.add_argument("--flow-heads", type=int, default=4)
    training.add_argument("--flow-dropout", type=float, default=0.1)
    training.add_argument(
        "--flow-action-dropout",
        type=float,
        default=0.1,
        help="MIRA-style probability of dropping action conditioning during latent-flow training",
    )
    training.add_argument("--reconstruction-weight", type=float, default=1.0)
    training.add_argument("--flow-weight", type=float, default=1.0)
    training.add_argument("--flow-visreg-weight", type=float, default=0.01)
    training.add_argument("--delta-hidden-channels", type=int, default=64)
    training.add_argument("--delta-residual-blocks", type=int, default=3)
    training.add_argument("--delta-pixel-weight", type=float, default=1.0)
    training.add_argument("--delta-changed-pixel-weight", type=float, default=1.0)
    training.add_argument(
        "--delta-change-threshold", type=float, default=1.0 / 255.0
    )
    training.add_argument("--reward-weight", type=float, default=1.0)
    training.add_argument("--done-weight", type=float, default=0.25)
    training.add_argument("--legal-weight", type=float, default=0.25)
    training.add_argument(
        "--state-weight",
        type=float,
        default=1.0,
        help="Overall structured next-state loss weight",
    )
    training.add_argument(
        "--state-change-weight",
        type=float,
        default=0.0,
        help="Extra loss weight for state features that changed this transition",
    )
    training.add_argument(
        "--predict-legality",
        action="store_true",
        help="Train vector dynamics to model the environment's legal-action mask",
    )
    training.add_argument("--hidden-dim", type=int, default=256)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--num-workers", type=int, default=0)
    training.add_argument(
        "--cpu-threads",
        type=int,
        default=8,
        help="Maximum PyTorch intra-op CPU threads used by world-model training",
    )
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-5)
    training.add_argument("--gradient-clip", type=float, default=10.0)
    training.add_argument("--warmup-epochs", type=int, default=0)
    training.add_argument(
        "--final-learning-rate-divisor",
        type=float,
        default=1.0,
        help="Cosine-decay the learning rate to initial/divisor after warmup",
    )
    training.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    training.add_argument("--output", default="artifacts/world_model.pt")
    training.add_argument("--experiment-database", default="data/experiments.duckdb")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--eval-horizon", type=int, default=8)
    training.add_argument("--eval-sequences", type=int, default=128)
    training.add_argument("--eval-batch-size", type=int, default=16)
    training.set_defaults(function=train_world_model)

    world_model_evaluation = subparsers.add_parser(
        "eval-wm", help="Evaluate one-step and open-loop checkpoint predictions"
    )
    world_model_evaluation.add_argument("--checkpoint", required=True)
    world_model_evaluation.add_argument("--database", required=True)
    world_model_evaluation.add_argument(
        "--split", choices=("validation", "training", "all"), default="validation"
    )
    world_model_evaluation.add_argument("--horizon", type=int, default=8)
    world_model_evaluation.add_argument("--sequences", type=int, default=128)
    world_model_evaluation.add_argument("--batch-size", type=int, default=16)
    world_model_evaluation.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    world_model_evaluation.set_defaults(function=evaluate_world_model_checkpoint)

    agent_training = subparsers.add_parser("train-agent", help="Train a masked RL agent")
    agent_training.add_argument(
        "--agent",
        choices=(
            "dqn",
            "dyna-dqn",
            "ppo",
            "reinforce",
            "grpo",
            "flow-ppo",
            "bc",
            "iql",
        ),
        required=True,
    )
    agent_training.add_argument(
        "--environment",
        choices=("clean-room", "python-port", "portable"),
        default="clean-room",
    )
    agent_training.add_argument("--steps", type=int, default=50_000)
    agent_training.add_argument("--batch-size", type=int, default=128)
    agent_training.add_argument("--learning-rate", type=float, default=3e-4)
    agent_training.add_argument("--hidden-dim", type=int, default=256)
    agent_training.add_argument(
        "--policy-architecture",
        choices=POLICY_ARCHITECTURES,
        default="flat",
    )
    agent_training.add_argument("--replay-capacity", type=int, default=100_000)
    agent_training.add_argument("--initialize-from", type=Path)
    agent_training.add_argument("--bc-epochs", type=int, default=20)
    agent_training.add_argument("--offline-epochs", type=int, default=50)
    agent_training.add_argument(
        "--bc-success-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    agent_training.add_argument(
        "--bc-informative-fraction",
        type=float,
        default=0.5,
        help="Sample fraction assigned to non-wait demonstrations",
    )
    agent_training.add_argument(
        "--bc-sampling",
        choices=("informative", "branch-balanced"),
        default="informative",
        help="Balance BC replay by wait/informative actions or semantic action branch",
    )
    agent_training.add_argument(
        "--bc-top-fraction",
        type=float,
        default=None,
        help="Select only the highest-return fraction of completed replay episodes",
    )
    agent_training.add_argument("--iql-expectile", type=float, default=0.7)
    agent_training.add_argument(
        "--iql-advantage-temperature", type=float, default=3.0
    )
    agent_training.add_argument("--iql-max-advantage-weight", type=float, default=100.0)
    agent_training.add_argument("--iql-reward-scale", type=float, default=0.01)
    agent_training.add_argument("--iql-target-tau", type=float, default=0.005)
    agent_training.add_argument("--iql-success-fraction", type=float, default=0.5)
    agent_training.add_argument("--iql-informative-fraction", type=float, default=0.5)
    agent_training.add_argument("--warmup-steps", type=int, default=1_000)
    agent_training.add_argument("--target-interval", type=int, default=500)
    agent_training.add_argument(
        "--database",
        help="Replay database required by BC, IQL, Dyna DQN, or flow PPO",
    )
    agent_training.add_argument(
        "--world-model", help="Learned dynamics checkpoint required by model-based agents"
    )
    agent_training.add_argument(
        "--replay-source", help="Optional replay source filter for model-based agents"
    )
    agent_training.add_argument("--model-ratio", type=float, default=0.5)
    agent_training.add_argument("--model-rollout-horizon", type=int, default=1)
    agent_training.add_argument("--model-epsilon", type=float, default=0.1)
    agent_training.add_argument("--uncertainty-penalty", type=float, default=1.0)
    agent_training.add_argument("--max-uncertainty", type=float, default=float("inf"))
    agent_training.add_argument("--max-replay-transitions", type=int, default=None)
    agent_training.add_argument(
        "--informative-replay-fraction",
        type=float,
        default=0.0,
        help="Fraction of each real Dyna batch drawn from non-wait or rewarded transitions",
    )
    agent_training.add_argument("--parallel-envs", type=int, default=64)
    agent_training.add_argument("--imagination-horizon", type=int, default=16)
    agent_training.add_argument("--flow-steps", type=int, default=4)
    agent_training.add_argument(
        "--legality-threshold",
        type=float,
        default=None,
        help="Flow legality cutoff; omitted selects an F1-calibrated replay cutoff",
    )
    agent_training.add_argument("--done-threshold", type=float, default=0.5)
    agent_training.add_argument("--deterministic-dynamics", action="store_true")
    agent_training.add_argument("--rollout-steps", type=int, default=512)
    agent_training.add_argument("--update-epochs", type=int, default=4)
    agent_training.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Same-seed rollout count for GRPO (and batch episode count for REINFORCE)",
    )
    agent_training.add_argument(
        "--reference-kl-weight",
        type=float,
        default=0.01,
        help="Frozen-reference sampled KL weight for GRPO",
    )
    agent_training.add_argument(
        "--parameter-noise-std",
        type=float,
        default=0.0,
        help="Fixed per-rollout Gaussian policy-parameter noise for grouped RL",
    )
    agent_training.add_argument(
        "--importance-ratio-cap",
        type=float,
        default=10.0,
        help="Truncated noisy-to-clean behavior importance ratio for grouped RL",
    )
    agent_training.add_argument(
        "--group-win-bonus",
        type=float,
        default=0.0,
        help="Optional terminal win bonus added only to grouped rollout scores",
    )
    agent_training.add_argument("--discount", type=float, default=0.99)
    agent_training.add_argument("--gae-lambda", type=float, default=0.95)
    agent_training.add_argument("--ppo-clip-ratio", type=float, default=0.2)
    agent_training.add_argument("--value-weight", type=float, default=0.5)
    agent_training.add_argument("--entropy-weight", type=float, default=0.01)
    agent_training.add_argument(
        "--intrinsic-method",
        choices=("rnd", "ride"),
        default="rnd",
        help="Intrinsic exploration model used when intrinsic reward weight is nonzero",
    )
    agent_training.add_argument("--intrinsic-reward-weight", type=float, default=0.0)
    agent_training.add_argument("--intrinsic-reward-clip", type=float, default=5.0)
    agent_training.add_argument(
        "--adaptive-intrinsic-weight",
        action="store_true",
        help="CIM-style dual reduction of intrinsic weight when extrinsic return regresses",
    )
    agent_training.add_argument("--intrinsic-lagrange-rate", type=float, default=1.0)
    agent_training.add_argument(
        "--intrinsic-performance-smoothing", type=float, default=0.1
    )
    agent_training.add_argument("--rnd-hidden-dim", type=int, default=128)
    agent_training.add_argument("--rnd-feature-dim", type=int, default=64)
    agent_training.add_argument("--rnd-learning-rate", type=float, default=3e-4)
    agent_training.add_argument("--rnd-update-epochs", type=int, default=1)
    agent_training.add_argument(
        "--vdn-importance-anneal-fraction",
        type=float,
        default=1.0,
        help="Fraction of PPO training used to anneal VDN range weighting from uniform to full",
    )
    agent_training.add_argument("--max-time", type=float, default=180.0)
    agent_training.add_argument("--final-wave-time", type=float, default=145.0)
    agent_training.add_argument("--port-level", type=int, choices=range(0, 4), default=1)
    agent_training.add_argument("--port-decision-ms", type=int, default=250)
    agent_training.add_argument("--port-frame-ms", type=int, default=25)
    agent_training.add_argument("--port-episode-steps", type=int, default=1_200)
    agent_training.add_argument(
        "--port-campaign-stage", choices=supported_campaign_stage_ids(), default=None
    )
    agent_training.add_argument("--portable-binary", type=Path)
    agent_training.add_argument("--portable-resources", type=Path)
    agent_training.add_argument(
        "--portable-level", type=int, choices=range(1, 51), default=1
    )
    agent_training.add_argument(
        "--portable-curriculum-end",
        type=int,
        choices=range(1, 51),
        default=None,
        help="Advance one Adventure level after each win, through this level",
    )
    agent_training.add_argument(
        "--portable-rehearsal-probability",
        type=float,
        default=0.0,
        help="Revisit an already-unlocked stage with this reset probability",
    )
    agent_training.add_argument(
        "--portable-completed-through",
        type=int,
        choices=range(0, 50),
        default=0,
        help="Seed rehearsal with the authentic sequential prefix ending here",
    )
    agent_training.add_argument("--portable-ticks-per-step", type=int, default=25)
    agent_training.add_argument("--portable-episode-steps", type=int, default=3_600)
    agent_training.add_argument(
        "--portable-reward-mode",
        choices=("bridge", "clipped-damage"),
        default="bridge",
        help="Use raw bridge rewards or clip spawn penalties while retaining terminal rewards",
    )
    agent_training.add_argument(
        "--portable-disable-mowers",
        action="store_true",
        help="Remove all lane-clearing mowers at reset for robustness training",
    )
    agent_training.add_argument(
        "--portable-seed-types",
        type=_parse_seed_types,
        default=None,
        help="Exact comma-separated seed enum loadout; omit for the compatibility chooser",
    )
    agent_training.add_argument(
        "--portable-archive",
        type=Path,
        help="Go-Explore archive used only to restore policy-training start states",
    )
    agent_training.add_argument(
        "--portable-loadout-archive",
        type=Path,
        help="MAP-Elites JSON whose legal loadouts are sampled per clean episode",
    )
    agent_training.add_argument(
        "--portable-archive-start-probability", type=float, default=0.8
    )
    agent_training.add_argument(
        "--portable-archive-frontier-fraction", type=float, default=0.5
    )
    agent_training.add_argument(
        "--portable-archive-minimum-depth", type=int, default=1
    )
    agent_training.add_argument(
        "--portable-archive-maximum-depth", type=int, default=None
    )
    agent_training.add_argument(
        "--portable-archive-trace-window", type=int, default=64
    )
    agent_training.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    agent_training.add_argument("--seed", type=int, default=0)
    agent_training.add_argument("--output", default="artifacts/agent.pt")
    agent_training.add_argument("--experiment-database", default="data/experiments.duckdb")
    agent_training.set_defaults(function=train_agent)

    agent_evaluation = subparsers.add_parser(
        "eval-agent", help="Evaluate a learned policy checkpoint in the real environment"
    )
    agent_evaluation.add_argument("--checkpoint", required=True)
    agent_evaluation.add_argument("--episodes", type=int, default=10)
    agent_evaluation.add_argument("--seed", type=int, default=10_000)
    agent_evaluation.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="DQN evaluation exploration rate",
    )
    agent_evaluation.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actor policies instead of taking the masked argmax",
    )
    agent_evaluation.add_argument(
        "--portable-level",
        type=int,
        choices=range(1, 51),
        default=None,
        help="Override the Adventure level stored in a portable checkpoint",
    )
    agent_evaluation.add_argument(
        "--portable-episode-steps",
        type=int,
        default=None,
        help="Override the portable checkpoint's safety cap for complete long stages",
    )
    agent_evaluation.add_argument(
        "--portable-disable-mowers",
        action="store_true",
        help="Ablate all lane-clearing mowers at reset for robustness evaluation",
    )
    agent_evaluation.add_argument(
        "--portable-seed-types",
        type=_parse_seed_types,
        default=None,
        help="Override the checkpoint with an exact seed enum loadout",
    )
    agent_evaluation.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    agent_evaluation.set_defaults(function=evaluate_agent)

    planner_evaluation = subparsers.add_parser(
        "eval-planner", help="Evaluate categorical CEM with learned vector dynamics"
    )
    planner_evaluation.add_argument("--checkpoint", required=True)
    planner_evaluation.add_argument("--episodes", type=int, default=5)
    planner_evaluation.add_argument("--horizon", type=int, default=8)
    planner_evaluation.add_argument("--samples", type=int, default=256)
    planner_evaluation.add_argument("--elites", type=int, default=32)
    planner_evaluation.add_argument("--iterations", type=int, default=4)
    planner_evaluation.add_argument("--uncertainty-penalty", type=float, default=0.1)
    planner_evaluation.add_argument("--illegal-action-penalty", type=float, default=0.0)
    planner_evaluation.add_argument(
        "--elite-prior-strength",
        type=float,
        default=1.0,
        help="Total categorical prior mass mixed with each CEM elite histogram",
    )
    planner_evaluation.add_argument(
        "--database",
        type=Path,
        help="Optional DuckDB destination for complete planner trajectories",
    )
    planner_evaluation.add_argument(
        "--record-source",
        help="Replay source override; portable vector recording defaults to pvz_portable_vector",
    )
    planner_evaluation.add_argument(
        "--environment",
        choices=("clean-room", "python-port", "portable"),
        default="python-port",
    )
    planner_evaluation.add_argument("--port-level", type=int, choices=range(0, 4), default=1)
    planner_evaluation.add_argument(
        "--port-campaign-stage", choices=supported_campaign_stage_ids(), default=None
    )
    planner_evaluation.add_argument("--port-decision-ms", type=int, default=250)
    planner_evaluation.add_argument("--port-frame-ms", type=int, default=25)
    planner_evaluation.add_argument("--port-episode-steps", type=int, default=1_200)
    planner_evaluation.add_argument("--portable-binary", type=Path)
    planner_evaluation.add_argument("--portable-resources", type=Path)
    planner_evaluation.add_argument(
        "--portable-level", type=int, choices=range(1, 51), default=1
    )
    planner_evaluation.add_argument("--portable-ticks-per-step", type=int, default=25)
    planner_evaluation.add_argument("--portable-episode-steps", type=int, default=3_600)
    planner_evaluation.add_argument("--max-time", type=float, default=180.0)
    planner_evaluation.add_argument("--seed", type=int, default=10_000)
    planner_evaluation.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    planner_evaluation.set_defaults(function=evaluate_planner)

    evaluation = subparsers.add_parser("eval", help="Evaluate non-learning controls")
    evaluation.add_argument("--episodes", type=int, default=10)
    evaluation.add_argument("--policy", choices=("random", "heuristic"), default="heuristic")
    evaluation.add_argument("--seed", type=int, default=1000)
    evaluation.add_argument("--max-time", type=float, default=180.0)
    evaluation.set_defaults(function=evaluate)

    live_recording = subparsers.add_parser(
        "record-live", help="Record calibrated screen/action transitions"
    )
    live_recording.add_argument("--geometry", required=True)
    live_recording.add_argument("--actions", required=True)
    live_recording.add_argument("--database", default="data/live_replay.duckdb")
    live_recording.add_argument("--height", type=int, default=96)
    live_recording.add_argument("--width", type=int, default=160)
    live_recording.add_argument("--settle-seconds", type=float, default=0.5)
    live_recording.add_argument("--seed", type=int, default=0)
    live_recording.add_argument("--source", default="live_screen")
    live_recording.add_argument("--game-id", default="original_pvz")
    live_recording.add_argument(
        "--execute",
        action="store_true",
        help="Actually click the calibrated game window; omitted means capture-only dry run",
    )
    live_recording.set_defaults(function=record_live)

    perception_training = subparsers.add_parser(
        "train-perception", help="Train a synthetic pixel-to-board-state model"
    )
    perception_training.add_argument("--steps", type=int, default=2_000)
    perception_training.add_argument("--batch-size", type=int, default=64)
    perception_training.add_argument("--learning-rate", type=float, default=3e-4)
    perception_training.add_argument("--latent-dim", type=int, default=256)
    perception_training.add_argument("--cell-size", type=int, default=12)
    perception_training.add_argument("--max-time", type=float, default=180.0)
    perception_training.add_argument("--final-wave-time", type=float, default=145.0)
    perception_training.add_argument("--seed", type=int, default=0)
    perception_training.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    perception_training.add_argument("--output", default="artifacts/perception.pt")
    perception_training.add_argument(
        "--experiment-database", default="data/experiments.duckdb"
    )
    perception_training.set_defaults(function=train_perception_model)

    live_play = subparsers.add_parser(
        "play-live", help="Run perception and a masked agent against a calibrated window"
    )
    live_play.add_argument("--geometry", required=True)
    live_play.add_argument("--perception", required=True)
    live_play.add_argument("--agent", required=True)
    live_play.add_argument("--steps", type=int, default=20)
    live_play.add_argument("--height", type=int, default=96)
    live_play.add_argument("--width", type=int, default=160)
    live_play.add_argument("--settle-seconds", type=float, default=0.5)
    live_play.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    live_play.add_argument(
        "--execute",
        action="store_true",
        help="Actually click the game; omitted prints the predicted click plan only",
    )
    live_play.set_defaults(function=play_live)

    asset_sync = subparsers.add_parser(
        "sync-assets", help="Sync hashed PvZ1 wiki icon metadata or local-only files"
    )
    asset_sync.add_argument("--output", default="data/raw/wiki_gg")
    asset_sync.add_argument(
        "--download",
        action="store_true",
        help="Download files locally; without this flag only metadata is written",
    )
    asset_sync.set_defaults(function=sync_assets)

    campaign_sync = subparsers.add_parser(
        "sync-campaign-wiki",
        help="Archive revision-pinned wiki text for Adventure levels 1-1 through 5-10",
    )
    campaign_sync.add_argument("--database", default="data/campaign.duckdb")
    campaign_sync.add_argument("--pretty", action="store_true")
    campaign_sync.set_defaults(function=sync_campaign_wiki)

    status = subparsers.add_parser(
        "data-status", help="Audit synthetic, wiki, and original-game data separately"
    )
    status.add_argument("--data-root", default="data")
    status.add_argument("--pretty", action="store_true")
    status.set_defaults(function=data_status)

    replay_merge = subparsers.add_parser(
        "merge-replay", help="Merge independent DuckDB replay shards without decoding arrays"
    )
    replay_merge.add_argument("--database", required=True)
    replay_merge.add_argument("sources", nargs="+")
    replay_merge.set_defaults(function=merge_replays)

    campaign = subparsers.add_parser(
        "campaign-status", help="Show authentic Adventure and port-demo coverage separately"
    )
    campaign.add_argument("--pretty", action="store_true")
    campaign.set_defaults(function=show_campaign_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

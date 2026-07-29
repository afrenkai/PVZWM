from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import torch

from pvzwm.agents.dqn import MaskedDoubleDQN
from pvzwm.agents.ppo import PolicyModel, build_actor_critic
from pvzwm.data.store import ReplayStore, Transition
from pvzwm.device import select_device
from pvzwm.domain import GameConfig
from pvzwm.portable import (
    PortableEnv,
    PortableObservation,
    collect_sun_first_mask,
    portable_process_config_from_checkpoint,
)
from pvzwm.portable_collection import encode_portable_observation


def write_video_frame(
    stream: av.video.stream.VideoStream,
    container: av.container.OutputContainer,
    chw: np.ndarray,
) -> None:
    image = np.ascontiguousarray(np.transpose(chw, (1, 2, 0)))
    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


def choose_action(
    agent: MaskedDoubleDQN | PolicyModel,
    observation: PortableObservation,
    environment: PortableEnv,
    game_config: GameConfig,
    device: torch.device,
    *,
    epsilon: float,
    stochastic: bool,
) -> int:
    mask_tensor = torch.from_numpy(
        collect_sun_first_mask(environment.codec, environment.legal_action_mask())
    ).to(device)
    state = torch.from_numpy(observation.vector(game_config)).to(device)
    if isinstance(agent, MaskedDoubleDQN):
        return agent.act(state, mask_tensor, epsilon=epsilon)
    with torch.inference_mode():
        logits, _ = agent(state.unsqueeze(0))
        masked = logits.squeeze(0).masked_fill(~mask_tensor, -torch.inf)
        if stochastic:
            return int(torch.distributions.Categorical(logits=masked).sample().item())
        return int(masked.argmax().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a learned policy in the authentic portable engine"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--burn-in-episodes",
        type=int,
        default=0,
        help="Run unrecorded episodes first to reproduce a later stochastic-policy RNG state",
    )
    parser.add_argument("--level", type=int, choices=range(1, 51), default=None)
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="Override the checkpoint's safety cap for complete long stages",
    )
    parser.add_argument(
        "--observation", choices=("pixels", "vector"), default="pixels"
    )
    parser.add_argument("--seed", type=int, default=16_000)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--reseed-policy-per-episode",
        action="store_true",
        help="Reset policy sampling to the environment seed for matched ablations",
    )
    parser.add_argument(
        "--disable-mowers",
        action="store_true",
        help="Remove all mowers at reset for a policy robustness ablation",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        help="Keep a full-logical-resolution on-demand MP4 only when an episode wins",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-frame-stride", type=int, default=4)
    parser.add_argument("--stop-on-win", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")
    if args.episode_steps is not None and args.episode_steps < 1:
        raise SystemExit("--episode-steps must be positive")
    if args.burn_in_episodes < 0:
        raise SystemExit("--burn-in-episodes cannot be negative")
    if args.reseed_policy_per_episode and args.burn_in_episodes:
        raise SystemExit(
            "Per-episode policy reseeding cannot be combined with RNG burn-in episodes"
        )
    if not 0.0 <= args.epsilon <= 1.0:
        raise SystemExit("--epsilon must be in [0, 1]")
    if args.video_fps < 1 or args.video_frame_stride < 1:
        raise SystemExit("Video FPS and frame stride must be positive")

    device = select_device(args.device)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("agent") not in {"dqn", "dyna-dqn", "ppo"}:
        raise SystemExit("Portable recording requires a DQN, Dyna-DQN, or PPO checkpoint")
    raw_config = checkpoint.get("portable_config")
    if not isinstance(raw_config, dict):
        raise SystemExit("Checkpoint has no portable engine configuration")
    portable_config = replace(
        portable_process_config_from_checkpoint(
            raw_config, state_dim=int(checkpoint["state_dim"])
        ),
        level=(
            int(raw_config.get("level", 1)) if args.level is None else args.level
        ),
        max_steps=(
            int(raw_config.get("max_steps", 3_600))
            if args.episode_steps is None
            else args.episode_steps
        ),
        frame_width=args.width if args.observation == "pixels" else 0,
        frame_height=args.height if args.observation == "pixels" else 0,
        disable_mowers=args.disable_mowers,
    )
    game_config = GameConfig(**checkpoint["game_config"])
    if checkpoint["agent"] in {"dqn", "dyna-dqn"}:
        agent: MaskedDoubleDQN | PolicyModel = MaskedDoubleDQN(
            int(checkpoint["state_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint["hidden_dim"]),
            device=device.device,
        )
        agent.online.load_state_dict(checkpoint["state_dict"])
        agent.online.eval()
    else:
        agent = build_actor_critic(
            int(checkpoint["state_dim"]),
            int(checkpoint["action_dim"]),
            int(checkpoint["hidden_dim"]),
            str(checkpoint.get("policy_architecture", "flat")),
        ).to(device.device)
        agent.load_state_dict(checkpoint["state_dict"])
        agent.eval()

    for burn_index in range(args.burn_in_episodes):
        if args.reseed_policy_per_episode:
            torch.manual_seed(args.seed + burn_index)
        environment = PortableEnv(
            replace(portable_config, frame_width=0, frame_height=0)
        )
        try:
            observation, _ = environment.reset(seed=args.seed + burn_index)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = choose_action(
                    agent,
                    observation,
                    environment,
                    game_config,
                    device.device,
                    epsilon=args.epsilon,
                    stochastic=args.stochastic,
                )
                observation, _, terminated, truncated, _ = environment.step(action)
        finally:
            environment.close()

    with ReplayStore(args.database) as store:
        for episode_index in range(args.episodes):
            seed = args.seed + args.burn_in_episodes + episode_index
            if args.reseed_policy_per_episode:
                torch.manual_seed(seed)
            environment = PortableEnv(portable_config)
            video_container: av.container.OutputContainer | None = None
            video_stream: av.video.stream.VideoStream | None = None
            video_candidate: Path | None = None
            video_frames = 0
            episode_won = False
            try:
                observation, info = environment.reset(seed=seed)
                if args.video_output is not None:
                    args.video_output.parent.mkdir(parents=True, exist_ok=True)
                    video_candidate = args.video_output.with_name(
                        f".{args.video_output.stem}-candidate-{seed}.mp4"
                    )
                    video_container = av.open(str(video_candidate), mode="w")
                    video_stream = video_container.add_stream(
                        "libx264", rate=args.video_fps
                    )
                    video_stream.width = args.width
                    video_stream.height = args.height
                    video_stream.pix_fmt = "yuv420p"
                    video_stream.options = {"crf": "18", "preset": "medium"}
                    write_video_frame(
                        video_stream,
                        video_container,
                        environment.capture_frame(args.width, args.height),
                    )
                    video_frames += 1
                encoded = encode_portable_observation(
                    observation,
                    args.observation,
                    environment.render() if args.observation == "pixels" else None,
                )
                with store.episode(
                    (
                        "pvz_portable_vector"
                        if args.observation == "vector"
                        else "pvz_portable_learned_agent"
                    ),
                    seed,
                    {
                        "agent": checkpoint["agent"],
                        "checkpoint": str(args.checkpoint),
                        "level_id": info["level"],
                        "observation": args.observation,
                        "observation_shape": list(encoded.shape),
                        "vector_version": (
                            observation.vector_version
                            if args.observation == "vector"
                            else None
                        ),
                        "frame_resolution": (
                            [args.width, args.height]
                            if args.observation == "pixels"
                            else None
                        ),
                        "logical_resolution": [800, 600],
                        "episode_step_cap": portable_config.max_steps,
                        "evaluation_mode": (
                            f"epsilon-{args.epsilon:g}"
                            if checkpoint["agent"] in {"dqn", "dyna-dqn"}
                            else "stochastic" if args.stochastic else "argmax"
                        ),
                        "action_semantics": "collect_sun_first_v1",
                        "mowers_disabled": portable_config.disable_mowers,
                        "initial_mowers_ready": observation.mowers_ready,
                        "policy_rng_seed": args.seed,
                        "policy_reseeded_per_episode": args.reseed_policy_per_episode,
                        "burn_in_episodes": args.burn_in_episodes,
                        "protocol_version": observation.payload.get("protocol_version"),
                    },
                ) as writer:
                    terminated = False
                    truncated = False
                    while not (terminated or truncated):
                        mask = collect_sun_first_mask(
                            environment.codec, environment.legal_action_mask()
                        )
                        action = choose_action(
                            agent,
                            observation,
                            environment,
                            game_config,
                            device.device,
                            epsilon=args.epsilon,
                            stochastic=args.stochastic,
                        )
                        observation, reward, terminated, truncated, _ = environment.step(
                            action
                        )
                        next_encoded = encode_portable_observation(
                            observation,
                            args.observation,
                            environment.render()
                            if args.observation == "pixels"
                            else None,
                        )
                        writer.append(
                            Transition(
                                encoded,
                                action,
                                reward,
                                next_encoded,
                                terminated,
                                truncated,
                                mask,
                            )
                        )
                        if (
                            video_stream is not None
                            and video_container is not None
                            and (
                                writer.length % args.video_frame_stride == 0
                                or terminated
                                or truncated
                            )
                        ):
                            write_video_frame(
                                video_stream,
                                video_container,
                                environment.capture_frame(args.width, args.height),
                            )
                            video_frames += 1
                        encoded = next_encoded
                    writer.metadata.update(
                        {
                            "won": bool(observation.won),
                            "truncated": bool(truncated and not terminated),
                            "return": writer.total_return,
                            "video_frames": video_frames,
                            "mowers_triggered": observation.mowers_triggered,
                            "final_mowers_ready": observation.mowers_ready,
                        }
                    )
                episode_won = bool(observation.won)
                if video_stream is not None and video_container is not None:
                    for packet in video_stream.encode():
                        video_container.mux(packet)
                    video_container.close()
                    video_container = None
                    if episode_won:
                        assert video_candidate is not None
                        video_candidate.replace(args.video_output)
                    elif video_candidate is not None:
                        video_candidate.unlink(missing_ok=True)
                print(
                    json.dumps(
                        {
                            "episode_id": writer.episode_id,
                            "level": info["level"],
                            "seed": seed,
                            "length": writer.length,
                            "return": writer.total_return,
                            "won": observation.won,
                            "video": (
                                str(args.video_output)
                                if observation.won and args.video_output is not None
                                else None
                            ),
                            "video_frames": video_frames,
                            "mowers_disabled": portable_config.disable_mowers,
                            "mowers_triggered": observation.mowers_triggered,
                            "final_mowers_ready": observation.mowers_ready,
                            "database": str(args.database),
                        }
                    ),
                    flush=True,
                )
                if observation.won and args.stop_on_win:
                    break
            finally:
                if video_container is not None:
                    video_container.close()
                if video_candidate is not None and not episode_won:
                    video_candidate.unlink(missing_ok=True)
                environment.close()


if __name__ == "__main__":
    main()

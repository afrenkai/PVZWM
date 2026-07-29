from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass, replace
from enum import IntEnum
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any, Callable, IO, Sequence

import numpy as np


PROTOCOL_PREFIX = "PVZWM "


class PortableActionKind(IntEnum):
    WAIT = 0
    PLANT = 1
    SHOVEL = 2
    BOARD_CLICK = 3
    COLLECT = 4


@dataclass(frozen=True, slots=True)
class PortableAction:
    kind: PortableActionKind
    slot: int = 0
    row: int = 0
    col: int = 0


class PortableActionCodec:
    rows = 6
    cols = 9
    seed_slots = 10
    cells = rows * cols
    plant_begin = 1
    shovel_begin = plant_begin + seed_slots * cells
    board_click_begin = shovel_begin + cells
    collect_action = board_click_begin + cells
    size = collect_action + 1

    def encode(self, action: PortableAction) -> int:
        if action.kind == PortableActionKind.WAIT:
            return 0
        if action.kind == PortableActionKind.COLLECT:
            return self.collect_action
        if not (0 <= action.row < self.rows and 0 <= action.col < self.cols):
            raise ValueError("Action coordinates are outside the 6x9 logical board")
        cell = action.row * self.cols + action.col
        if action.kind == PortableActionKind.PLANT:
            if not 0 <= action.slot < self.seed_slots:
                raise ValueError("Seed slot is outside [0, 10)")
            return self.plant_begin + action.slot * self.cells + cell
        if action.kind == PortableActionKind.SHOVEL:
            return self.shovel_begin + cell
        if action.kind == PortableActionKind.BOARD_CLICK:
            return self.board_click_begin + cell
        raise ValueError(f"Unsupported portable action kind: {action.kind}")

    def decode(self, action_id: int) -> PortableAction:
        if action_id == 0:
            return PortableAction(PortableActionKind.WAIT)
        if action_id == self.collect_action:
            return PortableAction(PortableActionKind.COLLECT)
        if self.plant_begin <= action_id < self.shovel_begin:
            slot, cell = divmod(action_id - self.plant_begin, self.cells)
            row, col = divmod(cell, self.cols)
            return PortableAction(PortableActionKind.PLANT, slot, row, col)
        if self.shovel_begin <= action_id < self.board_click_begin:
            row, col = divmod(action_id - self.shovel_begin, self.cols)
            return PortableAction(PortableActionKind.SHOVEL, 0, row, col)
        if self.board_click_begin <= action_id < self.collect_action:
            row, col = divmod(action_id - self.board_click_begin, self.cols)
            return PortableAction(PortableActionKind.BOARD_CLICK, 0, row, col)
        raise ValueError(f"Action id {action_id} outside [0, {self.size})")

    def plant(self, slot: int, row: int, col: int) -> int:
        return self.encode(PortableAction(PortableActionKind.PLANT, slot, row, col))

    def shovel(self, row: int, col: int) -> int:
        return self.encode(PortableAction(PortableActionKind.SHOVEL, 0, row, col))

    def board_click(self, row: int, col: int) -> int:
        return self.encode(PortableAction(PortableActionKind.BOARD_CLICK, 0, row, col))


def collect_sun_first_mask(codec: object, legal_mask: np.ndarray) -> np.ndarray:
    """Make sun collection deterministic without changing other control choices.

    The authentic bridge exposes collection as a discrete action.  When it is
    legal, returning a singleton mask makes that housekeeping action carry zero
    policy choice (and therefore zero actor gradient).  The environment step is
    still retained in trajectories so dynamics models can learn its effect.
    Non-portable action vocabularies pass through unchanged.
    """

    mask = np.asarray(legal_mask, dtype=np.bool_)
    expected_shape = (int(getattr(codec, "size", -1)),)
    if mask.shape != expected_shape:
        raise ValueError(f"Expected legal mask shape {expected_shape}, got {mask.shape}")
    prioritized = mask.copy()
    if isinstance(codec, PortableActionCodec) and prioritized[codec.collect_action]:
        prioritized.fill(False)
        prioritized[codec.collect_action] = True
    return prioritized


@dataclass(frozen=True, slots=True)
class PortableProcessConfig:
    binary: Path
    resources: Path
    xvfb_run: str | None = "xvfb-run"
    level: int = 1
    ticks_per_step: int = 10
    # Three-flag Adventure stages can exceed 1,200 decisions at the default
    # tick rate. Keep a conservative full-stage budget; callers can still use a
    # smaller explicit cap for smoke tests.
    max_steps: int = 3_600
    startup_timeout_seconds: float = 120.0
    response_timeout_seconds: float = 30.0
    frame_width: int = 160
    frame_height: int = 120
    debugger: bool = False
    vector_version: int = 2
    reward_mode: str = "bridge"
    disable_mowers: bool = False
    seed_types: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.level <= 50:
            raise ValueError("Portable Adventure level must be in [1, 50]")
        if self.ticks_per_step < 1 or self.max_steps < 1:
            raise ValueError("Portable tick and episode-step counts must be positive")
        dimensions = (self.frame_width, self.frame_height)
        if any(value < 0 for value in dimensions):
            raise ValueError("Portable frame dimensions cannot be negative")
        if (self.frame_width == 0) != (self.frame_height == 0):
            raise ValueError("Portable frame dimensions must both be zero or both positive")
        if self.frame_width > 800 or self.frame_height > 600:
            raise ValueError("Portable frame dimensions cannot exceed the 800x600 source")
        if self.vector_version not in {1, 2}:
            raise ValueError("Portable vector version must be 1 or 2")
        if self.reward_mode not in {"bridge", "clipped-damage"}:
            raise ValueError("Portable reward mode must be bridge or clipped-damage")
        if self.seed_types is not None:
            if not 1 <= len(self.seed_types) <= PortableActionCodec.seed_slots:
                raise ValueError("Portable seed loadout must contain 1 to 10 types")
            if len(set(self.seed_types)) != len(self.seed_types):
                raise ValueError("Portable seed loadout cannot contain duplicates")
            if any(not 0 <= seed_type < 48 for seed_type in self.seed_types):
                raise ValueError("Portable seed types must be in [0, 48)")

    def command(self) -> list[str]:
        game = [str(self.binary), f"-resdir={self.resources}"]
        if self.debugger:
            game = [
                "gdb",
                "--batch",
                "-ex",
                "run",
                "-ex",
                "thread apply all bt",
                "--args",
                *game,
            ]
        if self.xvfb_run is None:
            return game
        return [
            self.xvfb_run,
            "-a",
            "-s",
            "-screen 0 800x600x24 -nolisten tcp",
            *game,
        ]

    def checkpoint_dict(self) -> dict[str, object]:
        return {
            "binary": str(self.binary),
            "resources": str(self.resources),
            "xvfb_run": self.xvfb_run,
            "level": self.level,
            "ticks_per_step": self.ticks_per_step,
            "max_steps": self.max_steps,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "response_timeout_seconds": self.response_timeout_seconds,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "debugger": self.debugger,
            "vector_version": self.vector_version,
            "reward_mode": self.reward_mode,
            "disable_mowers": self.disable_mowers,
            "seed_types": self.seed_types,
        }


def portable_process_config_from_checkpoint(
    raw_config: dict[str, object], *, state_dim: int | None = None
) -> PortableProcessConfig:
    """Restore portable settings while preserving pre-versioned vector checkpoints."""

    restored = dict(raw_config)
    if "vector_version" not in restored and state_dim == 250:
        restored["vector_version"] = 1
    if isinstance(restored.get("seed_types"), list):
        restored["seed_types"] = tuple(int(value) for value in restored["seed_types"])
    return PortableProcessConfig(**restored)


@dataclass(slots=True)
class PortableObservation:
    payload: dict[str, Any]
    vector_version: int = 2

    @property
    def won(self) -> bool:
        return bool(self.payload.get("won", False))

    @property
    def lost(self) -> bool:
        return int(self.payload.get("board_result", 0)) == 2

    @property
    def level_id(self) -> str:
        return f"{int(self.payload['area'])}-{int(self.payload['stage'])}"

    @property
    def mowers_ready(self) -> int:
        return int(self.payload.get("mowers_ready", 0))

    @property
    def mowers_triggered(self) -> int:
        return int(self.payload.get("mowers_triggered", 0))

    @property
    def mower_states(self) -> tuple[int, ...]:
        states = [-1] * PortableActionCodec.rows
        raw_mowers = self.payload.get("mowers", [])
        if not isinstance(raw_mowers, list):
            return tuple(states)
        for raw_mower in raw_mowers:
            if not isinstance(raw_mower, dict):
                continue
            row = int(raw_mower.get("row", -1))
            if 0 <= row < len(states):
                states[row] = int(raw_mower.get("state", -1))
        return tuple(states)

    @property
    def seed_bank_capacity(self) -> int:
        return int(self.payload.get("seed_bank_capacity", 0))

    @property
    def available_seed_types(self) -> tuple[int, ...]:
        raw_types = self.payload.get("available_seed_types", [])
        if not isinstance(raw_types, list):
            return ()
        return tuple(int(seed_type) for seed_type in raw_types)

    def frame(self) -> np.ndarray | None:
        frame = self.payload.get("frame")
        if frame is None:
            return None
        if not isinstance(frame, dict) or frame.get("encoding") != "rgb8-base64":
            raise RuntimeError("Portable bridge returned an unsupported frame encoding")
        width = int(frame["width"])
        height = int(frame["height"])
        if width <= 0 or height <= 0:
            raise RuntimeError("Portable bridge returned invalid frame dimensions")
        raw = base64.b64decode(str(frame["data"]), validate=True)
        expected = width * height * 3
        if len(raw) != expected:
            raise RuntimeError(
                f"Portable bridge frame has {len(raw)} bytes; expected {expected}"
            )
        hwc = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        return np.ascontiguousarray(hwc.transpose(2, 0, 1))

    def vector(self, _config: object | None = None) -> np.ndarray:
        version = int(getattr(_config, "vector_version", self.vector_version))
        if version == 1:
            return self._vector_v1()
        if version != 2:
            raise ValueError(f"Unsupported portable vector version: {version}")

        rows, cols = PortableActionCodec.rows, PortableActionCodec.cols
        plant_types = np.zeros((rows, cols), dtype=np.float32)
        plant_health = np.zeros((rows, cols), dtype=np.float32)
        plant_states = np.zeros((rows, cols), dtype=np.float32)
        zombie_counts = np.zeros((rows, cols), dtype=np.float32)
        zombie_types = np.zeros((rows, cols), dtype=np.float32)
        zombie_x = np.zeros((rows, cols), dtype=np.float32)
        zombie_body = np.zeros((rows, cols), dtype=np.float32)
        zombie_helm = np.zeros((rows, cols), dtype=np.float32)
        zombie_shield = np.zeros((rows, cols), dtype=np.float32)
        zombie_phases = np.zeros((rows, cols), dtype=np.float32)

        for plant in self.payload.get("plants", []):
            row, col = int(plant["row"]), int(plant["col"])
            if 0 <= row < rows and 0 <= col < cols:
                plant_types[row, col] = max(float(plant["type"]), 0.0) / 52.0
                plant_health[row, col] += max(float(plant["health"]), 0.0)
                plant_states[row, col] = max(float(plant.get("state", 0.0)), 0.0) / 100.0

        for zombie in self.payload.get("zombies", []):
            row = int(zombie["row"])
            x = float(zombie["x"])
            col = int(np.clip((x - 40.0) // 80.0, 0, cols - 1))
            if 0 <= row < rows:
                zombie_counts[row, col] += 1.0
                zombie_types[row, col] = max(
                    zombie_types[row, col], max(float(zombie["type"]), 0.0) / 40.0
                )
                zombie_x[row, col] += np.clip(x / 800.0, 0.0, 1.0)
                zombie_body[row, col] += max(float(zombie["body_health"]), 0.0)
                zombie_helm[row, col] += max(float(zombie["helm_health"]), 0.0)
                zombie_shield[row, col] += max(float(zombie["shield_health"]), 0.0)
                zombie_phases[row, col] = max(
                    zombie_phases[row, col],
                    max(float(zombie.get("phase", 0.0)), 0.0) / 100.0,
                )

        occupied = zombie_counts > 0
        zombie_x[occupied] /= zombie_counts[occupied]

        def log_scale(value: np.ndarray, maximum: float) -> np.ndarray:
            return np.clip(np.log1p(value) / np.log1p(maximum), 0.0, 1.0)

        packets = np.zeros((PortableActionCodec.seed_slots, 3), dtype=np.float32)
        for packet in self.payload.get("seed_packets", []):
            slot = int(packet["slot"])
            if 0 <= slot < PortableActionCodec.seed_slots:
                packets[slot, 0] = max(float(packet["type"]), 0.0) / 52.0
                packets[slot, 1] = float(bool(packet["active"]))
                packets[slot, 2] = min(max(float(packet["refresh"]), 0.0) / 5000.0, 1.0)

        scalars = np.asarray(
            [
                np.clip(float(self.payload.get("sun", 0.0)) / 1000.0, 0.0, 1.0),
                np.clip(float(self.payload.get("wave", 0.0)) / 20.0, 0.0, 1.0),
                float(self.won),
                float(self.lost),
                np.clip(float(self.payload.get("level", 1.0)) / 50.0, 0.0, 1.0),
                np.clip(float(self.payload.get("main_counter", 0.0)) / 20000.0, 0.0, 1.0),
                np.clip(float(self.payload.get("zombie_countdown", 0.0)) / 2000.0, 0.0, 1.0),
                np.clip(float(self.payload.get("num_waves", 0.0)) / 20.0, 0.0, 1.0),
                np.clip(float(self.payload.get("total_spawned_waves", 0.0)) / 20.0, 0.0, 1.0),
                float(bool(self.payload.get("level_complete", False))),
                np.clip(float(self.payload.get("coins_count", 0.0)) / 20.0, 0.0, 1.0),
                np.clip(float(self.payload.get("scene", 0.0)) / 5.0, 0.0, 1.0),
                np.clip(float(self.payload.get("tutorial_state", 0.0)) / 50.0, 0.0, 1.0),
                np.clip(float(self.payload.get("plants_count", 0.0)) / 54.0, 0.0, 1.0),
                np.clip(float(self.payload.get("zombies_count", 0.0)) / 100.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                plant_types.ravel(),
                log_scale(plant_health, 4000.0).ravel(),
                np.clip(plant_states, 0.0, 1.0).ravel(),
                np.clip(zombie_counts / 8.0, 0.0, 1.0).ravel(),
                np.clip(zombie_types, 0.0, 1.0).ravel(),
                zombie_x.ravel(),
                log_scale(zombie_body, 60000.0).ravel(),
                log_scale(zombie_helm, 10000.0).ravel(),
                log_scale(zombie_shield, 10000.0).ravel(),
                np.clip(zombie_phases, 0.0, 1.0).ravel(),
                packets.ravel(),
                scalars,
            ]
        ).astype(np.float32, copy=False)

    def _vector_v1(self) -> np.ndarray:
        rows, cols = PortableActionCodec.rows, PortableActionCodec.cols
        plant_types = np.zeros((rows, cols), dtype=np.float32)
        plant_health = np.zeros((rows, cols), dtype=np.float32)
        zombie_counts = np.zeros((rows, cols), dtype=np.float32)
        zombie_health = np.zeros((rows, cols), dtype=np.float32)

        for plant in self.payload.get("plants", []):
            row, col = int(plant["row"]), int(plant["col"])
            if 0 <= row < rows and 0 <= col < cols:
                plant_types[row, col] = max(float(plant["type"]), 0.0) / 52.0
                plant_health[row, col] += max(float(plant["health"]), 0.0) / 4000.0

        for zombie in self.payload.get("zombies", []):
            row = int(zombie["row"])
            col = int(np.clip((float(zombie["x"]) - 40.0) // 80.0, 0, cols - 1))
            if 0 <= row < rows:
                zombie_counts[row, col] += 1.0
                zombie_health[row, col] += (
                    max(float(zombie["body_health"]), 0.0)
                    + max(float(zombie["helm_health"]), 0.0)
                    + max(float(zombie["shield_health"]), 0.0)
                ) / 5000.0

        packets = np.zeros((PortableActionCodec.seed_slots, 3), dtype=np.float32)
        for packet in self.payload.get("seed_packets", []):
            slot = int(packet["slot"])
            if 0 <= slot < PortableActionCodec.seed_slots:
                packets[slot, 0] = max(float(packet["type"]), 0.0) / 52.0
                packets[slot, 1] = float(bool(packet["active"]))
                packets[slot, 2] = min(max(float(packet["refresh"]), 0.0) / 5000.0, 1.0)

        scalars = np.asarray(
            [
                min(max(float(self.payload.get("sun", 0.0)) / 1000.0, 0.0), 1.0),
                min(max(float(self.payload.get("wave", 0.0)) / 20.0, 0.0), 1.0),
                float(self.won),
                float(self.lost),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                plant_types.ravel(),
                np.clip(plant_health, 0.0, 1.0).ravel(),
                np.clip(zombie_counts / 8.0, 0.0, 1.0).ravel(),
                np.clip(zombie_health, 0.0, 1.0).ravel(),
                packets.ravel(),
                scalars,
            ]
        ).astype(np.float32, copy=False)


def parse_protocol_record(line: str) -> dict[str, Any] | None:
    if not line.startswith(PROTOCOL_PREFIX):
        return None
    value = json.loads(line[len(PROTOCOL_PREFIX) :])
    if not isinstance(value, dict):
        raise ValueError("Portable bridge protocol record must be a JSON object")
    return value


class PortableBridgeClient:
    def __init__(
        self,
        config: PortableProcessConfig,
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self.config = config
        environment = os.environ.copy()
        environment["SDL_AUDIODRIVER"] = "dummy"
        self.process = process_factory(
            config.command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Portable bridge process did not expose pipes")
        self._stdin: IO[str] = self.process.stdin
        self._stdout: IO[str] = self.process.stdout
        self._records: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._diagnostic_tail: deque[str] = deque(maxlen=256)
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            name="pvzwm-portable-bridge-reader",
            daemon=True,
        )
        self._reader_thread.start()
        ready = self._read_record(config.startup_timeout_seconds)
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"Portable bridge did not become ready: {ready}")
        if int(ready.get("action_count", -1)) != PortableActionCodec.size:
            self.close()
            raise RuntimeError("Portable bridge and Python action codecs disagree")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(f"Portable bridge exited with code {self.process.returncode}")
        self._stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._stdin.flush()
        response = self._read_record(self.config.response_timeout_seconds)
        if not response.get("ok", False):
            raise RuntimeError(f"Portable bridge request failed: {response}")
        return response

    def _read_record(self, timeout_seconds: float) -> dict[str, Any]:
        try:
            record = self._records.get(timeout=timeout_seconds)
        except queue.Empty as error:
            raise TimeoutError("Timed out waiting for the portable bridge") from error
        if isinstance(record, BaseException):
            raise RuntimeError(str(record)) from record
        return record

    def _read_stdout(self) -> None:
        try:
            for line in self._stdout:
                record = parse_protocol_record(line)
                if record is not None:
                    self._records.put(record)
                else:
                    self._diagnostic_tail.append(line.rstrip())
            diagnostics = "\n".join(self._diagnostic_tail)
            self._records.put(
                RuntimeError(
                    f"Portable bridge closed stdout with code {self.process.poll()}"
                    + (f"\n{diagnostics}" if diagnostics else "")
                )
            )
        except BaseException as error:
            self._records.put(error)

    def close(self) -> None:
        if getattr(self, "process", None) is None:
            return
        if self.process.poll() is None:
            try:
                self._stdin.write('{"cmd":"quit"}\n')
                self._stdin.flush()
                self.process.wait(timeout=5.0)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3.0)


class PortableEnv:
    def __init__(
        self,
        config: PortableProcessConfig,
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self.config = config
        self.codec = PortableActionCodec()
        self.client = PortableBridgeClient(config, process_factory)
        self._observation: PortableObservation | None = None
        self._legal_mask = np.zeros(self.codec.size, dtype=np.bool_)
        self._frame: np.ndarray | None = None
        self._step_count = 0

    def reset(
        self,
        *,
        level: int | None = None,
        seed: int | None = None,
        seed_types: tuple[int, ...] | None = None,
    ) -> tuple[PortableObservation, dict[str, object]]:
        level = self.config.level if level is None else level
        seed = 0 if seed is None else seed
        selected_seed_types = self.config.seed_types if seed_types is None else seed_types
        if selected_seed_types is not None:
            replace(self.config, seed_types=selected_seed_types)
        payload = self.client.request(
            {
                "cmd": "reset",
                "level": level,
                "seed": seed,
                "disable_mowers": int(self.config.disable_mowers),
                **(
                    {"seed_types": list(selected_seed_types)}
                    if selected_seed_types is not None
                    else {}
                ),
                **self._frame_request(),
            }
        )
        self._set_payload(payload)
        self._step_count = 0
        assert self._observation is not None
        return self._observation, {
            "source": "pvz_portable",
            "level": self._observation.level_id,
            "legal_action_mask": self._legal_mask.copy(),
            "mowers_ready": self._observation.mowers_ready,
            "mowers_triggered": self._observation.mowers_triggered,
            "mowers_disabled": self.config.disable_mowers,
            "seed_types": selected_seed_types,
        }

    def step(
        self, action: int | PortableAction
    ) -> tuple[PortableObservation, float, bool, bool, dict[str, object]]:
        action_id = self.codec.encode(action) if isinstance(action, PortableAction) else action
        if not 0 <= action_id < self.codec.size:
            raise ValueError(f"Action id {action_id} outside [0, {self.codec.size})")
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        if not self._legal_mask[action_id]:
            raise ValueError(f"Illegal portable action id: {action_id}")
        payload = self.client.request(
            {
                "cmd": "step",
                "action": action_id,
                "ticks": self.config.ticks_per_step,
                **self._frame_request(),
            }
        )
        self._set_payload(payload)
        self._step_count += 1
        assert self._observation is not None
        terminated = bool(payload.get("done", False))
        truncated = not terminated and self._step_count >= self.config.max_steps
        bridge_reward = float(payload.get("reward", 0.0))
        reward = bridge_reward
        if self.config.reward_mode == "clipped-damage":
            if self._observation.won:
                reward = 100.0 + max(0.0, bridge_reward - 100.0)
            elif self._observation.lost:
                reward = -100.0
            else:
                reward = max(0.0, bridge_reward)
        return self._observation, reward, terminated, truncated, {
            "source": "pvz_portable",
            "level": self._observation.level_id,
            "legal_action_mask": self._legal_mask.copy(),
            "bridge_reward": bridge_reward,
            "reward_mode": self.config.reward_mode,
            "mowers_ready": self._observation.mowers_ready,
            "mowers_triggered": self._observation.mowers_triggered,
            "mowers_disabled": self.config.disable_mowers,
        }

    def legal_action_mask(self) -> np.ndarray:
        return self._legal_mask.copy()

    def action_support_mask(self) -> np.ndarray:
        return np.ones(self.codec.size, dtype=np.bool_)

    def render(self) -> np.ndarray:
        if self._frame is None:
            raise RuntimeError(
                "No portable frame is available; reset the environment with frame capture enabled"
            )
        return self._frame.copy()

    def capture_frame(self, width: int = 800, height: int = 600) -> np.ndarray:
        """Render the current board without advancing simulation time."""
        if self._observation is None:
            raise RuntimeError("reset() must be called before capture_frame()")
        if width < 1 or height < 1 or width > 800 or height > 600:
            raise ValueError("Portable capture dimensions must be within 800x600")
        payload = self.client.request(
            {"cmd": "observe", "frame_width": width, "frame_height": height}
        )
        frame = PortableObservation(payload, self.config.vector_version).frame()
        if frame is None:
            raise RuntimeError("Portable observe response contains no frame")
        return frame

    def close(self) -> None:
        self.client.close()

    def _set_payload(self, payload: dict[str, Any]) -> None:
        self._observation = PortableObservation(payload, self.config.vector_version)
        self._frame = self._observation.frame()
        self._legal_mask.fill(False)
        legal = np.asarray(payload.get("legal_actions", []), dtype=np.int64)
        if legal.size and (legal.min() < 0 or legal.max() >= self.codec.size):
            raise RuntimeError("Portable bridge returned an out-of-range legal action")
        self._legal_mask[legal] = True

    def _frame_request(self) -> dict[str, int]:
        if self.config.frame_width == 0:
            return {}
        return {
            "frame_width": self.config.frame_width,
            "frame_height": self.config.frame_height,
        }


class PortableLoadoutCurriculum:
    """Train one observation-conditioned controller across a legal loadout set."""

    def __init__(
        self,
        config: PortableProcessConfig,
        loadouts: Sequence[Sequence[int]],
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        normalized = tuple(tuple(int(item) for item in loadout) for loadout in loadouts)
        if not normalized:
            raise ValueError("Loadout curriculum requires at least one loadout")
        expected_size = len(normalized[0])
        if any(len(loadout) != expected_size for loadout in normalized):
            raise ValueError("Every curriculum loadout must have equal capacity")
        for loadout in normalized:
            replace(config, seed_types=loadout)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Loadout curriculum entries must be unique")
        self.config = config
        self.codec = PortableActionCodec()
        self.loadouts = normalized
        self.episode_loadout = normalized[0]
        self._environment = PortableEnv(config, process_factory=process_factory)

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[PortableObservation, dict[str, object]]:
        rng = np.random.default_rng(seed)
        self.episode_loadout = self.loadouts[
            int(rng.integers(0, len(self.loadouts)))
        ]
        observation, info = self._environment.reset(
            seed=seed, seed_types=self.episode_loadout
        )
        return observation, {
            **info,
            "loadout_curriculum_size": len(self.loadouts),
            "loadout_curriculum_episode_loadout": self.episode_loadout,
        }

    def step(
        self, action: int | PortableAction
    ) -> tuple[PortableObservation, float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self._environment.step(action)
        return observation, reward, terminated, truncated, {
            **info,
            "loadout_curriculum_size": len(self.loadouts),
            "loadout_curriculum_episode_loadout": self.episode_loadout,
        }

    def legal_action_mask(self) -> np.ndarray:
        return self._environment.legal_action_mask()

    def action_support_mask(self) -> np.ndarray:
        return self._environment.action_support_mask()

    def render(self) -> np.ndarray:
        return self._environment.render()

    def close(self) -> None:
        self._environment.close()


class PortableSequentialCurriculum:
    """Keep replay on one Adventure stage until a win unlocks the next stage."""

    def __init__(
        self,
        config: PortableProcessConfig,
        final_level: int,
        rehearsal_probability: float = 0.0,
        completed_levels: Sequence[int] = (),
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        if not config.level <= final_level <= 50:
            raise ValueError("Curriculum end must be between its start level and 50")
        if not 0.0 <= rehearsal_probability <= 1.0:
            raise ValueError("Curriculum rehearsal probability must be in [0, 1]")
        normalized_completed = sorted(set(int(level) for level in completed_levels))
        if any(level < 1 or level >= config.level for level in normalized_completed):
            raise ValueError("Completed rehearsal levels must precede the frontier")
        self.config = config
        self.codec = PortableActionCodec()
        self.final_level = final_level
        self.current_level = config.level
        self.episode_level = config.level
        self.rehearsal_probability = rehearsal_probability
        self.completed_levels = normalized_completed
        self._environment = PortableEnv(config, process_factory=process_factory)

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[PortableObservation, dict[str, object]]:
        self.episode_level = self.current_level
        if self.completed_levels and self.rehearsal_probability > 0.0:
            rng = np.random.default_rng(seed)
            if rng.random() < self.rehearsal_probability:
                self.episode_level = int(rng.choice(self.completed_levels))
        observation, info = self._environment.reset(
            level=self.episode_level, seed=seed
        )
        return observation, self._curriculum_info(info)

    def step(
        self, action: int | PortableAction
    ) -> tuple[PortableObservation, float, bool, bool, dict[str, object]]:
        episode_level = self.episode_level
        observation, reward, terminated, truncated, info = self._environment.step(
            action
        )
        if (
            terminated
            and observation.won
            and episode_level == self.current_level
            and self.current_level < self.final_level
        ):
            self.completed_levels.append(self.current_level)
            self.current_level += 1
        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                **self._curriculum_info(info),
                "curriculum_episode_level_number": episode_level,
            },
        )

    def legal_action_mask(self) -> np.ndarray:
        return self._environment.legal_action_mask()

    def action_support_mask(self) -> np.ndarray:
        return self._environment.action_support_mask()

    def render(self) -> np.ndarray:
        return self._environment.render()

    def close(self) -> None:
        self._environment.close()

    def _curriculum_info(self, info: dict[str, object]) -> dict[str, object]:
        return {
            **info,
            "curriculum_level_number": self.current_level,
            "curriculum_episode_level_number": self.episode_level,
            "curriculum_final_level": self.final_level,
            "curriculum_completed_levels": tuple(self.completed_levels),
            "curriculum_rehearsal_probability": self.rehearsal_probability,
        }

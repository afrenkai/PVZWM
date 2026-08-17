from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from pvzwm.domain import GameConfig

ACTION_COUNT = 181
OBSERVATION_SIZE = 192


class _NativeConfig(ctypes.Structure):
    _fields_ = [
        ("tick_seconds", ctypes.c_float),
        ("max_time", ctypes.c_float),
        ("starting_sun", ctypes.c_float),
        ("passive_sun_interval", ctypes.c_float),
        ("spawn_interval", ctypes.c_float),
        ("final_wave_time", ctypes.c_float),
        ("max_zombies", ctypes.c_int32),
    ]


def _library_name() -> str:
    if platform.system() == "Darwin":
        return "libpvzwm_native.dylib"
    if platform.system() == "Windows":
        return "pvzwm_native.dll"
    return "libpvzwm_native.so"


def native_source_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "native"


def build_native_library(
    output_directory: str | Path,
    *,
    compiler: str | None = None,
    optimize: bool = True,
) -> Path:
    """Compile the dependency-free CPU simulator into a shared library."""

    source_directory = native_source_directory()
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / _library_name()
    selected_compiler = compiler or os.environ.get("CC") or shutil.which("cc")
    if selected_compiler is None:
        raise RuntimeError("No C compiler was found; set CC to a C11 compiler")
    command = [
        selected_compiler,
        "-std=c11",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-O3" if optimize else "-O0",
        "-dynamiclib" if platform.system() == "Darwin" else "-shared",
        str(source_directory / "pvz_sim.c"),
        "-o",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Native simulator compilation failed:\n"
            + (completed.stderr or completed.stdout)
        )
    return output


def find_native_library() -> Path:
    explicit = os.environ.get("PVZWM_NATIVE_LIBRARY")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(native_source_directory() / "build" / _library_name())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "The native simulator is not built. Run `make -C native` or set "
        "PVZWM_NATIVE_LIBRARY to the shared library path."
    )


def _as_native_config(config: GameConfig) -> _NativeConfig:
    if config.rows != 5 or config.cols != 9:
        raise ValueError("The native simulator currently requires a 5x9 lawn")
    return _NativeConfig(
        config.tick_seconds,
        config.max_time,
        config.starting_sun,
        config.passive_sun_interval,
        config.spawn_interval,
        config.final_wave_time,
        config.max_zombies,
    )


def _load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.pvz_action_count.argtypes = []
    library.pvz_action_count.restype = ctypes.c_int32
    library.pvz_observation_size.argtypes = []
    library.pvz_observation_size.restype = ctypes.c_int32
    library.pvz_batch_create.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(_NativeConfig),
    ]
    library.pvz_batch_create.restype = ctypes.c_void_p
    library.pvz_batch_destroy.argtypes = [ctypes.c_void_p]
    library.pvz_batch_destroy.restype = None
    library.pvz_batch_reset.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    library.pvz_batch_reset.restype = ctypes.c_int32
    library.pvz_batch_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    library.pvz_batch_step.restype = ctypes.c_int32
    return library


@dataclass(frozen=True, slots=True)
class NativeStep:
    observations: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray
    legal_action_masks: np.ndarray


class NativeBatchEnv:
    """Fixed-buffer vector environment backed by the C transition core.

    Returned arrays alias persistent storage and are overwritten by the next
    reset or step. This avoids a copy at the learner boundary.
    """

    def __init__(
        self,
        num_envs: int,
        config: GameConfig | None = None,
        *,
        library: str | Path | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.config = config or GameConfig()
        path = Path(library).resolve() if library is not None else find_native_library()
        self._library = _load_library(path)
        if self._library.pvz_action_count() != ACTION_COUNT:
            raise RuntimeError("Native action-space ABI mismatch")
        if self._library.pvz_observation_size() != OBSERVATION_SIZE:
            raise RuntimeError("Native observation ABI mismatch")
        native_config = _as_native_config(self.config)
        self._handle = self._library.pvz_batch_create(
            self.num_envs, ctypes.byref(native_config)
        )
        if not self._handle:
            raise RuntimeError("Could not allocate native environment batch")
        self.observations = np.empty(
            (self.num_envs, OBSERVATION_SIZE), dtype=np.float32
        )
        self.rewards = np.empty(self.num_envs, dtype=np.float32)
        self._terminals = np.empty(self.num_envs, dtype=np.uint8)
        self._truncations = np.empty(self.num_envs, dtype=np.uint8)
        self._legal_masks = np.empty(
            (self.num_envs, ACTION_COUNT), dtype=np.uint8
        )
        self._closed = False

    @staticmethod
    def _pointer(array: np.ndarray, ctype: object):
        return array.ctypes.data_as(ctypes.POINTER(ctype))

    @property
    def legal_action_masks(self) -> np.ndarray:
        return self._legal_masks.view(np.bool_)

    def reset(self, seeds: int | np.ndarray = 0) -> tuple[np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("NativeBatchEnv is closed")
        if np.isscalar(seeds):
            seed_array = np.arange(
                int(seeds), int(seeds) + self.num_envs, dtype=np.uint64
            )
        else:
            seed_array = np.ascontiguousarray(seeds, dtype=np.uint64)
            if seed_array.shape != (self.num_envs,):
                raise ValueError(f"Expected {self.num_envs} reset seeds")
        status = self._library.pvz_batch_reset(
            self._handle,
            self._pointer(seed_array, ctypes.c_uint64),
            self._pointer(self.observations, ctypes.c_float),
            self._pointer(self._legal_masks, ctypes.c_uint8),
        )
        if status != 0:
            raise RuntimeError(f"Native reset failed with status {status}")
        return self.observations, self.legal_action_masks

    def step(self, actions: np.ndarray) -> NativeStep:
        if self._closed:
            raise RuntimeError("NativeBatchEnv is closed")
        actions = np.ascontiguousarray(actions, dtype=np.int32)
        if actions.shape != (self.num_envs,):
            raise ValueError(f"Expected {self.num_envs} actions")
        status = self._library.pvz_batch_step(
            self._handle,
            self._pointer(actions, ctypes.c_int32),
            self._pointer(self.observations, ctypes.c_float),
            self._pointer(self.rewards, ctypes.c_float),
            self._pointer(self._terminals, ctypes.c_uint8),
            self._pointer(self._truncations, ctypes.c_uint8),
            self._pointer(self._legal_masks, ctypes.c_uint8),
        )
        if status > 0:
            raise ValueError(
                f"Illegal action or terminated episode in environment {status - 1}"
            )
        if status != 0:
            raise RuntimeError(f"Native step failed with status {status}")
        return NativeStep(
            self.observations,
            self.rewards,
            self._terminals.view(np.bool_),
            self._truncations.view(np.bool_),
            self.legal_action_masks,
        )

    def close(self) -> None:
        if not self._closed:
            self._library.pvz_batch_destroy(self._handle)
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_handle", None) and not getattr(self, "_closed", True):
            self.close()

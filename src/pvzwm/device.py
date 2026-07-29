from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    device: torch.device
    backend: str
    mixed_precision: bool
    autocast_dtype: torch.dtype | None = None


def select_device(preference: str = "auto") -> DeviceInfo:
    """Select MPS first on Apple Silicon, then CUDA, then CPU."""
    preference = preference.lower()
    if preference not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError(f"Unknown device preference: {preference}")

    if preference in {"auto", "mps"} and torch.backends.mps.is_available():
        return DeviceInfo(torch.device("mps"), "mps", False)
    if preference == "mps":
        raise RuntimeError("MPS was requested but is not available")
    if preference in {"auto", "cuda"} and torch.cuda.is_available():
        return DeviceInfo(torch.device("cuda"), "cuda", True)
    if preference == "cuda":
        raise RuntimeError("CUDA was requested but is not available")
    return DeviceInfo(torch.device("cpu"), "cpu", False)


def autocast_context(info: DeviceInfo, enabled: bool = True) -> ContextManager[object]:
    # Float16 training is still fragile for several MPS operations used here.
    if enabled and info.mixed_precision:
        return torch.autocast(
            device_type=info.backend,
            dtype=info.autocast_dtype or torch.float16,
        )
    return nullcontext()

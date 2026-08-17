from __future__ import annotations

import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pvzwm.goty_modes import GAME_MODE_BY_ID

V4_SAVE_MAGIC = b"PVZP_SAVE4"
V4_SAVE_VERSION = 1
V4_SAVE_HEADER = struct.Struct("<12sIII")

# Board::NeedSaveGame deliberately stores these persistent modes outside the
# mid-level board file.
MIDLEVEL_SAVE_EXCLUDED_MODES = frozenset({42, 43, 50})
MIDLEVEL_SAVEABLE_MODES = tuple(
    game_mode
    for game_mode in sorted(GAME_MODE_BY_ID)
    if game_mode not in MIDLEVEL_SAVE_EXCLUDED_MODES
)


@dataclass(frozen=True, slots=True)
class SaveTarget:
    label: str
    game_mode: int
    level: int | None = None

    @property
    def reset_key(self) -> str:
        return "level" if self.game_mode == 0 else "game_mode"

    @property
    def reset_value(self) -> int:
        return self.level if self.game_mode == 0 and self.level is not None else self.game_mode

    def reset_command(self, *, seed: int) -> dict[str, Any]:
        return {
            "cmd": "reset",
            self.reset_key: self.reset_value,
            "seed": seed,
            "frame_width": 0,
            "frame_height": 0,
        }

    def save_path(self, save_directory: Path, *, profile_id: int) -> Path:
        return save_directory / "userdata" / f"game{profile_id}_{self.game_mode}.v4"


def adventure_save_targets(levels: Iterable[int]) -> tuple[SaveTarget, ...]:
    targets: list[SaveTarget] = []
    for level in levels:
        if not 1 <= level <= 50:
            raise ValueError(f"Adventure save level must be in [1, 50], got {level}")
        world = (level - 1) // 10 + 1
        stage = (level - 1) % 10 + 1
        targets.append(SaveTarget(f"adventure-{world}-{stage}", 0, level))
    return tuple(targets)


def secondary_save_targets(game_modes: Iterable[int]) -> tuple[SaveTarget, ...]:
    targets: list[SaveTarget] = []
    for game_mode in game_modes:
        if game_mode not in GAME_MODE_BY_ID:
            raise ValueError(f"Unknown secondary game mode: {game_mode}")
        if game_mode in MIDLEVEL_SAVE_EXCLUDED_MODES:
            raise ValueError(
                f"{GAME_MODE_BY_ID[game_mode].name} does not use a mid-level board save"
            )
        targets.append(
            SaveTarget(
                f"mode-{game_mode}-{GAME_MODE_BY_ID[game_mode].name}",
                game_mode,
            )
        )
    return tuple(targets)


def validate_v4_save(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = path.read_bytes()
    except OSError as error:
        return [f"cannot read {path}: {error}"]

    if len(data) < V4_SAVE_HEADER.size:
        return [
            f"save is {len(data)} bytes; expected at least {V4_SAVE_HEADER.size}"
        ]

    magic, version, payload_size, payload_crc = V4_SAVE_HEADER.unpack_from(data)
    payload = data[V4_SAVE_HEADER.size :]
    if magic.rstrip(b"\0") != V4_SAVE_MAGIC:
        errors.append(f"magic: expected {V4_SAVE_MAGIC!r}, got {magic!r}")
    if version != V4_SAVE_VERSION:
        errors.append(f"version: expected {V4_SAVE_VERSION}, got {version}")
    if payload_size != len(payload):
        errors.append(
            f"payload size: header says {payload_size}, file contains {len(payload)}"
        )
    actual_crc = zlib.crc32(payload)
    if payload_crc != actual_crc:
        errors.append(
            f"payload CRC: header says {payload_crc:#010x}, got {actual_crc:#010x}"
        )
    return errors

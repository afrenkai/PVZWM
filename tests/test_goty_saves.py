from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from pvzwm.goty_saves import (
    MIDLEVEL_SAVE_EXCLUDED_MODES,
    MIDLEVEL_SAVEABLE_MODES,
    V4_SAVE_HEADER,
    V4_SAVE_MAGIC,
    V4_SAVE_VERSION,
    adventure_save_targets,
    secondary_save_targets,
    validate_v4_save,
)


def _save_bytes(payload: bytes = b"portable-save-payload") -> bytes:
    return V4_SAVE_HEADER.pack(
        V4_SAVE_MAGIC,
        V4_SAVE_VERSION,
        len(payload),
        zlib.crc32(payload),
    ) + payload


def test_save_target_manifest_covers_every_native_saveable_board() -> None:
    adventure = adventure_save_targets(range(1, 51))
    secondary = secondary_save_targets(MIDLEVEL_SAVEABLE_MODES)

    assert len(adventure) == 50
    assert [target.level for target in adventure] == list(range(1, 51))
    assert len(secondary) == 67
    assert {target.game_mode for target in secondary} == (
        set(range(1, 71)) - MIDLEVEL_SAVE_EXCLUDED_MODES
    )
    assert MIDLEVEL_SAVE_EXCLUDED_MODES == {42, 43, 50}


def test_save_targets_produce_native_bridge_commands_and_paths() -> None:
    adventure = adventure_save_targets((31,))[0]
    mode = secondary_save_targets((35,))[0]

    assert adventure.reset_command(seed=7) == {
        "cmd": "reset",
        "level": 31,
        "seed": 7,
        "frame_width": 0,
        "frame_height": 0,
    }
    assert mode.reset_command(seed=7)["game_mode"] == 35
    assert adventure.save_path(Path("/tmp/save"), profile_id=2) == Path(
        "/tmp/save/userdata/game2_0.v4"
    )
    assert mode.save_path(Path("/tmp/save"), profile_id=2) == Path(
        "/tmp/save/userdata/game2_35.v4"
    )


@pytest.mark.parametrize("game_mode", sorted(MIDLEVEL_SAVE_EXCLUDED_MODES))
def test_persistent_modes_reject_midlevel_board_save_targets(game_mode: int) -> None:
    with pytest.raises(ValueError, match="does not use a mid-level board save"):
        secondary_save_targets((game_mode,))


def test_validate_v4_save_accepts_matching_header_crc_and_size(tmp_path: Path) -> None:
    path = tmp_path / "game1_0.v4"
    path.write_bytes(_save_bytes())

    assert validate_v4_save(path) == []


def test_validate_v4_save_reports_corrupt_metadata(tmp_path: Path) -> None:
    path = tmp_path / "game1_0.v4"
    path.write_bytes(
        struct.pack("<12sIII", b"WRONG", 99, 999, 0) + b"payload"
    )

    errors = validate_v4_save(path)
    assert any(error.startswith("magic:") for error in errors)
    assert any(error.startswith("version:") for error in errors)
    assert any(error.startswith("payload size:") for error in errors)
    assert any(error.startswith("payload CRC:") for error in errors)

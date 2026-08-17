from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAY_SCRIPT = PROJECT_ROOT / "scripts" / "play_goty.sh"
FORK_SCRIPT = PROJECT_ROOT / "scripts" / "fork_goty_save.sh"
FORK_CHECKPOINT_SCRIPT = PROJECT_ROOT / "scripts" / "fork_teacher_checkpoint.sh"


def _fake_install(tmp_path: Path) -> tuple[Path, Path]:
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "main.pak").write_bytes(b"test")

    source = tmp_path / "source"
    build_name = "build-macos" if platform.system() == "Darwin" else "build"
    binary = source / build_name / "pvz-portable"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    binary.chmod(0o755)
    return resources, source


def _environment(
    tmp_path: Path,
    resources: Path,
    source: Path,
    **overrides: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PVZ_GOTY_RESOURCES": str(resources),
            "PVZ_PORTABLE_SOURCE_DIR": str(source),
            "PVZ_GOTY_SAVE_ROOT": str(tmp_path / "slots"),
        }
    )
    environment.update(overrides)
    return environment


def test_named_save_slot_is_windowed_and_isolated(tmp_path: Path) -> None:
    resources, source = _fake_install(tmp_path)
    result = subprocess.run(
        [str(PLAY_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(
            tmp_path,
            resources,
            source,
            PVZ_GOTY_SAVE_SLOT="world1-a",
        ),
    )

    slot = tmp_path / "slots" / "world1-a"
    assert slot.is_dir()
    assert f"Save slot: world1-a ({slot})" in result.stdout
    assert "-windowed" in result.stdout.splitlines()
    assert "-native-resolution" in result.stdout.splitlines()
    assert f"-savedir={slot}" in result.stdout.splitlines()


def test_fullscreen_slot_and_trailing_arguments(tmp_path: Path) -> None:
    resources, source = _fake_install(tmp_path)
    result = subprocess.run(
        [str(PLAY_SCRIPT), "-foo=bar"],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(
            tmp_path,
            resources,
            source,
            PVZ_GOTY_FULLSCREEN="1",
            PVZ_GOTY_SAVE_SLOT="world1-b",
        ),
    )

    assert "-fullscreen" in result.stdout.splitlines()
    assert "-native-resolution" not in result.stdout.splitlines()
    assert "-foo=bar" in result.stdout.splitlines()


def test_fixed_resolution_window_can_be_requested(tmp_path: Path) -> None:
    resources, source = _fake_install(tmp_path)
    result = subprocess.run(
        [str(PLAY_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(
            tmp_path,
            resources,
            source,
            PVZ_GOTY_NATIVE_RESOLUTION="0",
        ),
    )

    assert "-windowed" in result.stdout.splitlines()
    assert "-native-resolution" not in result.stdout.splitlines()


def test_teacher_play_records_segment_in_durable_campaign(tmp_path: Path) -> None:
    resources, source = _fake_install(tmp_path)
    teacher_root = tmp_path / "teacher-runs"
    result = subprocess.run(
        [str(PLAY_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=_environment(
            tmp_path,
            resources,
            source,
            PVZ_GOTY_SAVE_SLOT="world1-teacher",
            PVZ_GOTY_TEACHER="1",
            PVZ_GOTY_TEACHER_ROOT=str(teacher_root),
        ),
    )

    campaigns = list(teacher_root.iterdir())
    assert len(campaigns) == 1
    campaign = campaigns[0]
    segments = list((campaign / "segments").iterdir())
    assert len(segments) == 1
    segment = segments[0]
    assert f"Teacher campaign: {campaign}" in result.stdout
    assert f"Teacher segment: {segment}" in result.stdout
    assert f"-record={segment / 'teacher.dmo'}" in result.stdout.splitlines()
    assert f"-teacher-session-dir={campaign}" in result.stdout.splitlines()
    assert (campaign / "pvzwm-teacher-campaign.json").is_file()


def test_teacher_play_adds_segments_to_the_same_campaign(tmp_path: Path) -> None:
    resources, source = _fake_install(tmp_path)
    teacher_root = tmp_path / "teacher-runs"
    environment = _environment(
        tmp_path,
        resources,
        source,
        PVZ_GOTY_SAVE_SLOT="campaign",
        PVZ_GOTY_TEACHER="1",
        PVZ_GOTY_TEACHER_ROOT=str(teacher_root),
    )

    first = subprocess.run(
        [str(PLAY_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    second = subprocess.run(
        [str(PLAY_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    campaign = teacher_root / "campaign.campaign"
    segments = sorted((campaign / "segments").iterdir())
    assert len(segments) == 2
    assert segments[0].name.startswith("segment-0001.")
    assert segments[1].name.startswith("segment-0002.")
    assert f"Teacher campaign: {campaign}" in first.stdout
    assert f"Teacher campaign: {campaign}" in second.stdout


def test_teacher_play_requires_named_save_slot(tmp_path: Path) -> None:
    resources, source = _fake_install(tmp_path)
    result = subprocess.run(
        [str(PLAY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=_environment(
            tmp_path,
            resources,
            source,
            PVZ_GOTY_TEACHER="1",
            PVZ_GOTY_TEACHER_ROOT=str(tmp_path / "teacher-runs"),
        ),
    )

    assert result.returncode == 2
    assert "requires a named save slot" in result.stderr


def test_invalid_or_ambiguous_slot_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    resources, source = _fake_install(tmp_path)
    base_environment = _environment(tmp_path, resources, source)

    invalid_environment = base_environment | {"PVZ_GOTY_SAVE_SLOT": "../escape"}
    invalid = subprocess.run(
        [str(PLAY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=invalid_environment,
    )
    assert invalid.returncode == 2
    assert "Invalid save slot" in invalid.stderr

    ambiguous_environment = base_environment | {
        "PVZ_GOTY_SAVE_DIR": str(tmp_path / "direct"),
        "PVZ_GOTY_SAVE_SLOT": "world1-c",
    }
    ambiguous = subprocess.run(
        [str(PLAY_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=ambiguous_environment,
    )
    assert ambiguous.returncode == 2
    assert "not both" in ambiguous.stderr


def test_save_slots_can_be_listed_without_game_assets(tmp_path: Path) -> None:
    save_root = tmp_path / "slots"
    (save_root / "world1-b").mkdir(parents=True)
    (save_root / "world1-a").mkdir()
    environment = os.environ.copy()
    environment["PVZ_GOTY_SAVE_ROOT"] = str(save_root)

    result = subprocess.run(
        [str(PLAY_SCRIPT), "--list-save-slots"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.splitlines()[1:] == ["world1-a", "world1-b"]


def test_save_slot_can_be_forked_without_modifying_source(tmp_path: Path) -> None:
    save_root = tmp_path / "slots"
    source = save_root / "world1-main"
    source.mkdir(parents=True)
    (source / "user1.dat").write_bytes(b"checkpoint")
    environment = os.environ.copy()
    environment["PVZ_GOTY_SAVE_ROOT"] = str(save_root)

    result = subprocess.run(
        [str(FORK_SCRIPT), "world1-main", "before-1-5"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "Forked save slot" in result.stdout
    assert (source / "user1.dat").read_bytes() == b"checkpoint"
    assert (save_root / "before-1-5" / "user1.dat").read_bytes() == b"checkpoint"


def test_save_slot_fork_never_overwrites_destination(tmp_path: Path) -> None:
    save_root = tmp_path / "slots"
    (save_root / "source").mkdir(parents=True)
    destination = save_root / "destination"
    destination.mkdir()
    marker = destination / "keep"
    marker.write_text("untouched")
    environment = os.environ.copy()
    environment["PVZ_GOTY_SAVE_ROOT"] = str(save_root)

    result = subprocess.run(
        [str(FORK_SCRIPT), "source", "destination"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "already exists" in result.stderr
    assert marker.read_text() == "untouched"


def test_teacher_checkpoint_can_be_forked_to_new_save_slot(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "after-1-4"
    checkpoint.mkdir(parents=True)
    (checkpoint / "userdata").mkdir()
    (checkpoint / "userdata" / "user1.dat").write_bytes(b"after-1-4")
    (checkpoint / "pvzwm-checkpoint.json").write_text(
        '{"kind": "pvzwm-human-teacher-checkpoint"}'
    )
    save_root = tmp_path / "slots"
    environment = os.environ.copy()
    environment["PVZ_GOTY_SAVE_ROOT"] = str(save_root)

    result = subprocess.run(
        [str(FORK_CHECKPOINT_SCRIPT), str(checkpoint), "repro-1-5"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    destination = save_root / "repro-1-5"
    assert "Forked teacher checkpoint" in result.stdout
    assert (destination / "userdata" / "user1.dat").read_bytes() == b"after-1-4"
    assert not (destination / "pvzwm-checkpoint.json").exists()
    assert (checkpoint / "pvzwm-checkpoint.json").exists()

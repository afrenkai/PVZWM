from __future__ import annotations

import struct
from pathlib import Path

import pytest

from pvzwm.teacher_demo import (
    ADVENTURE_LEVEL_COUNT,
    DEMO_FILE_ID,
    DEMO_VERSION,
    PROFILE_VERSION,
    USER_VERSION,
    DemoMarker,
    DemoSummary,
    adventure_stage,
    read_checkpoint_profiles,
    read_demo_summary,
    validate_complete_adventure,
)


def _write_demo(path: Path, markers: list[tuple[str, int]]) -> None:
    marker_buffer = bytearray(struct.pack("<I", len(markers)))
    for name, update in markers:
        encoded_name = name.encode()
        marker_buffer.extend(struct.pack("<H", len(encoded_name)))
        marker_buffer.extend(encoded_name)
        marker_buffer.extend(struct.pack("<I", update))

    path.write_bytes(
        struct.pack(
            "<IIIQiH",
            DEMO_FILE_ID,
            DEMO_VERSION,
            1234,
            1_700_000_000,
            -18_000,
            0,
        )
        + struct.pack("<I", len(marker_buffer))
        + marker_buffer
        + struct.pack("<I", 4321)
        + b"\x01\x02\x03"
    )


def _write_checkpoint_profiles(
    checkpoint: Path,
    profiles: list[tuple[str, int, int, int]],
) -> None:
    userdata = checkpoint / "userdata"
    userdata.mkdir(parents=True)
    summaries = bytearray(struct.pack("<IH", PROFILE_VERSION, len(profiles)))
    for name, use_sequence, profile_id, adventure_level in profiles:
        encoded_name = name.encode()
        summaries.extend(struct.pack("<H", len(encoded_name)))
        summaries.extend(encoded_name)
        summaries.extend(struct.pack("<II", use_sequence, profile_id))
        (userdata / f"user{profile_id}.dat").write_bytes(
            struct.pack("<IIII", USER_VERSION, adventure_level, 250, 0)
            + struct.pack("<100I", *([0] * 100))
            + struct.pack("<80I", *([0] * 80))
            + b"details"
        )
    (userdata / "users.dat").write_bytes(summaries)


def test_demo_summary_reads_teacher_checkpoint_markers(tmp_path: Path) -> None:
    demo_path = tmp_path / "teacher.dmo"
    _write_demo(
        demo_path,
        [
            ("pvzwm/checkpoint/after-1-1", 1200),
            ("pvzwm/checkpoint/after-1-2", 4321),
        ],
    )

    summary = read_demo_summary(demo_path)

    assert summary.random_seed == 1234
    assert summary.update_count == 4321
    assert summary.command_bytes == 3
    assert [(marker.name, marker.update) for marker in summary.markers] == [
        ("pvzwm/checkpoint/after-1-1", 1200),
        ("pvzwm/checkpoint/after-1-2", 4321),
    ]


def test_demo_summary_rejects_truncated_data(tmp_path: Path) -> None:
    demo_path = tmp_path / "teacher.dmo"
    demo_path.write_bytes(struct.pack("<I", DEMO_FILE_ID))

    with pytest.raises(ValueError, match="Truncated"):
        read_demo_summary(demo_path)


def test_checkpoint_profiles_read_native_save_level(tmp_path: Path) -> None:
    checkpoint = tmp_path / "after-1-3"
    _write_checkpoint_profiles(
        checkpoint,
        [
            ("Player One", 7, 1, 4),
            ("Second", 4, 3, 12),
        ],
    )

    profiles = read_checkpoint_profiles(checkpoint)

    assert [
        (
            profile.name,
            profile.use_sequence,
            profile.profile_id,
            profile.user_version,
            profile.adventure_level,
            profile.coins,
            profile.finished_adventure,
            profile.packet_upgrades,
            profile.seed_slots,
            profile.can_show_almanac,
        )
        for profile in profiles
    ] == [
        ("Player One", 7, 1, USER_VERSION, 4, 250, 0, 0, 6, False),
        ("Second", 4, 3, USER_VERSION, 12, 250, 0, 0, 6, False),
    ]


def test_checkpoint_profiles_read_packet_upgrade_and_almanac_unlock(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "after-2-4"
    _write_checkpoint_profiles(checkpoint, [("Player", 1, 1, 15)])
    details = checkpoint / "userdata" / "user1.dat"
    payload = bytearray(details.read_bytes())
    struct.pack_into("<I", payload, 16 + 100 * 4 + 21 * 4, 1)
    details.write_bytes(payload)

    (profile,) = read_checkpoint_profiles(checkpoint)

    assert profile.packet_upgrades == 1
    assert profile.seed_slots == 7
    assert profile.can_show_almanac is True


def test_checkpoint_profiles_reject_truncated_details(tmp_path: Path) -> None:
    checkpoint = tmp_path / "after-1-1"
    _write_checkpoint_profiles(checkpoint, [("Player", 1, 1, 2)])
    (checkpoint / "userdata" / "user1.dat").write_bytes(
        struct.pack("<I", USER_VERSION)
    )

    with pytest.raises(ValueError, match="Truncated PvZ profile details"):
        read_checkpoint_profiles(checkpoint)


def test_checkpoint_profiles_reject_unknown_versions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "after-1-1"
    _write_checkpoint_profiles(checkpoint, [("Player", 1, 1, 2)])
    summary = checkpoint / "userdata" / "users.dat"
    summary.write_bytes(struct.pack("<I", PROFILE_VERSION + 1) + summary.read_bytes()[4:])

    with pytest.raises(ValueError, match="profile summary version"):
        read_checkpoint_profiles(checkpoint)


def _complete_campaign() -> tuple[list[dict[str, object]], DemoSummary]:
    checkpoints: list[dict[str, object]] = []
    markers: list[DemoMarker] = []
    for level in range(1, ADVENTURE_LEVEL_COUNT + 1):
        world, stage = adventure_stage(level)
        boundary = f"after-{world}-{stage}"
        marker = f"pvzwm/checkpoint/{boundary}"
        next_level = level + 1 if level < ADVENTURE_LEVEL_COUNT else 1
        finished_adventure = 1 if level == ADVENTURE_LEVEL_COUNT else 0
        checkpoints.append(
            {
                "schema_version": 1,
                "kind": "pvzwm-human-teacher-checkpoint",
                "completed_level": level,
                "world": world,
                "stage": stage,
                "next_adventure_level": next_level,
                "demo_marker": marker,
                "demo_update": level * 100,
                "path": f"/teacher/checkpoints/{boundary}",
                "save_profiles": [
                    {
                        "adventure_level": next_level,
                        "finished_adventure": finished_adventure,
                    }
                ],
            }
        )
        markers.append(DemoMarker(name=marker, update=level * 100))
    return checkpoints, DemoSummary(
        path="/teacher/segments/segment-0001/teacher.dmo",
        version=DEMO_VERSION,
        random_seed=1,
        start_time=1,
        timezone_offset=0,
        update_count=5001,
        command_bytes=1,
        markers=tuple(markers),
    )


def test_complete_adventure_requires_exact_first_campaign_sequence() -> None:
    checkpoints, demo = _complete_campaign()

    assert validate_complete_adventure(checkpoints, demo) == []


def test_complete_adventure_rejects_missing_special_stage_and_false_finish() -> None:
    checkpoints, demo = _complete_campaign()
    checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint["completed_level"] != 35
    ]
    final_profile = checkpoints[-1]["save_profiles"]
    assert isinstance(final_profile, list)
    assert isinstance(final_profile[0], dict)
    final_profile[0]["finished_adventure"] = 0

    errors = validate_complete_adventure(checkpoints, demo)

    assert any("4-5" in error for error in errors)
    assert any("finished_adventure=1" in error for error in errors)


def test_complete_adventure_accepts_ordered_segments() -> None:
    checkpoints, demo = _complete_campaign()
    first = DemoSummary(
        **{
            **demo.__dict__,
            "path": "/teacher/segments/segment-0001/teacher.dmo",
            "markers": demo.markers[:20],
        }
    )
    second = DemoSummary(
        **{
            **demo.__dict__,
            "path": "/teacher/segments/segment-0002/teacher.dmo",
            "markers": demo.markers[20:],
        }
    )

    assert validate_complete_adventure(checkpoints, (first, second)) == []

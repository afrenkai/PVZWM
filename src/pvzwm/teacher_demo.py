from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

DEMO_FILE_ID = 0x42BEEF78
DEMO_VERSION = 6
PROFILE_VERSION = 14
USER_VERSION = 12
ADVENTURE_LEVEL_COUNT = 50

# First-Adventure stages whose rules differ materially from an ordinary
# choose-your-seeds level. Completing the full sequence exercises every one.
SPECIAL_ADVENTURE_LEVELS = {
    5: "Wall-nut Bowling",
    10: "Day conveyor",
    15: "Whack a Zombie",
    20: "Night conveyor",
    25: "Big Trouble Little Zombie",
    30: "Pool conveyor",
    35: "Vasebreaker",
    40: "Fog conveyor / Stormy Night",
    45: "Bungee Blitz",
    50: "Dr. Zomboss",
}


@dataclass(frozen=True)
class DemoMarker:
    name: str
    update: int


@dataclass(frozen=True)
class DemoSummary:
    path: str
    version: int
    random_seed: int
    start_time: int
    timezone_offset: int
    update_count: int
    command_bytes: int
    markers: tuple[DemoMarker, ...]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["markers"] = [asdict(marker) for marker in self.markers]
        return result


@dataclass(frozen=True)
class SavedProfile:
    name: str
    use_sequence: int
    profile_id: int
    user_version: int
    adventure_level: int
    coins: int
    finished_adventure: int
    packet_upgrades: int
    seed_slots: int
    can_show_almanac: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_exact(
    stream: BinaryIO,
    size: int,
    *,
    description: str = "PvZ demo file",
) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"Truncated {description}")
    return value


def _unpack(
    stream: BinaryIO,
    format_string: str,
    *,
    description: str = "PvZ demo file",
) -> tuple[int, ...]:
    size = struct.calcsize(format_string)
    return struct.unpack(
        format_string,
        _read_exact(stream, size, description=description),
    )


def read_checkpoint_profiles(checkpoint: Path) -> tuple[SavedProfile, ...]:
    checkpoint = checkpoint.resolve()
    userdata = checkpoint / "userdata"
    summary_path = userdata / "users.dat"
    with summary_path.open("rb") as stream:
        (profile_version,) = _unpack(
            stream,
            "<I",
            description="PvZ profile summary",
        )
        if profile_version != PROFILE_VERSION:
            raise ValueError(
                "Unsupported PvZ profile summary version "
                f"{profile_version}; expected {PROFILE_VERSION}"
            )

        (profile_count,) = _unpack(
            stream,
            "<H",
            description="PvZ profile summary",
        )
        summaries: list[tuple[str, int, int]] = []
        for _ in range(profile_count):
            (name_size,) = _unpack(
                stream,
                "<H",
                description="PvZ profile summary",
            )
            name_bytes = _read_exact(
                stream,
                name_size,
                description="PvZ profile summary",
            )
            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("Invalid UTF-8 in PvZ profile name") from error
            use_sequence, profile_id = _unpack(
                stream,
                "<II",
                description="PvZ profile summary",
            )
            summaries.append((name, use_sequence, profile_id))
        if stream.read(1):
            raise ValueError("Unexpected trailing data in PvZ profile summary")

    profiles: list[SavedProfile] = []
    for name, use_sequence, profile_id in summaries:
        details_path = userdata / f"user{profile_id}.dat"
        with details_path.open("rb") as stream:
            user_version, adventure_level, coins, finished_adventure = _unpack(
                stream,
                "<IIII",
                description=f"PvZ profile details for {name!r}",
            )
            _read_exact(
                stream,
                100 * 4,
                description=f"PvZ challenge records for {name!r}",
            )
            purchases = _unpack(
                stream,
                "<80I",
                description=f"PvZ purchases for {name!r}",
            )
        if user_version != USER_VERSION:
            raise ValueError(
                f"Unsupported PvZ user version {user_version} for {name!r}; "
                f"expected {USER_VERSION}"
            )
        profiles.append(
            SavedProfile(
                name=name,
                use_sequence=use_sequence,
                profile_id=profile_id,
                user_version=user_version,
                adventure_level=adventure_level,
                coins=coins,
                finished_adventure=finished_adventure,
                packet_upgrades=purchases[21],
                seed_slots=6 + purchases[21],
                can_show_almanac=bool(finished_adventure or adventure_level >= 15),
            )
        )
    return tuple(profiles)


def adventure_stage(level: int) -> tuple[int, int]:
    if not 1 <= level <= ADVENTURE_LEVEL_COUNT:
        raise ValueError(f"Adventure level must be in 1..{ADVENTURE_LEVEL_COUNT}")
    return (level - 1) // 10 + 1, (level - 1) % 10 + 1


def validate_complete_adventure(
    checkpoints: list[dict[str, object]],
    demos: DemoSummary | Sequence[DemoSummary],
) -> list[str]:
    """Validate an exact, clean-profile First Adventure checkpoint sequence."""
    errors: list[str] = []
    by_level: dict[int, dict[str, object]] = {}
    for checkpoint in checkpoints:
        completed = checkpoint.get("completed_level")
        if not isinstance(completed, int) or isinstance(completed, bool):
            errors.append("Campaign checkpoint has no integer completed_level")
            continue
        if not 1 <= completed <= ADVENTURE_LEVEL_COUNT:
            errors.append(f"Campaign checkpoint level is outside 1..50: {completed}")
            continue
        if completed in by_level:
            errors.append(f"Duplicate campaign checkpoint for level {completed}")
            continue
        by_level[completed] = checkpoint

    expected_levels = set(range(1, ADVENTURE_LEVEL_COUNT + 1))
    missing = sorted(expected_levels - by_level.keys())
    if missing:
        errors.append(
            "Missing campaign checkpoints: "
            + ", ".join(f"{adventure_stage(level)[0]}-{adventure_stage(level)[1]}" for level in missing)
        )

    for level in sorted(expected_levels & by_level.keys()):
        checkpoint = by_level[level]
        world, stage = adventure_stage(level)
        boundary = f"after-{world}-{stage}"
        expected_marker = f"pvzwm/checkpoint/{boundary}"
        expected_next = level + 1 if level < ADVENTURE_LEVEL_COUNT else 1

        path = checkpoint.get("path")
        if not isinstance(path, str) or Path(path).name != boundary:
            errors.append(f"Level {world}-{stage} checkpoint directory is not {boundary}")
        if checkpoint.get("schema_version") != 1:
            errors.append(f"Level {world}-{stage} has an unsupported checkpoint schema")
        if checkpoint.get("kind") != "pvzwm-human-teacher-checkpoint":
            errors.append(f"Level {world}-{stage} is not a human teacher checkpoint")
        if (checkpoint.get("world"), checkpoint.get("stage")) != (world, stage):
            errors.append(f"Level {world}-{stage} has inconsistent world/stage metadata")
        if checkpoint.get("next_adventure_level") != expected_next:
            errors.append(
                f"Level {world}-{stage} should save Adventure level {expected_next}, "
                f"found {checkpoint.get('next_adventure_level')}"
            )
        if checkpoint.get("demo_marker") != expected_marker:
            errors.append(f"Level {world}-{stage} has an inconsistent demo marker")

        profiles = checkpoint.get("save_profiles")
        if not isinstance(profiles, list):
            profiles = []
        matching_profiles = [
            profile
            for profile in profiles
            if isinstance(profile, dict)
            and profile.get("adventure_level") == expected_next
        ]
        expected_finished = 1 if level == ADVENTURE_LEVEL_COUNT else 0
        if not any(
            profile.get("finished_adventure") == expected_finished
            for profile in matching_profiles
        ):
            errors.append(
                f"Level {world}-{stage} has no profile at Adventure level "
                f"{expected_next} with finished_adventure={expected_finished}"
            )

    expected_markers = [
        f"pvzwm/checkpoint/after-{world}-{stage}"
        for level in range(1, ADVENTURE_LEVEL_COUNT + 1)
        for world, stage in (adventure_stage(level),)
    ]
    if isinstance(demos, DemoSummary):
        demos = (demos,)
    teacher_markers = [
        marker
        for demo in demos
        for marker in demo.markers
        if marker.name.startswith("pvzwm/checkpoint/")
    ]
    if [marker.name for marker in teacher_markers] != expected_markers:
        errors.append("Demo does not contain the exact ordered 1-1 through 5-10 checkpoint sequence")
    if any(
        current.update <= previous.update
        for demo in demos
        for previous, current in zip(demo.markers, demo.markers[1:])
    ):
        errors.append("Campaign checkpoint demo updates are not strictly increasing")

    return errors


def read_demo_summary(path: Path) -> DemoSummary:
    path = path.resolve()
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        file_id, version, random_seed = _unpack(stream, "<III")
        if file_id != DEMO_FILE_ID:
            raise ValueError(f"Not a PvZ demo file: {path}")
        if version != DEMO_VERSION:
            raise ValueError(
                f"Unsupported PvZ demo version {version}; expected {DEMO_VERSION}"
            )

        (start_time,) = _unpack(stream, "<Q")
        (timezone_offset,) = _unpack(stream, "<i")
        (product_version_size,) = _unpack(stream, "<H")
        _read_exact(stream, product_version_size)

        (marker_buffer_size,) = _unpack(stream, "<I")
        marker_buffer = memoryview(_read_exact(stream, marker_buffer_size))
        marker_offset = 0

        def read_marker_value(format_string: str) -> tuple[int, ...]:
            nonlocal marker_offset
            size = struct.calcsize(format_string)
            if marker_offset + size > len(marker_buffer):
                raise ValueError("Truncated PvZ demo marker list")
            values = struct.unpack_from(format_string, marker_buffer, marker_offset)
            marker_offset += size
            return values

        (marker_count,) = read_marker_value("<I")
        markers: list[DemoMarker] = []
        for _ in range(marker_count):
            (name_size,) = read_marker_value("<H")
            if marker_offset + name_size > len(marker_buffer):
                raise ValueError("Truncated PvZ demo marker name")
            name = bytes(marker_buffer[marker_offset : marker_offset + name_size]).decode(
                "utf-8"
            )
            marker_offset += name_size
            (update,) = read_marker_value("<I")
            markers.append(DemoMarker(name=name, update=update))
        if marker_offset != len(marker_buffer):
            raise ValueError("Unexpected trailing data in PvZ demo marker list")

        (update_count,) = _unpack(stream, "<I")
        command_bytes = file_size - stream.tell()
        if command_bytes <= 0:
            raise ValueError("PvZ demo has no recorded commands")

    return DemoSummary(
        path=str(path),
        version=version,
        random_seed=random_seed,
        start_time=start_time,
        timezone_offset=timezone_offset,
        update_count=update_count,
        command_bytes=command_bytes,
        markers=tuple(markers),
    )

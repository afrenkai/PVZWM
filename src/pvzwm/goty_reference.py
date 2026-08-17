from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pvzwm.campaign import campaign_stages
from pvzwm.goty_modes import SECONDARY_GAME_MODES

REFERENCE_SCHEMA_VERSION = 1
ACCEPTANCE_SCHEMA_VERSION = 1
ACCEPTANCE_RESULTS = frozenset({"pending", "match", "mismatch", "blocked"})

_FLOW_SCENARIOS = (
    ("title", "Title and startup flow"),
    ("profile-create", "Profile creation"),
    ("profile-select-delete", "Profile selection and deletion"),
    ("adventure-launch", "Adventure launch and Crazy Dave intro"),
    ("mode-selector", "Mini-game, Puzzle, and Survival selectors"),
    ("almanac", "Almanac"),
    ("shop", "Crazy Dave's Twiddydinkies"),
    ("achievements", "Achievements"),
    ("options", "Options and volume controls"),
    ("help", "Help and controls"),
    ("zombatar", "Zombatar editor and persistence"),
    ("credits", "Credits"),
    ("pause-restart-quit", "Pause, restart, and quit"),
    ("save-reload", "Profile and mid-level save/reload"),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _content_id(files: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_name(path: str) -> str:
    return "".join(character for character in Path(path).name.casefold() if character.isalnum())


def _is_game_executable(relative_path: str) -> bool:
    normalized = _normalized_name(relative_path)
    if normalized in {"plantsvszombiesexe", "plantsvszombiesgotyexe"}:
        return True
    return (
        "/contents/macos/" in f"/{relative_path.casefold()}"
        and normalized in {"plantsvszombies", "plantsvszombiesgoty"}
    )


def build_reference_manifest(
    root: str | Path,
    *,
    version: str | None = None,
    distribution: str = "unknown",
    ignored_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Hash a legally owned GOTY installation without retaining its contents."""

    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError(f"reference root is not a directory: {source_root}")

    ignored = {Path(path).expanduser().resolve() for path in ignored_paths}
    file_records: list[dict[str, object]] = []
    for directory, directory_names, file_names in os.walk(
        source_root,
        followlinks=False,
    ):
        current = Path(directory)
        symlink_directories = [
            name for name in directory_names if (current / name).is_symlink()
        ]
        if symlink_directories:
            relative = (current / symlink_directories[0]).relative_to(source_root)
            raise ValueError(f"reference root contains a symbolic link: {relative}")
        for name in file_names:
            path = current / name
            if path.resolve() in ignored:
                continue
            relative = path.relative_to(source_root).as_posix()
            if path.is_symlink():
                raise ValueError(f"reference root contains a symbolic link: {relative}")
            if not path.is_file():
                continue
            file_records.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    file_records.sort(key=lambda item: str(item["path"]))
    if not file_records:
        raise ValueError("reference root contains no regular files")

    main_paks = [
        str(item["path"])
        for item in file_records
        if Path(str(item["path"])).name.casefold() == "main.pak"
    ]
    executables = [
        str(item["path"])
        for item in file_records
        if _is_game_executable(str(item["path"]))
    ]
    declared_version = version.strip() if version is not None else None
    if declared_version == "":
        declared_version = None
    ready = bool(main_paks and executables and declared_version)
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "kind": "pvz-goty-reference",
        "captured_at": _utc_now(),
        "distribution": distribution,
        "declared_version": declared_version,
        "version_source": "user-declared" if declared_version else None,
        "baseline_kind": "installation" if executables else "assets-only",
        "content_id": _content_id(file_records),
        "file_count": len(file_records),
        "total_bytes": sum(int(item["size"]) for item in file_records),
        "main_paks": main_paks,
        "game_executables": executables,
        "ready_for_black_box_comparison": ready,
        "files": file_records,
    }


def build_acceptance_plan(reference_manifest: dict[str, Any]) -> dict[str, Any]:
    if reference_manifest.get("kind") != "pvz-goty-reference":
        raise ValueError("not a GOTY reference manifest")
    content_id = reference_manifest.get("content_id")
    if not isinstance(content_id, str) or len(content_id) != 64:
        raise ValueError("reference manifest has no valid content_id")

    scenarios: list[dict[str, object]] = []

    def append(
        scenario_id: str,
        family: str,
        name: str,
        *,
        game_mode: int | None = None,
    ) -> None:
        scenarios.append(
            {
                "id": scenario_id,
                "family": family,
                "name": name,
                "game_mode": game_mode,
                "result": "pending",
                "reference_artifacts": [],
                "candidate_artifacts": [],
                "notes": "",
            }
        )

    for slug, name in _FLOW_SCENARIOS:
        append(f"flow/{slug}", "flow", name)
    for stage in campaign_stages():
        append(
            f"adventure/{stage.stage_id}",
            "adventure",
            f"Adventure {stage.stage_id}",
            game_mode=0,
        )
    for spec in SECONDARY_GAME_MODES:
        append(
            f"mode/{spec.game_mode:02d}-{spec.name}",
            spec.family,
            spec.name,
            game_mode=spec.game_mode,
        )

    return {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "kind": "pvz-goty-acceptance",
        "created_at": _utc_now(),
        "reference_content_id": content_id,
        "reference_version": reference_manifest.get("declared_version"),
        "candidate": {
            "vendor_revision": "e2627de8446dfd573bce3046d41435911fb488aa",
            "do_fix_bugs": False,
        },
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def record_acceptance_result(
    plan: dict[str, Any],
    scenario_id: str,
    result: str,
    *,
    reference_artifacts: list[object] | None = None,
    candidate_artifacts: list[object] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if result not in ACCEPTANCE_RESULTS:
        raise ValueError(f"unsupported acceptance result: {result}")
    updated = copy.deepcopy(plan)
    scenarios = updated.get("scenarios")
    if not isinstance(scenarios, list):
        raise TypeError("acceptance plan has no scenarios")
    matches = [
        scenario
        for scenario in scenarios
        if isinstance(scenario, dict) and scenario.get("id") == scenario_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate scenario id: {scenario_id}")
    scenario = matches[0]
    scenario["result"] = result
    if reference_artifacts is not None:
        scenario["reference_artifacts"] = list(reference_artifacts)
    if candidate_artifacts is not None:
        scenario["candidate_artifacts"] = list(candidate_artifacts)
    if notes is not None:
        scenario["notes"] = notes
    scenario["updated_at"] = _utc_now()
    return updated


def build_evidence_record(
    path: str | Path,
    *,
    evidence_root: str | Path,
) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    root = Path(evidence_root).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"evidence is not a regular file: {source}")
    try:
        relative = source.relative_to(root)
    except ValueError as error:
        raise ValueError(f"evidence must be stored beneath {root}") from error
    return {
        "path": relative.as_posix(),
        "size": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)

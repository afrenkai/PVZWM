#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
resources="${PVZ_GOTY_RESOURCES:-$project_root/data/pvz-goty}"
source_dir="${PVZ_PORTABLE_SOURCE_DIR:-$project_root/vendor/PvZ-Portable}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  build_dir="${PVZ_PORTABLE_BUILD_DIR:-$source_dir/build-macos}"
  save_root="${PVZ_GOTY_SAVE_ROOT:-${HOME}/Library/Application Support/PVZWM/save-slots}"
  teacher_root="${PVZ_GOTY_TEACHER_ROOT:-${HOME}/Library/Application Support/PVZWM/teacher-runs}"
else
  build_dir="${PVZ_PORTABLE_BUILD_DIR:-$source_dir/build}"
  save_root="${PVZ_GOTY_SAVE_ROOT:-${XDG_DATA_HOME:-${HOME}/.local/share}/pvzwm/save-slots}"
  teacher_root="${PVZ_GOTY_TEACHER_ROOT:-${XDG_DATA_HOME:-${HOME}/.local/share}/pvzwm/teacher-runs}"
fi
binary="$build_dir/pvz-portable"

if [[ "${1:-}" == "--list-save-slots" ]]; then
  if [[ ! -d "$save_root" ]]; then
    echo "No named save slots in $save_root"
    exit 0
  fi
  echo "Named save slots in $save_root:"
  find "$save_root" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | LC_ALL=C sort
  exit 0
fi

if [[ ! -f "$resources/main.pak" ]]; then
  echo "Missing legally purchased GOTY data: $resources/main.pak" >&2
  exit 2
fi
if [[ ! -x "$binary" ]]; then
  echo "Playable binary not found; building it first." >&2
  PVZ_PORTABLE_SOURCE_DIR="$source_dir" \
    PVZ_PORTABLE_BUILD_DIR="$build_dir" \
    "$project_root/scripts/setup_portable_backend.sh"
fi

arguments=("-resdir=$resources")
if [[ "${PVZ_GOTY_FULLSCREEN:-0}" == "1" ]]; then
  arguments+=("-fullscreen")
else
  arguments+=("-windowed")
  if [[ "${PVZ_GOTY_NATIVE_RESOLUTION:-1}" == "1" ]]; then
    arguments+=("-native-resolution")
  fi
fi

save_directory="${PVZ_GOTY_SAVE_DIR:-}"
save_slot="${PVZ_GOTY_SAVE_SLOT:-}"
if [[ -n "$save_directory" && -n "$save_slot" ]]; then
  echo "Set PVZ_GOTY_SAVE_DIR or PVZ_GOTY_SAVE_SLOT, not both." >&2
  exit 2
fi
if [[ -n "$save_slot" ]]; then
  if [[ ! "$save_slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
    echo "Invalid save slot '$save_slot'; use 1-64 letters, digits, '.', '_' or '-'." >&2
    exit 2
  fi
  save_directory="$save_root/$save_slot"
fi
if [[ -n "$save_directory" ]]; then
  mkdir -p "$save_directory"
  arguments+=("-savedir=$save_directory")
  if [[ -n "$save_slot" ]]; then
    echo "Save slot: $save_slot ($save_directory)"
  fi
fi

if [[ "${PVZ_GOTY_TEACHER:-0}" == "1" ]]; then
  if [[ -z "$save_slot" ]]; then
    echo "Teacher play requires a named save slot." >&2
    exit 2
  fi
  mkdir -p "$teacher_root"
  teacher_campaign="${PVZ_GOTY_TEACHER_CAMPAIGN_DIR:-$teacher_root/${save_slot}.campaign}"
  if [[ -e "$teacher_campaign" && ! -d "$teacher_campaign" ]]; then
    echo "Teacher campaign path is not a directory: $teacher_campaign" >&2
    exit 2
  fi
  mkdir -p "$teacher_campaign/segments"
  campaign_manifest="$teacher_campaign/pvzwm-teacher-campaign.json"
  if [[ ! -e "$campaign_manifest" ]]; then
    temporary_manifest="$teacher_campaign/.pvzwm-teacher-campaign.json.$$"
    printf '{\n  "schema_version": 1,\n  "kind": "pvzwm-human-teacher-campaign",\n  "save_slot": "%s"\n}\n' \
      "$save_slot" > "$temporary_manifest"
    mv "$temporary_manifest" "$campaign_manifest"
  fi
  segment_count="$(find "$teacher_campaign/segments" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  segment_number=$((segment_count + 1))
  printf -v segment_label '%04d' "$segment_number"
  teacher_segment="$(mktemp -d "$teacher_campaign/segments/segment-${segment_label}.XXXXXX")"
  mkdir "$teacher_segment/start-save"
  cp -R "$save_directory/." "$teacher_segment/start-save"
  arguments+=("-record=$teacher_segment/teacher.dmo")
  arguments+=("-teacher-session-dir=$teacher_campaign")
  echo "Teacher campaign: $teacher_campaign"
  echo "Teacher segment: $teacher_segment"
  echo "Use SAVE CHECKPOINT on each level reward screen."
fi

exec "$binary" "${arguments[@]}" "$@"

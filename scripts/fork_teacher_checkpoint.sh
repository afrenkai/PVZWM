#!/usr/bin/env bash
set -euo pipefail

checkpoint_directory="${1:-}"
destination_slot="${2:-}"

if [[ -z "$checkpoint_directory" || -z "$destination_slot" ]]; then
  echo "Usage: $0 CHECKPOINT_DIRECTORY DESTINATION_SLOT" >&2
  exit 2
fi
if [[ ! "$destination_slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "Invalid save slot '$destination_slot'; use 1-64 letters, digits, '.', '_' or '-'." >&2
  exit 2
fi
if [[ ! -d "$checkpoint_directory" ]]; then
  echo "Checkpoint directory does not exist: $checkpoint_directory" >&2
  exit 2
fi
manifest="$checkpoint_directory/pvzwm-checkpoint.json"
if [[ ! -f "$manifest" ]] || ! grep -q '"kind": "pvzwm-human-teacher-checkpoint"' "$manifest"; then
  echo "Not a PVZWM human teacher checkpoint: $checkpoint_directory" >&2
  exit 2
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  save_root="${PVZ_GOTY_SAVE_ROOT:-${HOME}/Library/Application Support/PVZWM/save-slots}"
else
  save_root="${PVZ_GOTY_SAVE_ROOT:-${XDG_DATA_HOME:-${HOME}/.local/share}/pvzwm/save-slots}"
fi

destination_directory="$save_root/$destination_slot"
if [[ -e "$destination_directory" ]]; then
  echo "Destination save slot already exists: $destination_slot" >&2
  exit 2
fi

mkdir -p "$save_root"
temporary_directory="$(mktemp -d "$save_root/.checkpoint.XXXXXX")"
cleanup() {
  if [[ -d "$temporary_directory" ]]; then
    rm -rf "$temporary_directory"
  fi
}
trap cleanup EXIT

cp -pR "$checkpoint_directory/." "$temporary_directory/"
rm -f "$temporary_directory/pvzwm-checkpoint.json"
mv -n "$temporary_directory" "$destination_directory"
if [[ -d "$temporary_directory" ]]; then
  echo "Destination save slot already exists: $destination_slot" >&2
  exit 2
fi
trap - EXIT
echo "Forked teacher checkpoint into save slot '$destination_slot'."

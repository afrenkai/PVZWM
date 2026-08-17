#!/usr/bin/env bash
set -euo pipefail

source_slot="${1:-}"
destination_slot="${2:-}"

validate_slot() {
  local slot="$1"
  if [[ ! "$slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
    echo "Invalid save slot '$slot'; use 1-64 letters, digits, '.', '_' or '-'." >&2
    exit 2
  fi
}

if [[ -z "$source_slot" || -z "$destination_slot" ]]; then
  echo "Usage: $0 SOURCE_SLOT DESTINATION_SLOT" >&2
  exit 2
fi
validate_slot "$source_slot"
validate_slot "$destination_slot"

if [[ "$(uname -s)" == "Darwin" ]]; then
  save_root="${PVZ_GOTY_SAVE_ROOT:-${HOME}/Library/Application Support/PVZWM/save-slots}"
else
  save_root="${PVZ_GOTY_SAVE_ROOT:-${XDG_DATA_HOME:-${HOME}/.local/share}/pvzwm/save-slots}"
fi

source_directory="$save_root/$source_slot"
destination_directory="$save_root/$destination_slot"
if [[ ! -d "$source_directory" ]]; then
  echo "Source save slot does not exist: $source_slot" >&2
  exit 2
fi
if [[ -e "$destination_directory" ]]; then
  echo "Destination save slot already exists: $destination_slot" >&2
  exit 2
fi

mkdir -p "$save_root"
temporary_directory="$(mktemp -d "$save_root/.fork.XXXXXX")"
cleanup() {
  if [[ -d "$temporary_directory" ]]; then
    rm -rf "$temporary_directory"
  fi
}
trap cleanup EXIT

cp -pR "$source_directory/." "$temporary_directory/"
mv -n "$temporary_directory" "$destination_directory"
if [[ -d "$temporary_directory" ]]; then
  echo "Destination save slot already exists: $destination_slot" >&2
  exit 2
fi
trap - EXIT
echo "Forked save slot '$source_slot' to '$destination_slot'."

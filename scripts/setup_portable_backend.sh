#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${PVZ_PORTABLE_SOURCE_DIR:-$project_root/vendor/PvZ-Portable}"
build_dir="${PVZ_PORTABLE_BUILD_DIR:-$source_dir/build}"
patch_file="$project_root/vendor/patches/pvz-portable-pvzwm-bridge.patch"

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --depth 1 https://github.com/wszqkzqk/PvZ-Portable.git "$source_dir"
fi

if git -C "$source_dir" apply --check "$patch_file"; then
  git -C "$source_dir" apply "$patch_file"
elif ! git -C "$source_dir" apply --reverse --check "$patch_file"; then
  echo "PvZ-Portable has changes that conflict with the PvZWM bridge patch." >&2
  exit 2
fi

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDO_FIX_BUGS=ON \
  -DPVZWM_BUILD_BRIDGE=ON
cmake --build "$build_dir" --target pvz-portable pvzwm-bridge

echo "Built $build_dir/pvz-portable"
echo "Built $build_dir/pvzwm-bridge"

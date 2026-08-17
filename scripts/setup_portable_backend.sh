#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${PVZ_PORTABLE_SOURCE_DIR:-$project_root/vendor/PvZ-Portable}"
portable_revision="${PVZ_PORTABLE_REVISION:-e2627de8446dfd573bce3046d41435911fb488aa}"
fix_bugs="${PVZ_PORTABLE_FIX_BUGS:-OFF}"
macos_metal="${PVZ_PORTABLE_MACOS_METAL:-ON}"
macos_archs="${PVZ_MACOS_ARCHS:-}"
macos_deployment_target="${PVZ_MACOS_DEPLOYMENT_TARGET:-12.0}"
build_metal_bridge="${PVZWM_BUILD_METAL_BRIDGE:-OFF}"
build_tests="${PVZWM_BUILD_TESTS:-OFF}"
if [[ -n "${PVZ_PORTABLE_BUILD_DIR:-}" ]]; then
  build_dir="$PVZ_PORTABLE_BUILD_DIR"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  build_dir="$source_dir/build-macos"
else
  build_dir="$source_dir/build"
fi
patch_file="$project_root/vendor/patches/pvz-portable-pvzwm-bridge.patch"

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/wszqkzqk/PvZ-Portable.git "$source_dir"
  git -C "$source_dir" fetch --depth 1 origin "$portable_revision"
  git -C "$source_dir" checkout --detach "$portable_revision"
fi

actual_revision="$(git -C "$source_dir" rev-parse HEAD)"
if [[ "$actual_revision" != "$portable_revision" ]]; then
  echo "PvZ-Portable revision mismatch." >&2
  echo "Expected: $portable_revision" >&2
  echo "Actual:   $actual_revision" >&2
  echo "Use a clean source directory or set PVZ_PORTABLE_REVISION explicitly." >&2
  exit 2
fi

if git -C "$source_dir" apply --check "$patch_file" 2>/dev/null; then
  git -C "$source_dir" apply "$patch_file"
elif ! git -C "$source_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
  echo "PvZ-Portable has changes that conflict with the PvZWM bridge patch." >&2
  exit 2
fi

cmake_platform_args=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  cmake_platform_args+=(
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=$macos_deployment_target"
  )
  if [[ -n "$macos_archs" ]]; then
    cmake_platform_args+=("-DCMAKE_OSX_ARCHITECTURES=$macos_archs")
  fi
fi

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDO_FIX_BUGS="$fix_bugs" \
  -DPVZ_MACOS_METAL="$macos_metal" \
  -DPVZWM_BUILD_BRIDGE=ON \
  -DPVZWM_BUILD_METAL_BRIDGE="$build_metal_bridge" \
  -DPVZWM_BUILD_TESTS="$build_tests" \
  "${cmake_platform_args[@]}"
build_targets=(pvz-portable pvzwm-bridge)
if [[ "$(uname -s)" == "Darwin" && "$build_metal_bridge" == "ON" ]]; then
  build_targets+=(pvzwm-metal-bridge)
fi
if [[ "$build_tests" == "ON" ]]; then
  build_targets+=(pvzwm-string-format-test pvzwm-openmpt-tempo-test)
fi
cmake --build "$build_dir" --target "${build_targets[@]}"

echo "Built $build_dir/pvz-portable"
echo "Built $build_dir/pvzwm-bridge"
if [[ "$(uname -s)" == "Darwin" && "$build_metal_bridge" == "ON" ]]; then
  echo "Built $build_dir/pvzwm-metal-bridge"
fi
if [[ "$build_tests" == "ON" ]]; then
  echo "Built $build_dir/pvzwm-string-format-test"
  echo "Built $build_dir/pvzwm-openmpt-tempo-test"
fi

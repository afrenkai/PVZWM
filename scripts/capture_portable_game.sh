#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY must be set (run this script with xvfb-run)." >&2
  exit 2
fi

game_binary="${1:-/home/ubuntu/PVZWM/vendor/PvZ-Portable/build/pvz-portable}"
resource_dir="${2:-/home/ubuntu/PVZWM/data/pvz-goty}"
output_path="${3:-/home/ubuntu/PVZWM/artifacts/pvz-portable-boot.png}"
startup_seconds="${PVZWM_CAPTURE_DELAY_SECONDS:-8}"
click_at_seconds="${PVZWM_CLICK_AT_SECONDS:-}"
post_click_seconds="${PVZWM_POST_CLICK_DELAY_SECONDS:-5}"
output_width="${PVZWM_CAPTURE_WIDTH:-800}"
output_height="${PVZWM_CAPTURE_HEIGHT:-600}"

mkdir -p "$(dirname "$output_path")"

env SDL_AUDIODRIVER=dummy "$game_binary" -resdir="$resource_dir" \
  >/tmp/pvzwm-portable-capture.log 2>&1 &
game_pid=$!

cleanup() {
  kill "$game_pid" 2>/dev/null || true
  wait "$game_pid" 2>/dev/null || true
}
trap cleanup EXIT

if [[ -n "$click_at_seconds" ]]; then
  sleep "$click_at_seconds"
  xdotool mousemove --sync 400 300 click 1
  sleep "$post_click_seconds"
else
  sleep "$startup_seconds"
fi

window_id="$(xdotool search --onlyvisible --pid "$game_pid" | tail -n 1)"
if [[ -z "$window_id" ]]; then
  echo "Could not find the PvZ window for process $game_pid." >&2
  exit 3
fi

window_x=0
window_y=0
window_width=0
window_height=0
while IFS='=' read -r key value; do
  case "$key" in
    X) window_x="$value" ;;
    Y) window_y="$value" ;;
    WIDTH) window_width="$value" ;;
    HEIGHT) window_height="$value" ;;
  esac
done < <(xdotool getwindowgeometry --shell "$window_id")

echo "window_id=$window_id x=$window_x y=$window_y width=$window_width height=$window_height"

# The game always draws an 800x600 logical canvas, but the SDL window is
# resizable and fullscreen uses the desktop drawable. Current PvZ-Portable
# preserves 4:3 with centered letterboxing. X11 window metadata is unreliable
# for SDL_FULLSCREEN_DESKTOP without a window manager, so derive the centered
# 4:3 presentation rectangle from the private display itself. Running Xvfb at
# 800x600 keeps this crop pixel-identical to the logical canvas.
read -r display_width display_height < <(xdotool getdisplaygeometry)
if (( display_width * 3 > display_height * 4 )); then
  capture_height="$display_height"
  capture_width=$((display_height * 4 / 3))
  capture_x=$(((display_width - capture_width) / 2))
  capture_y=0
elif (( display_width * 3 < display_height * 4 )); then
  capture_width="$display_width"
  capture_height=$((display_width * 3 / 4))
  capture_x=0
  capture_y=$(((display_height - capture_height) / 2))
else
  capture_width="$display_width"
  capture_height="$display_height"
  capture_x=0
  capture_y=0
fi

echo "display=${display_width}x${display_height} presentation_x=$capture_x presentation_y=$capture_y presentation_width=$capture_width presentation_height=$capture_height output=${output_width}x${output_height}"
ffmpeg -hide_banner -loglevel error -y \
  -video_size "${capture_width}x${capture_height}" -draw_mouse 0 -f x11grab \
  -i "${DISPLAY}+${capture_x},${capture_y}" \
  -vf "scale=${output_width}:${output_height}:flags=lanczos" \
  -frames:v 1 "$output_path"

test -s "$output_path"

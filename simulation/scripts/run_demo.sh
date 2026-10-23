#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
tweezers_run_dir="${1:-runs/demo_$(date -u +%Y%m%dT%H%M%S)}"
tweezers_gpu="${2:-0}"
tweezers_python="${ISAAC_PYTHON:-/cpfs/user/lvzhaoyang/apps/bin/isaac-python}"
"$tweezers_python" src/run_tweezers.py --out "$tweezers_run_dir" --gpu "$tweezers_gpu" --render
python scripts/validate_run.py "$tweezers_run_dir"
"$tweezers_python" scripts/compose_video.py "$tweezers_run_dir" --output-dir "$tweezers_run_dir/delivery"
"$tweezers_python" scripts/compose_video.py "$tweezers_run_dir" --output-dir "$tweezers_run_dir/closeup_delivery" --closeup-only --playback-speed .5 --width 1280

#!/usr/bin/env bash
set -euo pipefail
calibrated_bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$calibrated_bundle/env/setup_sim.sh"
calibrated_out="${1:-runs/xhand_calibrated_demo_$(date -u +%Y%m%dT%H%M%S_%N)}"
if (( $# )); then shift; fi
calibrated_handoff="$calibrated_bundle/calibration"
"$ISAAC_PYTHON" scripts/run_pick_place.py \
  --out "$calibrated_out" \
  --num-envs 1 --seed 0 --device cuda:0 \
  --evaluate "$calibrated_bundle/models/policies/xhand_pick_place_ppo.pt" \
  --pose configs/xhand_local.json \
  --snapshot "$calibrated_bundle/models/snapshots/xhand_stable_grasp.json" \
  --scene-layout configs/calibrated_demo_y_table.json \
  --cases configs/calibrated_demo_y_case.json \
  --xy-lower -.08 .48 --xy-upper .22 .76 \
  --retreat-closure .8 \
  --calibrated-rig "$calibrated_handoff/results/camera_rig_isaac.json" \
  --world-base "$calibrated_handoff/results/world_base_isaac.json" \
  "$@"
printf '\n输出目录：%s\n腕部视频：%s/wrist.mp4\n全局视频：%s/global.mp4\n' \
  "$calibrated_out" "$calibrated_out" "$calibrated_out"

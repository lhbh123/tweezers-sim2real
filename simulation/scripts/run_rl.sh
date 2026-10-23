#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mode="${1:-evaluate}"
device="${2:-cuda:0}"
run_dir="${3:-runs/rl_reproduce_${mode}}"
pose_path=configs/human_grasp_distal_suspend30_10.json
default_checkpoint=runs/rl_distal30_10_s5/iteration_00100.pt
task_parameters=(--closure-bias .75 --closure-scale 1.2 --clearance .002)
if [[ "$mode" == approach-* ]]; then
  pose_path=configs/human_grasp_distal_suspend30.json
  default_checkpoint=runs/rl_distal_open20_s6/iteration_00250.pt
  if [[ "$mode" == approach-train ]]; then
    default_checkpoint=runs/rl_distal30_10_s5/iteration_00075.pt
  fi
  task_parameters=(--closure-bias .6 --closure-scale .9 --clearance .02
    --contact-offset .0002 --aperture-reward .4 --wide-closeup
    --reset-snapshot runs/natural_hold_distal_open060/stable_snapshot.json)
fi
checkpoint="${4:-$default_checkpoint}"
common=(--out "$run_dir" --device "$device"
  --pose "$pose_path"
  --dt 0.0020833333333333333 --decimation 8
  --spring-damping .02 --tool-width .009 --bridge-width .012
  --solver-iterations 64 --solver-velocity-iterations 8
  "${task_parameters[@]}" --xy-range .002
  --self-collisions --filter-mechanical-self-pairs)
case "$mode" in
  train) extra=(--num-envs 128 --iterations 100 --seed 5) ;;
  approach-train) extra=(--num-envs 128 --iterations 175 --seed 6 --resume "$checkpoint") ;;
  evaluate|approach-evaluate) extra=(--num-envs 64 --seed 0 --evaluate "$checkpoint") ;;
  render|approach-render) extra=(--num-envs 1 --seed 0 --evaluate "$checkpoint" --render-video) ;;
  *) printf 'Usage: bash scripts/run_rl.sh [approach-]train|evaluate|render [cuda:N] [unique-output-dir] [checkpoint]\n' >&2; exit 2 ;;
esac
if [[ -e "$run_dir/launch.json" ]]; then
  printf 'Run directory already contains evidence: %s; select a new directory.\n' "$run_dir" >&2
  exit 2
fi
exec /cpfs/user/lvzhaoyang/apps/bin/isaac-python scripts/train_rl.py "${common[@]}" "${extra[@]}"

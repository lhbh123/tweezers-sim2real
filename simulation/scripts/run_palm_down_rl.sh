#!/usr/bin/env bash
set -euo pipefail
mode=${1:?train|evaluate|render|bc-train|bc-evaluate|bc-render}
device=${2:?physical device, for example cuda:6}
out=${3:?new run directory}
checkpoint=${4:-}
seed=${5:-91}
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
if [[ -e "$out" || -e "$out.log" ]]; then
  echo "Refusing to overwrite existing run evidence: $out" >&2
  exit 2
fi
args=(--pose configs/human_grasp_distal_suspend30.json --out "$out" --device "$device"
  --dt 0.0020833333333333333 --decimation 8 --spring 5 --spring-damping .02
  --tool-width .009 --bridge-width .012 --solver-iterations 64 --solver-velocity-iterations 8
  --contact-offset .0002 --clearance .02 --closure-min .6 --aperture-reward .4
  --require-palm-down --tool-angle-range 25 60 --seed "$seed")
if [[ "$mode" == bc-* ]]; then
  args+=(--closure-bias .6 --closure-scale 1.4 --closure-max 1.8
    --reset-snapshot runs/natural_orientation_spring5_open/table40_x-1_close0_snapshot.json)
else
  args+=(--closure-bias .8 --closure-scale .7 --closure-max 1.6
    --reset-snapshot runs/natural_orientation_spring5_sweep/table40_x-1_close0.2_snapshot.json)
fi
case "$mode" in
  train) args+=(--num-envs 128 --iterations 300) ;;
  bc-train) args+=(--num-envs 128 --iterations 200 --initial-std .15
    --initialize-policy "${checkpoint:-runs/palm_down_bc32/bc_policy.pt}") ;;
  evaluate|bc-evaluate) args+=(--num-envs 64 --evaluate "${checkpoint:?checkpoint required}") ;;
  render|bc-render) args+=(--num-envs 1 --evaluate "${checkpoint:?checkpoint required}"
    --render-video --wide-closeup --orientation-views --orthographic-top) ;;
  *) echo "Unsupported mode: $mode" >&2; exit 2 ;;
esac
exec /cpfs/user/lvzhaoyang/apps/bin/isaac-python scripts/train_rl.py "${args[@]}" >"$out.log" 2>&1

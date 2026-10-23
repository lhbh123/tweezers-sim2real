#!/usr/bin/env bash
# Reproduce the W01 natural posture v6d; the general runner keeps its XHand defaults.
set -euo pipefail
sharpa_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sharpa_python="${ISAAC_PYTHON:-/cpfs/user/lvzhaoyang/apps/bin/isaac-python}"
if [[ $# -eq 0 || ${1:-} == --help || ${1:-} == -h ]]; then
  cat <<'USAGE'
Usage: scripts/run_sharpa_w01.sh --out NEW_DIRECTORY [run_pick_place.py arguments]

Defaults: W01 natural posture v6d with its stable preheld v6c snapshot and --evaluate zero,
1 environment, open/closed/retreat closure .75/1/1, hand stiffness/damping
100/1, finger friction 1.4, tweezer spring 5, speed .04 m/s, max 45 seconds.
Release feedback adds up to .10 opening, confirms clear jaw force for .10 s,
and sets a nominal target of .01 m/s before acceleration smoothing.
This is not a hard withdrawal speed cap.
The W01 config uses a 20–85 degree working-angle range and zero extra
finite torsional friction patch. The middle finger supports the tool, yields
slightly during release, and returns after the tips clear the cube.
See README_SHARPA_W01_NATURAL.md for results and remaining failed cases.

All arguments are forwarded; explicit arguments override these defaults.
Useful options: --num-envs N --seed N --device cuda:N --cases FILE
               --camera-config FILE --tactile --tacmap --record-substeps
--tacmap runs the official SharpaTacmap geometry sensor; --tactile records
independent PhysX contact forces. Both require --num-envs 1.
The default pose selects CAD-derived sampling maps matched to current W01;
the official TacMap algorithm and original rigid-contact task remain unchanged.
Use --pose configs/sharpa_w01_natural_v6d.json for original-map comparison.
Always use a new output directory.
USAGE
  exit 0
fi
cd -- "$sharpa_root"
sharpa_defaults=(
  --pose configs/sharpa_w01_natural_v6d_tacmap_matched_v2.json
  --snapshot configs/sharpa_w01_natural_v6c_preheld.json
  --num-envs 1 --evaluate zero
  --open-closure .75 --closed-closure 1 --retreat-closure 1
  --release-extra-opening-max .10 --release-contact-clear-seconds .10
  --withdraw-initial-speed .01
  --hand-stiffness 100 --hand-damping 1
  --finger-friction 1.4 --spring 5
  --speed .04 --episode-seconds 45
)
python3 - "${sharpa_defaults[@]}" "$@" <<'PY'
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--pose',required=True);p.add_argument('--snapshot',required=True)
p.add_argument('--out',required=True);p.add_argument('--num-envs',type=int,default=1)
p.add_argument('--tactile',action='store_true')
p.add_argument('--tacmap',action='store_true')
a,_=p.parse_known_args()
pose=Path(a.pose);snapshot=Path(a.snapshot)
if not pose.is_file() or not snapshot.is_file():
    p.error('--pose and --snapshot must refer to existing files')
config=json.loads(pose.read_text());state=json.loads(snapshot.read_text())
if config.get('palm_link')!='right_hand_C_MC' or 'W01' not in config.get('model',''):
    p.error('--pose must identify the Sharpa W01 model and right_hand_C_MC palm frame')
if not set(config['q_reference']).issubset(state.get('joint_names',[])):
    p.error('--snapshot joint names do not match the selected W01 hand')
targets = dict(zip(state['joint_names'], state.get('q_target', [])))
fixed_mismatches = [name for name, reference in config['q_reference'].items()
                    if abs(config['closure_delta'].get(name, 0.0)) < 1e-9
                    and (name not in targets or abs(targets[name] - reference) > 1e-3)]
if fixed_mismatches:
    p.error('--snapshot fixed-finger targets do not match --pose: '
            + ', '.join(fixed_mismatches))
if a.num_envs<1 or ((a.tactile or a.tacmap) and a.num_envs!=1):
    p.error('--num-envs must be positive; tactile recording requires one environment')
if Path(a.out).exists():
    p.error('--out must be a new directory so existing experiment evidence is retained')
PY
exec "$sharpa_python" "$sharpa_root/scripts/run_pick_place.py" "${sharpa_defaults[@]}" "$@"

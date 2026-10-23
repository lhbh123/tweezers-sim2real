#!/usr/bin/env bash
# W01 compliant-contact candidate, with explicitly measured squeeze readiness.
set -euo pipefail
sharpa_compliant_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -eq 0 || ${1:-} == --help || ${1:-} == -h ]]; then
  cat <<'USAGE'
Usage: scripts/run_sharpa_w01_compliant.sh --out NEW_DIRECTORY [runner arguments]

Current W01 matched_v2 maps; skin compliant stiffness/damping 280/20,
contact/rest offset 0.2/0 mm; existing friction, torque limits and spring retained.
Uses a dynamically re-equilibrated preheld snapshot (does not pick up the tool).
Experimental: fixed case 1/1, matching eight-case replay 5/8 versus rigid 8/8.
This is not a replacement for the original default task.
Joint synergy: open/closed/retreat = 1.0/2.5/1.0, thumb IP stays above its limit.
Before lifting: both actual jaw/block forces >=0.0604 N for >=0.25 s,
plus the original minimum squeeze time. This is not a slip guarantee.

Record native cameras and independent tactile quantities:
  --cases configs/sharpa_w01_hard_case.json
  --camera-config configs/sharpa_w01_tacmap_cameras.json
  --tactile --tacmap --tacmap-diagnostics
Optional audits: --collision-geometry --contact-shapes
--record-substeps also records expensive grasp-geometry diagnostics; it is not
needed for the 480 Hz raw tactile contact data.
Native collider reports may be unavailable; zero reports do not mean zero force.

The official multi-target CPD can pair different tweezer blades and accept
origins outside both objects. Preserve the raw output; inspect the independent
per-body replay in sharpa_w01_tactile_validation/improvement_validation/scripts/replay_per_target_tacmap.py.

All arguments are forwarded and can override these defaults.
Raw TacMap is geometric depth encoding, not pressure or hardware calibration.
Previous rigid-contact entry point: scripts/run_sharpa_w01.sh
USAGE
  exit 0
fi
exec "$sharpa_compliant_root/scripts/run_sharpa_w01.sh" \
  --pose "$sharpa_compliant_root/configs/sharpa_w01_tacmap_compliant_v1.json" \
  --snapshot "$sharpa_compliant_root/configs/sharpa_w01_tacmap_compliant_preheld_v1.json" \
  --open-closure 1.0 --closed-closure 2.5 --retreat-closure 1.0 "$@"

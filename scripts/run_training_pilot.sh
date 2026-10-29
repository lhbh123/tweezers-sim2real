#!/usr/bin/env bash
set -euo pipefail
pilot_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$pilot_root/env/setup_sim.sh"
exec "$ISAAC_PYTHON" "$pilot_root/scripts/collect_training_pilot.py" "$@"

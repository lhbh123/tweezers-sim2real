#!/usr/bin/env bash
set -euo pipefail
rack_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$rack_project_root/env/setup_sim.sh"
exec "$ISAAC_PYTHON" -u scripts/train_rack_pickup.py "$@"

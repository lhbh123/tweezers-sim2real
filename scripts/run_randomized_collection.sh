#!/usr/bin/env bash
set -euo pipefail
collection_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$collection_root/env/setup_sim.sh"
exec "$ISAAC_PYTHON" "$collection_root/scripts/collect_randomized_grasps.py" "$@"

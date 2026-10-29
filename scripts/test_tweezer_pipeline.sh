#!/usr/bin/env bash
set -euo pipefail
pipeline_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$pipeline_root/env/setup_sim.sh"
exec "$ISAAC_PYTHON" -u scripts/test_tweezer_pipeline.py "$@"

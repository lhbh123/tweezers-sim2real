#!/usr/bin/env bash
set -euo pipefail
tweezers_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$tweezers_root/simulation/scripts/run_calibrated_demo.sh" "$@"

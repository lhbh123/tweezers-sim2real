#!/usr/bin/env bash
set -euo pipefail
tweezers_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$tweezers_root/env/local.env" ]]; then source "$tweezers_root/env/local.env"; fi
calibration_python="${CALIBRATION_PYTHON:-$tweezers_root/.venv-calibration/bin/python}"
if [[ ! -x "$calibration_python" ]]; then
    echo '请先按 README 创建 .venv-calibration，或设置 CALIBRATION_PYTHON。' >&2
    exit 1
fi
cd "$tweezers_root/calibration"
export PYTHONPATH="$tweezers_root/calibration/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$calibration_python" -m xhand_calib "$@"

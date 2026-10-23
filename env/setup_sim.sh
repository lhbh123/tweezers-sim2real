# Source from Bash. Project paths are computed from this file after relocation.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo '请使用 source env/setup_sim.sh' >&2
    exit 1
fi

export TWEEZERS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$TWEEZERS_ROOT/env/local.env" ]]; then
    source "$TWEEZERS_ROOT/env/local.env" || return 1
fi
export BUNDLE="$TWEEZERS_ROOT"
export PROJECT="$TWEEZERS_ROOT/simulation"
export XHAND_SIM_ENV="${XHAND_SIM_ENV:-/mnt/workspace/tweezers-software/xhand-sim}"
export ISAAC_PYTHON="${ISAAC_PYTHON:-$XHAND_SIM_ENV/bin/python}"
export XHAND_DRIVER_DIR="${XHAND_DRIVER_DIR:-/mnt/workspace/tweezers-software/driver-570.153.02}"
export XHAND_EGL_CONFIG="${XHAND_EGL_CONFIG:-/mnt/workspace/tweezers-software/vulkan/nvidia_570_egl.json}"
export XHAND_VULKAN_CONFIG="${XHAND_VULKAN_CONFIG:-/mnt/workspace/tweezers-software/vulkan/nvidia_570_icd.json}"
export XHAND_CONSTRAINT_FILE="${XHAND_CONSTRAINT_FILE:-$XHAND_SIM_ENV/project-constraints.txt}"

if [[ ! -x "$ISAAC_PYTHON" ]]; then
    echo "仿真 Python 不存在：$ISAAC_PYTHON；请配置 env/local.env。" >&2
    return 1
fi
export PATH="$(dirname -- "$ISAAC_PYTHON"):$PATH"
hash -r

# Use this server's verified user-space driver profile when it is present.
# Another server may use its own correctly configured NVIDIA runtime.
if [[ -e "$XHAND_DRIVER_DIR/libcuda.so.1" && -f "$XHAND_EGL_CONFIG" && -f "$XHAND_VULKAN_CONFIG" ]]; then
    export LD_LIBRARY_PATH="$XHAND_DRIVER_DIR:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
    unset VK_DRIVER_FILES VK_ADD_DRIVER_FILES
    export __EGL_VENDOR_LIBRARY_FILENAMES="$XHAND_EGL_CONFIG"
    export VK_ICD_FILENAMES="$XHAND_VULKAN_CONFIG"
fi
export OMNI_KIT_ACCEPT_EULA=YES
if [[ "$(id -u)" -eq 0 ]]; then export OMNI_KIT_ALLOW_ROOT=1; fi
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "$XDG_RUNTIME_DIR" || ! -O "$XDG_RUNTIME_DIR" ]]; then
    export XDG_RUNTIME_DIR="$(mktemp -d /tmp/tweezers-runtime-XXXXXXXX)" || return 1
    chmod 700 "$XDG_RUNTIME_DIR" || return 1
fi
unset PIP_CONSTRAINT
cd "$PROJECT" || return 1
printf 'Tweezers root: %s\nSimulation Python: %s\n' "$TWEEZERS_ROOT" "$ISAAC_PYTHON"

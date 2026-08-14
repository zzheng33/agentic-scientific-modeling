#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults for ARM/Grace nodes with NVIDIA GPUs. For CPU-only ARM runs:
#   DEVICE=cpu VENDOR=auto DEVICES="" ./scripts/run_ptychi_arm.sh
export DEVICE="${DEVICE:-cuda}"
export VENDOR="${VENDOR:-auto}"
export CONDA_ENV="${CONDA_ENV:-ptychopinn_torch_arm}"
export GPU_LABEL="${GPU_LABEL:-}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/modeling_exp}"

MODULE_PATH="${MODULE_PATH:-/soft/modulefiles}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9.1}"
CONDA_MODULE="${CONDA_MODULE:-conda/nvidia/suse15.6/2025.01-11}"

set +u
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module use "${MODULE_PATH}"
  module load "${CUDA_MODULE}"
  module load "${CONDA_MODULE}"
  hash -r
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
set -u

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python)"
  else
    echo "No Python executable found. Activate the correct ARM env or set PYTHON_BIN." >&2
    exit 1
  fi
fi

exec "${SCRIPT_DIR}/run_ptychi.sh" "$@"

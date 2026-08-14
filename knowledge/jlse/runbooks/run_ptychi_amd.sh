#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults for AMD/ROCm nodes. PyTorch ROCm still uses device="cuda".
export DEVICE="${DEVICE:-cuda}"
export VENDOR="${VENDOR:-amd}"
export CONDA_ENV="${CONDA_ENV:-ptychi_rocm}"
export GPU_LABEL="${GPU_LABEL:-MI300X}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/modeling_exp}"

MODULE_PATH="${MODULE_PATH:-/soft/modulefiles}"
ROCM_MODULE="${ROCM_MODULE:-rocm/7.0.2}"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  [[ -d "${MODULE_PATH}" ]] && module use "${MODULE_PATH}"
  module load "${ROCM_MODULE}" || true
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  CONDA_BASE="${CONDA_BASE:-${HOME}/miniforge3}"
  if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
  fi

  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    export PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python)"
  else
    echo "No Python executable found. Activate the correct ROCm env or set PYTHON_BIN." >&2
    exit 1
  fi
fi

exec "${SCRIPT_DIR}/run_ptychi.sh" "$@"

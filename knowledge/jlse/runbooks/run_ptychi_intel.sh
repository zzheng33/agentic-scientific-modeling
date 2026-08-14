#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PBS_O_WORKDIR:-}" && -f "${PBS_O_WORKDIR}/scripts/run_ptychi.py" ]]; then
  REPO_ROOT="$(cd "${PBS_O_WORKDIR}" && pwd)"
elif [[ -f "${PWD}/scripts/run_ptychi.py" ]]; then
  REPO_ROOT="$(pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${REPO_ROOT}/scripts/run_ptychi.py" ]]; then
  echo "Could not locate pty-chi repo root. Submit from the repo root or set PBS_O_WORKDIR correctly." >&2
  echo "Resolved REPO_ROOT=${REPO_ROOT}" >&2
  exit 1
fi

# Defaults for Intel Max / XPU nodes.
export DEVICE="${DEVICE:-xpu}"
export VENDOR="${VENDOR:-intel}"
export GPU_LABEL="${GPU_LABEL:-Max}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/modeling_exp}"
export VENV_DIR="${VENV_DIR:-${REPO_ROOT}/../ptychopinn-venvs/aurora}"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load gcc/13.4.0 || true
  module load python/3.12.12 || true
  module load py-pip/25.1.1 || true
  module load py-numpy/2.3.4 || true
  module load py-scipy/1.16.3 || true
  module load py-h5py/3.14.0 || true
  module load py-matplotlib/3.10.7 || true
  module load py-pandas/2.3.3 || true
  module load py-torch/2.10.0 || true
  module load py-torchvision/0.25.0 || true
  module load py-torchaudio/2.10.0 || true
  module load xpu-smi/1.3.5 || true
fi

cd "${REPO_ROOT}"
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

export ZE_AFFINITY_MASK="${DEVICE_INDEX:-${DEVICES:-0}}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

exec "${SCRIPT_DIR}/run_ptychi.sh" "$@"

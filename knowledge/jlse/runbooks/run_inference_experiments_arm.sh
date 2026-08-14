#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Edit these defaults for your ARM/Grace experiment sweep.
DATASETS=(TP1 TP2 IC1 IC2 NCM FLY1 LFP W LCLS)
# DATASETS=(IC2)
BATCH_SIZES=(32 64 128 256 512 1024)
# BATCH_SIZES=(1024)
DEVICE="cuda"
VENDOR="auto"
DEVICES="${DEVICES:-0}"
INTERVAL="0.2"
WARMUP_SECONDS="0.5"
CONDA_ENV="ptychopinn_torch_arm"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU_LABEL=""
OUTPUT_ROOT="power_experiments"
CONTINUE_ON_ERROR="false"
TEST="${TEST:-false}"

MODULE_PATH="/soft/modulefiles"
CONDA_MODULE="conda/nvidia/suse15.6/2025.01-11"
CUDA_MODULE="cuda/12.9.1"

set +u
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi

module use "${MODULE_PATH}"
module load "${CUDA_MODULE}"
module load "${CONDA_MODULE}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
set -u

cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No Python executable found. Activate the correct ARM env or set PYTHON_BIN." >&2
    exit 1
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Using Python: ${PYTHON_BIN}"

CMD=(
  "${PYTHON_BIN}" scripts/run_inference_experiments.py
  --datasets "${DATASETS[@]}"
  --batch-sizes "${BATCH_SIZES[@]}"
  --device "${DEVICE}"
  --vendor "${VENDOR}"
  --interval "${INTERVAL}"
  --warmup-seconds "${WARMUP_SECONDS}"
  --conda-env "${CONDA_ENV}"
  --output-root "${OUTPUT_ROOT}"
)

if [[ -n "${DEVICES}" ]]; then
  CMD+=(--devices "${DEVICES}")
fi

if [[ -n "${GPU_LABEL}" ]]; then
  CMD+=(--gpu-label "${GPU_LABEL}")
fi

if [[ "${CONTINUE_ON_ERROR}" == "true" ]]; then
  CMD+=(--continue-on-error)
fi

if [[ "${TEST}" == "true" ]]; then
  CMD+=(--test)
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

"${CMD[@]}"

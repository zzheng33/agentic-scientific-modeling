#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Edit these defaults for your AMD/ROCm experiment sweep.
DATASETS=(TP1 TP2 IC1 IC2 NCM FLY1 LFP W LCLS)
# DATASETS=(TP2)
BATCH_SIZES=(32 64 128 256 512 1024)
# BATCH_SIZES=(1024)
DEVICE="${DEVICE:-cuda}"
VENDOR="${VENDOR:-amd}"
DEVICES="${DEVICES:-0}"
INTERVAL="${INTERVAL:-0.2}"
WARMUP_SECONDS="${WARMUP_SECONDS:-0.5}"
CONDA_ENV="${CONDA_ENV:-ptychi_rocm}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU_LABEL="${GPU_LABEL:-MI300A}"
OUTPUT_ROOT="${OUTPUT_ROOT:-power_experiments}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
TEST="${TEST:-false}"
CONDA_BASE="${CONDA_BASE:-${HOME}/miniforge3}"

MODULE_PATH="/soft/modulefiles"
ROCM_MODULE="rocm/7.0.2"

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi

module use "${MODULE_PATH}"
module load "${ROCM_MODULE}"

cd "${REPO_ROOT}"

if [[ -z "${PYTHON_BIN}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No Python executable found. Activate the correct ROCm env or set PYTHON_BIN." >&2
    exit 1
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Using Python: ${PYTHON_BIN}"

if ! "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  echo "Selected Python does not see AMD GPUs through ROCm PyTorch." >&2
  echo "Use a ROCm PyTorch env, then rerun. Example:" >&2
  echo "  CONDA_ENV=ptychopinn_torch_rocm ./scripts/run_inference_experiments_amd.sh" >&2
  echo "or:" >&2
  echo "  PYTHON_BIN=/path/to/rocm/env/bin/python ./scripts/run_inference_experiments_amd.sh" >&2
  exit 1
fi

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

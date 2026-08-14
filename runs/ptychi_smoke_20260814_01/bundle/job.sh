#!/bin/bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-/home/zhong.zheng/agentic-runs/ptychi_smoke_20260814_01}"
MODULE_PATH="${MODULE_PATH:-/soft/modulefiles}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9.1}"
CONDA_MODULE="${CONDA_MODULE:-conda/nvidia/suse15.6/2025.01-11}"
export CONDA_ENV="${CONDA_ENV:-ptychopinn_torch_arm}"

set +u
if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module use "$MODULE_PATH"
  module load "$CUDA_MODULE"
  module load "$CONDA_MODULE"
  hash -r
fi
MODULE_PYTHON="$(command -v python)"
CONDA_BASE="$(dirname "$(dirname "$MODULE_PYTHON")")"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
hash -r
set -u

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export PYTHONPATH="/home/zhong.zheng/pty-chi/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$BUNDLE_ROOT"
mkdir -p results
echo "hostname=$(hostname)"
echo "CONDA_ENV=$CONDA_ENV"
echo "PYTHON_BIN=$PYTHON_BIN"
module list 2>&1 || true
"$PYTHON_BIN" -c 'import torch, ptychi.api; print("torch=" + torch.__version__); print("cuda_available=" + str(torch.cuda.is_available())); print("device=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))'
"$PYTHON_BIN" run_benchmark.py \
  --bundle-root "$BUNDLE_ROOT" \
  --app-root /home/zhong.zheng/pty-chi \
  --python-bin "$PYTHON_BIN"

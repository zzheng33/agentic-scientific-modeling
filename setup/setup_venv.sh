#!/usr/bin/env bash
set -euo pipefail

AGENTIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${AGENTIC_DIR}/venv"
AGENTIC_PYTHON_BIN="${AGENTIC_PYTHON:-python3}"
PROJECT_ROOT="$(cd -- "${AGENTIC_DIR}/.." && pwd)"
export HF_HOME="${AGENTIC_HF_HOME:-${PROJECT_ROOT}/models/huggingface}"

if ! "${AGENTIC_PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "error: Python 3.11 or newer is required; set AGENTIC_PYTHON to its executable" >&2
  exit 1
fi

"${AGENTIC_PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install -r "${AGENTIC_DIR}/requirements.txt"

if [[ "${AGENTIC_SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  "${VENV_DIR}/bin/python" "${AGENTIC_DIR}/download_models.py" \
    --hf-home "${HF_HOME}"
else
  echo "Skipping model download because AGENTIC_SKIP_MODEL_DOWNLOAD=1"
fi

echo "Agentic virtual environment is ready: ${VENV_DIR}"
echo "Hugging Face models are stored under: ${HF_HOME}"
echo "Verify the local model cache without network access with:"
echo "${VENV_DIR}/bin/python ${AGENTIC_DIR}/download_models.py --hf-home ${HF_HOME} --local-files-only"
echo "Build the characterization literature index with:"
echo "${AGENTIC_DIR}/../agentic rag-index build"
echo "Build the JLSE operational index with:"
echo "${AGENTIC_DIR}/../agentic jlse-rag build"

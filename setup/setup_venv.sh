#!/usr/bin/env bash
set -euo pipefail

AGENTIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${AGENTIC_DIR}/venv"
AGENTIC_PYTHON_BIN="${AGENTIC_PYTHON:-python3}"

if ! "${AGENTIC_PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "error: Python 3.11 or newer is required; set AGENTIC_PYTHON to its executable" >&2
  exit 1
fi

"${AGENTIC_PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install -r "${AGENTIC_DIR}/requirements.txt"

echo "Agentic virtual environment is ready: ${VENV_DIR}"
echo "Build the characterization literature index with:"
echo "${AGENTIC_DIR}/../agentic rag-index build"

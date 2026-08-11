#!/usr/bin/env bash
set -euo pipefail

AGENTIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${AGENTIC_DIR}/venv"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install -r "${AGENTIC_DIR}/requirements.txt"

echo "Agentic virtual environment is ready: ${VENV_DIR}"
echo "Run the API check with:"
echo "${VENV_DIR}/bin/python ${AGENTIC_DIR}/agents/characterization/test.py"

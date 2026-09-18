#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_VERSION="${VLLM_VERSION:-0.12.0}"

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), "Se requiere Python >= 3.10"'
"${PYTHON_BIN}" -m pip install --upgrade "pip>=24.2"
"${PYTHON_BIN}" -m pip install --upgrade "packaging>=24.0"
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  "vllm==${VLLM_VERSION}" \
  "huggingface-hub>=0.34,<1.0"
"${PYTHON_BIN}" -m pip check

echo "Dependencias Nemotron/L40S instaladas (vLLM ${VLLM_VERSION})."
echo "El deployment debe usar una L40S completa de 48 GB, no una partición vGPU/MIG."

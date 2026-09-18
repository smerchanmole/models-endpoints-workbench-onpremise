#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SENTENCE_TRANSFORMERS_VERSION="${SENTENCE_TRANSFORMERS_VERSION:-5.1.2}"

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), "Se requiere Python >= 3.10"'
"${PYTHON_BIN}" -m pip install --upgrade "pip>=24.2"
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  "sentence-transformers==${SENTENCE_TRANSFORMERS_VERSION}" \
  "huggingface-hub>=0.34,<1.0" \
  "sentencepiece>=0.2,<0.3"
"${PYTHON_BIN}" -m pip check

echo "Dependencias de embeddings/L40S instaladas."

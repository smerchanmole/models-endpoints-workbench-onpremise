#!/usr/bin/env bash
set -euo pipefail

# SP2 observado: el driver expone CUDA 12.9. Se fija cu128 para evitar que
# pip instale una rueda CUDA 13 incompatible al resolver sentence-transformers.
PYTHON_BIN="${PYTHON_BIN:-python3}"
SENTENCE_TRANSFORMERS_VERSION="${SENTENCE_TRANSFORMERS_VERSION:-5.1.2}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10), "Se requiere Python >= 3.10"'
"${PYTHON_BIN}" -m pip install --upgrade "pip>=24.2"
"${PYTHON_BIN}" -m pip install --upgrade "packaging>=24.0"
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==${TORCH_VERSION}"
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  "sentence-transformers==${SENTENCE_TRANSFORMERS_VERSION}" \
  "huggingface-hub>=0.34,<1.0" \
  "sentencepiece>=0.2,<0.3"
"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" -c 'import torch; assert torch.version.cuda == "12.8", torch.version.cuda; print("torch", torch.__version__, "CUDA", torch.version.cuda)'

echo "Dependencias de embeddings/A100 instaladas con PyTorch CUDA 12.8."

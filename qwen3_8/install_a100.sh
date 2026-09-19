#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
CUDA_VARIANT="${VLLM_CUDA_VARIANT:-129}"
CPU_ARCH="$(uname -m)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${CPU_ARCH}" in
  x86_64|aarch64) ;;
  *)
    echo "Arquitectura no soportada por la rueda oficial de vLLM: ${CPU_ARCH}" >&2
    exit 1
    ;;
esac

DEFAULT_VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}%2Bcu${CUDA_VARIANT}-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-${DEFAULT_VLLM_WHEEL}}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu${CUDA_VARIANT}}"

echo "Instalando Qwen3.8/A100 con Python=$(${PYTHON_BIN} --version 2>&1), vLLM=${VLLM_VERSION}, CUDA=cu${CUDA_VARIANT}, arch=${CPU_ARCH}"
"${PYTHON_BIN}" -c 'import sys; assert (3, 10) <= sys.version_info < (3, 14), "Se requiere Python 3.10-3.13"'

# Cloudera incluye wheel 0.46.x (packaging>=24) y langchain-core 0.3.x
# (packaging<26). Este intervalo satisface ambos y hace reproducible el build.
"${PYTHON_BIN}" -m pip install --upgrade \
  "pip>=24.2,<26.0" \
  "packaging>=24.0,<26.0"

# El artefacto PyPI de vLLM 0.29.0 usa CUDA 13.0. Se fuerza la rueda cu129
# completa para que pip no pueda seleccionar accidentalmente ese artefacto.
"${PYTHON_BIN}" -m pip uninstall -y vllm torch torchvision torchaudio
"${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
  "${VLLM_WHEEL_URL}" \
  --extra-index-url "${PYTORCH_INDEX_URL}"
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  -r "${SCRIPT_DIR}/requirements.txt"

"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" - <<'PY'
import torch
import vllm

print("Installed vLLM", vllm.__version__)
print("Installed PyTorch", torch.__version__)
print("PyTorch CUDA build", torch.version.cuda)
if not (torch.version.cuda or "").startswith("12.9"):
    raise RuntimeError(
        f"Se esperaba PyTorch CUDA 12.9, pero se encontró {torch.version.cuda!r}"
    )
PY

echo "Dependencias Qwen3.8/A100 instaladas correctamente."

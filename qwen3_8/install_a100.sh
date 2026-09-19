#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
CUDA_VARIANT="${VLLM_CUDA_VARIANT:-129}"
CPU_ARCH="$(uname -m)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

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

# No instalar sobre el Python global del Runtime: Cloudera SP2 incluye, entre
# otros, numpy<2 y protobuf==4.25.3, mientras que vLLM 0.29 necesita versiones
# posteriores. El venv evita modificar o validar esos paquetes del sistema.
if [[ -e "${VENV_DIR}" && ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Existe ${VENV_DIR}, pero no es un entorno virtual válido" >&2
  exit 1
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"

# Algunos ML Runtimes exportan PIP_USER=true para que las instalaciones del
# proyecto terminen en ~/.local. Dentro de un venv pip rechaza explícitamente
# esa combinación. La sobrescritura solo afecta a este proceso de build.
export PIP_USER=false
unset PIP_PREFIX PIP_TARGET PYTHONUSERBASE

"${VENV_PYTHON}" -m pip install --upgrade \
  "pip>=24.2,<26.0" \
  "packaging>=24.0,<26.0"

# El artefacto PyPI de vLLM 0.29.0 usa CUDA 13.0. Se fuerza la rueda cu129
# completa para que pip no pueda seleccionar accidentalmente ese artefacto.
"${VENV_PYTHON}" -m pip install --no-cache-dir --upgrade \
  "${VLLM_WHEEL_URL}" \
  --extra-index-url "${PYTORCH_INDEX_URL}"
"${VENV_PYTHON}" -m pip install --no-cache-dir \
  -r "${SCRIPT_DIR}/requirements.txt"

"${VENV_PYTHON}" -m pip check
EXPECTED_VLLM_VERSION="${VLLM_VERSION}" \
EXPECTED_CUDA_VARIANT="${CUDA_VARIANT}" \
"${VENV_PYTHON}" - <<'PY'
import os

import torch
import transformers
import vllm
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

print("Installed vLLM", vllm.__version__)
print("Installed PyTorch", torch.__version__)
print("PyTorch CUDA build", torch.version.cuda)
print("Installed Transformers", transformers.__version__)
print("Qwen architecture", Qwen3_5ForConditionalGeneration.__name__)
expected_vllm = os.environ["EXPECTED_VLLM_VERSION"]
if not (
    vllm.__version__ == expected_vllm
    or vllm.__version__.startswith(f"{expected_vllm}+")
):
    raise RuntimeError(
        f"Se esperaba vLLM {expected_vllm}, encontrado {vllm.__version__!r}"
    )
if transformers.__version__ != "5.15.0":
    raise RuntimeError(
        f"Se esperaba Transformers 5.15.0, encontrado {transformers.__version__!r}"
    )
expected_cuda = os.environ["EXPECTED_CUDA_VARIANT"]
expected_cuda = f"{expected_cuda[:-1]}.{expected_cuda[-1]}"
if not (torch.version.cuda or "").startswith(expected_cuda):
    raise RuntimeError(
        f"Se esperaba PyTorch CUDA {expected_cuda}, pero se encontró "
        f"{torch.version.cuda!r}"
    )
PY

echo "Dependencias Qwen3.8/A100 instaladas correctamente en ${VENV_DIR}."

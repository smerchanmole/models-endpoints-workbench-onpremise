#!/usr/bin/env bash
set -euo pipefail

# Cloudera ejecuta este fichero durante el build del Model Deployment.
# Configure MODEL_FAMILY y GPU_TYPE como variables del build.
MODEL_FAMILY="${MODEL_FAMILY:-nemotron}"
GPU_TYPE="${GPU_TYPE:-l40s}"

case "${MODEL_FAMILY,,}:${GPU_TYPE,,}" in
  nemotron:l40s) exec bash nemotron/install_l40s.sh ;;
  nemotron:h100) exec bash nemotron/install_h100.sh ;;
  embedding:l40s) exec bash embedding/install_l40s.sh ;;
  embedding:a100) exec bash embedding/install_a100.sh ;;
  embedding:h100) exec bash embedding/install_h100.sh ;;
  qwen3_8:a100) exec bash qwen3_8/install_a100.sh ;;
  *)
    echo "Combinación no válida: MODEL_FAMILY=${MODEL_FAMILY}, GPU_TYPE=${GPU_TYPE}" >&2
    echo "Valores válidos: MODEL_FAMILY=nemotron con l40s|h100; embedding con l40s|a100|h100; qwen3_8 con a100" >&2
    exit 2
    ;;
esac

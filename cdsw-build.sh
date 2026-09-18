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
  embedding:h100) exec bash embedding/install_h100.sh ;;
  *)
    echo "Combinación no válida: MODEL_FAMILY=${MODEL_FAMILY}, GPU_TYPE=${GPU_TYPE}" >&2
    echo "Valores válidos: nemotron|embedding y l40s|h100" >&2
    exit 2
    ;;
esac


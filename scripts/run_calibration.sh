#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: $0 CASE GPU_ID OUTPUT_DIR ARTIFACT_ROOT" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="$2"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CORET_ARTIFACT_ROOT="$4"
export PYTHONPATH="$REPO/scripts:$REPO/research_hab${PYTHONPATH:+:$PYTHONPATH}"
python "$REPO/scripts/run_calibration.py" --case "$1" --output "$3" \
  --artifact-root "$4"
python "$REPO/scripts/compare_calibration.py" --case "$1" \
  --calibration-output "$3" --artifact-root "$4" | tee "$3/comparison.json"

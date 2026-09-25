#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "usage: $0 SHARD_MANIFEST WORKER_ID GPU_ID WORKER_DIR ARTIFACT_ROOT" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="$3"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CORET_ARTIFACT_ROOT="$5"
export PYTHONPATH="$REPO/research_hab${PYTHONPATH:+:$PYTHONPATH}"
exec python "$REPO/scripts/run_worker.py" \
  --shard-manifest "$1" --worker-id "$2" --worker-dir "$4" \
  --artifact-root "$5"

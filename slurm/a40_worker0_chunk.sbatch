#!/usr/bin/env bash
#SBATCH --job-name=coret-a40-w0
#SBATCH --partition=gpu-a40
#SBATCH --nodelist=afrodita
#SBATCH --gres=gpu:gpu0:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:30:00
#SBATCH --output=slurm_logs/a40_worker0_%j.out
#SBATCH --error=slurm_logs/a40_worker0_%j.err
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-3]$ ]]; then
  echo "usage: sbatch $0 CHUNK_INDEX" >&2
  exit 2
fi

REPO="$HOME/projects/coret-proof-carrying-transformer-cluster"
FRESH_ROOT="$HOME/coret-a40-historical-127-v1"
WORKER="$FRESH_ROOT/worker_0"
ENV_PREFIX="$HOME/.conda/envs/coret-cluster-a268be7"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
mkdir -p "$WORKER"/{runtime,tmpdir,cache,cuda_cache,torch_extensions,pycache,chunk_records}
export TMPDIR="$WORKER/tmpdir"
export XDG_CACHE_HOME="$WORKER/cache"
export CUDA_CACHE_PATH="$WORKER/cuda_cache"
export TORCH_EXTENSIONS_DIR="$WORKER/torch_extensions"
export PYTHONPYCACHEPREFIX="$WORKER/pycache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CORET_ARTIFACT_ROOT="$REPO/runtime_inputs"
export PYTHONPATH="$REPO/scripts:$REPO/research_hab"
cd "$WORKER/runtime"

exec flock -n "$WORKER/worker.lock" \
  python "$REPO/scripts/run_a40_fresh_chunk.py" \
    --worker-id 0 --chunk-index "$1" --worker-dir "$WORKER" \
    --artifact-root "$REPO/runtime_inputs"

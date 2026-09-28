#!/usr/bin/env bash
#SBATCH --job-name=coret-d6-full-w0
#SBATCH --partition=gpu-a40
#SBATCH --nodelist=afrodita
#SBATCH --gres=gpu:gpu0:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/depth6_full_w0_%j.out
#SBATCH --error=slurm_logs/depth6_full_w0_%j.err
set -euo pipefail
: "${CHUNK_INDEX:?submit with --export=ALL,CHUNK_INDEX=N}"
case "$CHUNK_INDEX" in 0|1|2|3) ;; *) echo "invalid CHUNK_INDEX" >&2; exit 2;; esac
REPO="$HOME/projects/coret-proof-carrying-transformer-cluster"
FRESH_ROOT="$HOME/coret-a40-historical-depth6-full-v1"
WORKER="$FRESH_ROOT/worker_0"
ENV_PREFIX="$HOME/.conda/envs/coret-cluster-a268be7"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
mkdir -p "$REPO/slurm_logs" "$WORKER"/{runtime,tmp,cache,cuda_cache,torch_extensions,pycache,chunk_records}
export TMPDIR="$WORKER/tmp" XDG_CACHE_HOME="$WORKER/cache"
export CUDA_CACHE_PATH="$WORKER/cuda_cache" TORCH_EXTENSIONS_DIR="$WORKER/torch_extensions"
export PYTHONPYCACHEPREFIX="$WORKER/pycache" CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$REPO/scripts:$REPO/research_hab" CUDA_VISIBLE_DEVICES=0
cd "$WORKER/runtime"
exec flock -n "$WORKER/worker.lock" python "$REPO/scripts/run_depth6_full_chunk.py" chunk \
  --worker-id 0 --chunk-index "$CHUNK_INDEX" --worker-dir "$WORKER"

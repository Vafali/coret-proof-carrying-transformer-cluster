#!/usr/bin/env bash
#SBATCH --job-name=coret-b24-screen-w1
#SBATCH --partition=gpu-a40
#SBATCH --nodelist=afrodita
#SBATCH --gres=gpu:gpu1:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/benchmark24_screening_w1_%j.out
#SBATCH --error=slurm_logs/benchmark24_screening_w1_%j.err
set -euo pipefail
REPO="$HOME/projects/coret-proof-carrying-transformer-cluster"
CAMPAIGN="$HOME/coret-benchmark24-producer-screening-v1"
WORKER="$CAMPAIGN/workers/worker_1"
ENV_PREFIX="$HOME/.conda/envs/coret-cluster-a268be7"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
mkdir -p "$WORKER"/{tmp,cache,cuda_cache,torch_extensions,pycache}
export TMPDIR="$WORKER/tmp" XDG_CACHE_HOME="$WORKER/cache"
export CUDA_CACHE_PATH="$WORKER/cuda_cache" TORCH_EXTENSIONS_DIR="$WORKER/torch_extensions"
export PYTHONPYCACHEPREFIX="$WORKER/pycache" CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$REPO/scripts:$REPO/research_hab" CUDA_VISIBLE_DEVICES=1
export CORET_ARTIFACT_ROOT="$REPO/runtime_inputs"
cd "$REPO"
exec python -u scripts/run_benchmark24_producer_screening_v1.py worker --worker-id 1 --campaign-root "$CAMPAIGN" --artifact-root "$REPO/runtime_inputs"

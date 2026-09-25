#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKTREE="/home/vafali_ubuntu/worktrees/lookahead-branching-runtime-opt-v1"
MAIN_RESEARCH="/mnt/c/users/david-despacho/documents/vafali projects/lookahead-branching/research_hab"
export PYTHONPATH="$WORKTREE/research_hab:$MAIN_RESEARCH${PYTHONPATH:+:$PYTHONPATH}"

cd "$WORKTREE"
python research_hab/coret_optimized_historical_127_v1.py benchmark --authorized

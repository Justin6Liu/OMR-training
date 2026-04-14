#!/usr/bin/env bash
# Convenience launcher: activates venv, sets defaults, and runs TorchVision Faster R-CNN training.
# Usage (from repo root or any dir):
#   EPOCHS=40 LR=0.002 BATCH=1 NUM_WORKERS=2 ./scripts/run_frcnn_tv.sh
# Env defaults: EPOCHS=20, LR=0.002, BATCH=1, NUM_WORKERS=2, PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate virtualenv
source ~/venvs/omr-cascade/bin/activate

# Sensible defaults; override via env vars on invocation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
export EPOCHS="${EPOCHS:-20}"
export LR="${LR:-0.002}"
export BATCH="${BATCH:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"

cd "$ROOT_DIR"
echo "Using env: EPOCHS=$EPOCHS LR=$LR BATCH=$BATCH NUM_WORKERS=$NUM_WORKERS PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
python3 scripts/train_fasterrcnn_tv.py

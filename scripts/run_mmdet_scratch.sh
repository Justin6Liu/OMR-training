#!/bin/bash
# Generic MMDetection launcher for MUSCIMA++ experiments on the cluster.
# Defaults target the jl1430 cluster paths from the current handoff.

set -euo pipefail

VENV=${VENV:-/home/users/jl1430/venvs/omr-cascade}
REPO=${REPO:-/home/users/jl1430/jl1430/OMR-training}
SCRATCH_BASE=${SCRATCH_BASE:-/usr/project/xtmp/$USER/omr_runs}
CONFIG=${CONFIG:-configs/faster_rcnn_swin_t_omr.py}
TRAIN_ANN_FILE=${TRAIN_ANN_FILE:-train.json}
VAL_ANN_FILE=${VAL_ANN_FILE:-val.json}
DATA_ROOT=${DATA_ROOT:-/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/}
IMG_ROOT=${IMG_ROOT:-/home/users/jl1430/muscima-pp/v2.0/data/images/}
RUN_NAME=${RUN_NAME:-$(basename "${CONFIG%.py}")_$(date +%Y%m%d_%H%M%S)}
WORK_DIR="$SCRATCH_BASE/$RUN_NAME"
CONSOLE_LOG="$WORK_DIR/console.log"

mkdir -p "$SCRATCH_BASE" "$WORK_DIR"

source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

cd "$REPO"
echo "Config: $CONFIG"
echo "Train ann: $TRAIN_ANN_FILE"
echo "Val ann: $VAL_ANN_FILE"
echo "Work dir: $WORK_DIR"
echo "Console log: $CONSOLE_LOG"

set +e
mim train mmdet "$CONFIG" \
  --work-dir "$WORK_DIR" \
  --cfg-options \
    train_dataloader.dataset.data_root="$DATA_ROOT" \
    val_dataloader.dataset.data_root="$DATA_ROOT" \
    test_dataloader.dataset.data_root="$DATA_ROOT" \
    train_dataloader.dataset.data_prefix.img="$IMG_ROOT" \
    val_dataloader.dataset.data_prefix.img="$IMG_ROOT" \
    test_dataloader.dataset.data_prefix.img="$IMG_ROOT" \
    train_dataloader.dataset.ann_file="$TRAIN_ANN_FILE" \
    val_dataloader.dataset.ann_file="$VAL_ANN_FILE" \
    test_dataloader.dataset.ann_file="$VAL_ANN_FILE" \
  >"$CONSOLE_LOG" 2>&1
status=$?
set -e

if [ "$status" -ne 0 ]; then
  echo "Training failed with exit code $status"
  echo "Last 60 lines from $CONSOLE_LOG:"
  tail -n 60 "$CONSOLE_LOG" || true
  exit "$status"
fi

echo "Run finished: $WORK_DIR"

#!/bin/bash
# Example YOLO11 training launcher for MUSCIMA++ on the cluster.
#
# Usage:
#   MODEL=yolo11l.pt bash docs/yolo11_baseline/train_yolo11_cluster.sh
#   MODEL=/home/users/jl1430/Documents/GitHub/Schenkerian_OMR/trained_models/yolo11l_muscima.pt \
#     RUN_NAME=yolo11l_muscima_resume \
#     bash docs/yolo11_baseline/train_yolo11_cluster.sh

set -euo pipefail

VENV=${VENV:-/home/users/jl1430/venvs/omr-cascade}
REPO=${REPO:-/home/users/jl1430/jl1430/OMR-training}
DATA_DIR=${DATA_DIR:-$REPO/outputs/muscima_yolo}
MODEL=${MODEL:-yolo11l.pt}
PROJECT=${PROJECT:-/usr/project/xtmp/$USER/omr_runs/yolo11}
RUN_NAME=${RUN_NAME:-yolo11l_muscima}
EPOCHS=${EPOCHS:-200}
IMGSZ=${IMGSZ:-960}
BATCH=${BATCH:-4}
DEVICE=${DEVICE:-0}
WORKERS=${WORKERS:-4}

source "$VENV/bin/activate"
cd "$REPO"

echo "Model: $MODEL"
echo "Data: $DATA_DIR/data.yaml"
echo "Project: $PROJECT"
echo "Run name: $RUN_NAME"

yolo detect train \
  model="$MODEL" \
  data="$DATA_DIR/data.yaml" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  device="$DEVICE" \
  workers="$WORKERS" \
  project="$PROJECT" \
  name="$RUN_NAME"

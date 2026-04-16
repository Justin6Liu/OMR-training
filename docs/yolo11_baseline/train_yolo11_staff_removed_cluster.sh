#!/bin/bash
# Prepare a YOLO dataset from cleaned staff-removed MUSCIMA++ pairings and
# fine-tune a checkpoint on that reconstructed subset.
#
# Usage:
#   MATCH_JSON=/path/to/muscima_gt_match_filtered_clean.json \
#   COCO_JSONS="/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/train.json /home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/val.json" \
#   MODEL=/home/users/jl1430/Documents/GitHub/Schenkerian_OMR/trained_models/yolo11l_muscima.pt \
#   RUN_NAME=yolo11l_staff_removed_adapt \
#   bash docs/yolo11_baseline/train_yolo11_staff_removed_cluster.sh

set -euo pipefail

VENV=${VENV:-/home/users/jl1430/venvs/omr-cascade}
REPO=${REPO:-/home/users/jl1430/jl1430/OMR-training}
MATCH_JSON=${MATCH_JSON:-$REPO/outputs/muscima_gt_match_filtered_clean.json}
COCO_JSONS=${COCO_JSONS:-"$REPO/datasets/muscima_coco/train.json $REPO/datasets/muscima_coco/val.json"}
DATA_DIR=${DATA_DIR:-$REPO/outputs/muscima_staff_removed_yolo}
MODEL=${MODEL:-/home/users/jl1430/Documents/GitHub/Schenkerian_OMR/trained_models/yolo11l_muscima.pt}
PROJECT=${PROJECT:-/usr/project/xtmp/$USER/omr_runs/yolo11}
RUN_NAME=${RUN_NAME:-yolo11l_staff_removed_adapt}
EPOCHS=${EPOCHS:-40}
IMGSZ=${IMGSZ:-960}
BATCH=${BATCH:-4}
DEVICE=${DEVICE:-0}
WORKERS=${WORKERS:-4}
VAL_RATIO=${VAL_RATIO:-0.2}
SEED=${SEED:-42}
LR0=${LR0:-0.001}

source "$VENV/bin/activate"
cd "$REPO"

echo "Preparing dataset from: $MATCH_JSON"
echo "COCO JSONs: $COCO_JSONS"
python scripts/prepare_staff_removed_yolo_dataset.py \
  --match-json "$MATCH_JSON" \
  --coco-jsons $COCO_JSONS \
  --out-dir "$DATA_DIR" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --link-images

echo "Model: $MODEL"
echo "Data: $DATA_DIR/data.yaml"
echo "Project: $PROJECT"
echo "Run name: $RUN_NAME"
echo "Learning rate: $LR0"

yolo detect train \
  model="$MODEL" \
  data="$DATA_DIR/data.yaml" \
  epochs="$EPOCHS" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  device="$DEVICE" \
  workers="$WORKERS" \
  lr0="$LR0" \
  project="$PROJECT" \
  name="$RUN_NAME"

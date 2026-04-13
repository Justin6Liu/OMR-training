#!/bin/bash
# Run Cascade R-CNN training on the cluster, writing heavy outputs to scratch.
# Requires:
#   - venv at ~/venvs/omr-cascade
#   - data: COCO JSON at /home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco
#           images at /home/users/jl1430/muscima-pp/v2.0/data/images
# Adjust paths if your layout differs.

set -euo pipefail

VENV=~/venvs/omr-cascade
REPO=~/jl1430/OMR-training
SCRATCH_BASE=/usr/project/xtmp/$USER/omr_runs

mkdir -p "$SCRATCH_BASE" "$REPO/artifacts"

source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH=""

RUN_DIR="$SCRATCH_BASE/cascade_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

cd "$REPO"
echo "Working dir: $RUN_DIR"

mim train mmdet configs/cascade_omr.py \
  --work-dir "$RUN_DIR" \
  --cfg-options \
    train_dataloader.dataset.data_root=/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco \
    val_dataloader.dataset.data_root=/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco \
    test_dataloader.dataset.data_root=/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco \
    train_dataloader.dataset.data_prefix.img=/home/users/jl1430/muscima-pp/v2.0/data/images \
    val_dataloader.dataset.data_prefix.img=/home/users/jl1430/muscima-pp/v2.0/data/images \
    test_dataloader.dataset.data_prefix.img=/home/users/jl1430/muscima-pp/v2.0/data/images \
    train_dataloader.dataset.serialize_data=False \
    val_dataloader.dataset.serialize_data=False \
    test_dataloader.dataset.serialize_data=False \
    train_dataloader.dataset.filter_cfg=None \
    val_dataloader.dataset.filter_cfg=None \
    test_dataloader.dataset.filter_cfg=None \
    train_dataloader.batch_size=2 train_dataloader.num_workers=1 train_dataloader.persistent_workers=False \
    val_dataloader.batch_size=1 val_dataloader.num_workers=1 val_dataloader.persistent_workers=False \
    train_cfg.max_epochs=${EPOCHS:-12} \
    load_from=None \
    default_hooks.checkpoint.interval=1 \
    default_hooks.checkpoint.save_last=True

echo "Training done, collecting artifacts..."
for f in "$RUN_DIR"/latest.pth "$RUN_DIR"/epoch_*.pth; do
  [ -f "$f" ] && cp -n "$f" "$REPO/artifacts/"
done
cp -n "$RUN_DIR"/*.log.json "$REPO/artifacts/" 2>/dev/null || true
cp -nr "$RUN_DIR"/vis_data "$REPO/artifacts/" 2>/dev/null || true

echo "Cleaning scratch run dir..."
rm -rf "$RUN_DIR"

echo "Artifacts stored in $REPO/artifacts"
exit 0

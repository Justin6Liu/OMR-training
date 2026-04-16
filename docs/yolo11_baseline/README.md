# YOLO11 Fine-Tuning Guide

This folder describes a direct YOLO11 baseline for MUSCIMA++ using the same train/val split already present in this repo.

There are two valid starting points:

1. Official YOLO11 pretrained checkpoint
   - start from `yolo11l.pt`
   - Ultralytics will download it automatically the first time

2. Your teammate's already fine-tuned MUSCIMA++ checkpoint
   - start from `yolo11l_muscima.pt`
   - this is useful if you want to resume or adapt an already-strong music-symbol detector

## Files

- [train_yolo11_cluster.sh](/Users/justinliu/Documents/GitHub/OMR-training/docs/yolo11_baseline/train_yolo11_cluster.sh)
- [train_yolo11_staff_removed_cluster.sh](/Users/justinliu/Documents/GitHub/OMR-training/docs/yolo11_baseline/train_yolo11_staff_removed_cluster.sh)
- [convert_coco_to_yolo.py](/Users/justinliu/Documents/GitHub/OMR-training/scripts/convert_coco_to_yolo.py)
- [prepare_staff_removed_yolo_dataset.py](/Users/justinliu/Documents/GitHub/OMR-training/scripts/prepare_staff_removed_yolo_dataset.py)

## 1. Prepare a YOLO-format dataset

Convert the existing COCO split into Ultralytics YOLO format:

```bash
cd /home/users/jl1430/jl1430/OMR-training
python scripts/convert_coco_to_yolo.py \
  --train-json datasets/muscima_coco/train.json \
  --val-json datasets/muscima_coco/val.json \
  --images-root /home/users/jl1430/muscima-pp/v2.0/data/images \
  --out-dir outputs/muscima_yolo
```

This creates:

- `outputs/muscima_yolo/train/images`
- `outputs/muscima_yolo/train/labels`
- `outputs/muscima_yolo/val/images`
- `outputs/muscima_yolo/val/labels`
- `outputs/muscima_yolo/data.yaml`

By default the script symlinks images instead of copying them.

## 2. Install Ultralytics if needed

```bash
source /home/users/jl1430/venvs/omr-cascade/bin/activate
pip install ultralytics
```

## 3. Fine-tune from the official YOLO11 checkpoint

This is the cleanest baseline if you want to compare YOLO directly against the R-CNN line:

```bash
cd /home/users/jl1430/jl1430/OMR-training
yolo detect train \
  model=yolo11l.pt \
  data=outputs/muscima_yolo/data.yaml \
  epochs=200 \
  imgsz=960 \
  batch=4 \
  device=0 \
  workers=4 \
  project=/usr/project/xtmp/jl1430/omr_runs/yolo11 \
  name=yolo11l_muscima_from_official
```

If memory is tight, lower `imgsz` to `640` and `batch` to `2` or `1`.

## 4. Fine-tune from your teammate's MUSCIMA++ checkpoint

If you already have a trained checkpoint from the Schenkerian repo, you can start from that instead of `yolo11l.pt`.

Local path on this laptop appears to be:

```text
/Users/justinliu/Documents/GitHub/Schenkerian_OMR/trained_models/yolo11l_muscima.pt
```

If running on the cluster, use the cluster path to the same file. Based on your earlier commands, that is likely:

```text
/home/users/jl1430/Documents/GitHub/Schenkerian_OMR/trained_models/yolo11l_muscima.pt
```

Resume/fine-tune from that checkpoint:

```bash
cd /home/users/jl1430/jl1430/OMR-training
yolo detect train \
  model=/home/users/jl1430/Documents/GitHub/Schenkerian_OMR/trained_models/yolo11l_muscima.pt \
  data=outputs/muscima_yolo/data.yaml \
  epochs=100 \
  imgsz=960 \
  batch=4 \
  device=0 \
  workers=4 \
  project=/usr/project/xtmp/jl1430/omr_runs/yolo11 \
  name=yolo11l_muscima_resume
```

This is not a fair "from generic pretraining" baseline anymore, but it is a pragmatic way to test whether YOLO is already strong enough to beat your current R-CNN line.

## 5. Evaluate the trained checkpoint

Ultralytics stores the best checkpoint in:

```text
/usr/project/xtmp/jl1430/omr_runs/yolo11/<run_name>/weights/best.pt
```

Validate it with:

```bash
yolo detect val \
  model=/usr/project/xtmp/jl1430/omr_runs/yolo11/yolo11l_muscima_from_official/weights/best.pt \
  data=outputs/muscima_yolo/data.yaml \
  imgsz=960 \
  device=0
```

## 6. Fine-tune on the cleaned staff-removed MUSCIMA++ subset

If you want to adapt a detector to the reconstructed 94-image staff-removed
subset, the repo now includes a dataset builder plus a dedicated launcher.

The dataset builder:

- reads your cleaned match JSON
- maps each original MUSCIMA++ page back to its COCO annotations
- replaces the image pixels with the paired staff-removed image
- writes a deterministic YOLO train/val split plus `manifest.json`

Prepare the dataset directly:

```bash
cd /home/users/jl1430/jl1430/OMR-training
python scripts/prepare_staff_removed_yolo_dataset.py \
  --match-json outputs/muscima_gt_match_filtered_clean.json \
  --coco-jsons datasets/muscima_coco/train.json datasets/muscima_coco/val.json \
  --out-dir outputs/muscima_staff_removed_yolo \
  --val-ratio 0.2 \
  --seed 42
```

Or use the end-to-end launcher to prepare the subset and fine-tune from your
teammate's checkpoint in one step:

```bash
cd /home/users/jl1430/jl1430/OMR-training
MATCH_JSON=/path/to/muscima_gt_match_filtered_clean.json \
MODEL=/home/users/jl1430/Documents/GitHub/Schenkerian_OMR/trained_models/yolo11l_muscima.pt \
RUN_NAME=yolo11l_staff_removed_adapt \
EPOCHS=40 \
IMGSZ=960 \
BATCH=4 \
LR0=0.001 \
bash docs/yolo11_baseline/train_yolo11_staff_removed_cluster.sh
```

Recommended adaptation settings for this subset:

- start from an existing MUSCIMA++ checkpoint
- keep the learning rate modest, e.g. `LR0=0.001`
- train briefly, e.g. `EPOCHS=30` to `50`
- use `IMGSZ=960` if memory allows, else `640`
- treat the held-out split as a practical adaptation check, not a paper-style benchmark

## Which route should you try first?

Recommended order:

1. `yolo11l.pt`
   - gives you the cleanest apples-to-apples YOLO baseline

2. `yolo11l_muscima.pt`
   - useful if you mainly care about practical performance and want to see whether a strong YOLO checkpoint already solves the task better than your MMDetection runs

## Why this is worth trying

Your MMDetection Cascade R101 line is constrained by 32 GB VRAM, especially with pseudo labels and small-object tuning. A direct YOLO11 baseline may be stronger partly because:

- it starts from a strong detector checkpoint
- its training stack is simpler
- it is often easier to fit in memory
- your teammate's checkpoint is already adapted to music-symbol detection

That makes YOLO a valid practical baseline even if it does not settle the architecture question by itself.

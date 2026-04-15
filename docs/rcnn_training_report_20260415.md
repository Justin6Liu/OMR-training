# R-CNN Training Report

## Scope

This report summarizes only the in-house R-CNN detector work on MUSCIMA++, excluding the later direct YOLO11 fine-tuning baseline.

## Dataset and Setup

- Dataset: MUSCIMA++ detection task using COCO-format annotations
- Images: `/home/users/jl1430/muscima-pp/v2.0/data/images`
- COCO JSONs: `datasets/muscima_coco/train.json` and `datasets/muscima_coco/val.json`
- Pseudo-label source: YOLO11 teacher predictions merged into COCO JSONs
- Primary MMDetection config: `configs/cascade_omr.py`
- Additional experiments: `configs/faster_rcnn_swin_t_omr.py`, `configs/cascade_rcnn_swin_t_omr.py`

## Models Tried

### 1. TorchVision Faster R-CNN fallback

- Purpose: lower-memory fallback when MMDetection Cascade experiments were unstable or blocked
- Initialization: TorchVision COCO-pretrained Faster R-CNN
- Outcome: trained successfully and produced meaningful detections
- Best earlier result was approximately:
  - `mAP ≈ 0.026`

This confirmed that the MUSCIMA++ COCO conversion and basic detector pipeline were functional.

### 2. Cascade R-CNN with ResNet-101 backbone

- Purpose: main MMDetection detector line
- Initialization: COCO-pretrained Cascade R-CNN R101 FPN checkpoint
- Key run: pseudo-label training with YOLO11-generated train annotations
- Best result achieved:
  - `mAP ≈ 0.057`
  - `AP50 ≈ 0.089`
- Best checkpoint:
  - `/usr/project/xtmp/jl1430/omr_runs/cascade_r101_pseudo05/epoch_24.pth`

This was the strongest R-CNN result obtained and remained the best non-YOLO model throughout the project.

### 3. Faster R-CNN with Swin-T backbone

- Purpose: test whether a hierarchical transformer backbone would outperform CNN backbones on small music symbols
- Initialization: pretrained Swin-T checkpoint
- Training completed successfully after config/debug fixes
- Final result:
  - `mAP = 0.024`
  - `AP50 = 0.036`

This underperformed the Cascade R101 pseudo-label baseline.

### 4. Cascade R-CNN with Swin-T backbone

- Purpose: test a stronger two-stage transformer-based detector variant
- Initialization: pretrained Swin-T checkpoint
- Training completed successfully after config/debug fixes
- Final result:
  - `mAP = 0.024`
  - `AP50 = 0.034`
  - `AP_small = 0.012`

This also underperformed the Cascade R101 pseudo-label baseline and did not justify further backbone switching.

## Pseudo-Label Experiments

Pseudo labels were generated from a YOLO11 teacher using:

- `scripts/generate_pseudo_labels_yolo.py`

Several pseudo-label variants were tried:

- merged GT + pseudo labels
- cleaner pseudo labels with higher confidence thresholds, NMS, and top-k filtering
- sparse pseudo labels with strong pruning

### What worked

- A moderate pseudo-label setup improved Cascade R101 substantially and produced the best overall R-CNN result (`mAP ≈ 0.057`).

### What failed

- Cleaner and denser pseudo-label sets repeatedly caused GPU out-of-memory failures during RPN IoU assignment in MMDetection Cascade R101.
- Even after reducing image scale and proposal counts aggressively, merged pseudo-label training remained unstable on 32 GB GPUs.
- Sparse pseudo-label variants still caused OOM in some full-page Cascade configurations.

## Main Failures and Blockers

### 1. Severe VRAM limits in Cascade R-CNN

The main failure mode was repeated CUDA OOM during:

- `MaxIoUAssigner`
- `bbox_overlaps`
- RPN target assignment

This was especially severe for:

- small-object-tuned anchor settings
- merged GT + pseudo-label runs
- higher-resolution or denser-target configurations

The problem was not only steady-state memory use; there were large transient spikes during IoU computation.

### 2. Small-object tuning was difficult to stabilize

We tried:

- smaller anchors
- more small-object-biased proposal settings
- lower image scales
- lower proposal counts
- lower per-image max detections

These changes reduced memory somewhat, but often degraded learning or still failed to prevent OOM.

### 3. Swin-T did not improve performance

Although Swin-T models trained successfully after debugging, they did not outperform the best CNN-based Cascade run. This suggests that for the current data size and hardware constraints, a direct backbone switch did not solve the actual bottlenecks.

## Key Conclusions

1. The best R-CNN result was the COCO-pretrained Cascade R-CNN R101 model trained with a moderate YOLO11 pseudo-label set.
2. Small-object performance remained weak overall, especially for thin symbols and dense notation.
3. MMDetection Cascade training was strongly constrained by 32 GB VRAM, particularly when pseudo labels increased the number of targets per page.
4. Swin-T backbones were technically feasible but did not improve validation performance over the best R101 Cascade result.
5. Full-page Cascade retraining with cleaner merged pseudo-labels was not practical on the available hardware.

## Final Status of the R-CNN Line

- Best non-YOLO detector:
  - Cascade R-CNN R101 + pseudo labels
  - `mAP ≈ 0.057`, `AP50 ≈ 0.089`
- Best checkpoint:
  - `/usr/project/xtmp/jl1430/omr_runs/cascade_r101_pseudo05/epoch_24.pth`
- Main reason the line stalled:
  - hardware-limited MMDetection training under dense small-object supervision

## Recommended Interpretation

The R-CNN experiments were useful and produced a meaningful baseline, but they were ultimately constrained by GPU memory and did not scale well to denser pseudo-label training. The most successful configuration remained a ResNet-101 Cascade model with carefully limited pseudo-label use, rather than more aggressive small-object tuning or transformer backbones.

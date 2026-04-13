# OMR Model Training

Goal: train high-accuracy detectors for optical music recognition (dense, tiny symbols).

Planned model families:
- Deformable DETR
- RT-DETRv2 / RT-DETRv3
- Cascade R-CNN + FPN

Next steps:
1. Add dataset pointers and tiling/preprocessing scripts for high-res scores.
2. Create training configs for each model family.
3. Run smoke tests, then full training/eval; track metrics and checkpoints here.


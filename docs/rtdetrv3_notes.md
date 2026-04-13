# RT-DETRv3 OMR Training Playbook

## Why RT-DETRv3 for OMR
- Transformer detector, NMS-free → fewer duplicate boxes on crowded notation.
- Dense positive supervision head (training only) fixes sparse labels in classic DETR, lifting small/overlapping-symbol recall.
- Still real-time enough for tiled inference on 3k×2k pages.

## Key Obstacles
- High-res scores require tiling with overlap; must merge boxes back to page coordinates.
- Severe class imbalance (rare musical symbols) → need focal/class weights.
- Tiny objects → choose sufficient queries, decoder layers, multi-scale features.
- Memory pressure from large tiles; may need grad-accum, AMP, smaller tile sizes.

## Hyperparameters to Tune
- Tile size/overlap: recall vs. speed; overlap ~15–25%.
- Queries (300–500) & decoder layers (6–8): more improves crowded recall, costs compute.
- Dense-head loss weight: balances dense supervision vs. Hungarian main loss.
- Matching weights (cls/box/GIoU): tighten localization vs. precision/recall.
- LR & scheduler: AdamW, cosine, backbone LR multiplier (0.1× typical).
- Augmentations: crop/flip/color; mosaic for printed scores; moderate rotation.
- Class weighting / focal gamma–alpha: mitigate imbalance.
- Batch size / grad accumulation: fit VRAM; preserves effective LR.
- EMA, weight decay: regularize small-data training.
- Inference tile stride & score/NMS-less thresholds: recall/precision tradeoff.

## Recommended Training Flow
1) Prepare COCO-format annotations; generate tiles (e.g., 1216×1216, 20% overlap) plus mapping back.
2) Start from RT-DETRv3 pretrained backbone (or v2 if v3 weights unavailable).
3) Config: queries 400, decoder 6, loss weights cls 1.0 / box 5.0 / GIoU 2.0, dense head 0.5–1.0; LR 1e-4 (backbone 0.1×), warmup 1k, cosine, weight decay 0.05; batch 8 (use grad-accum as needed); AMP on.
4) Phases: (a) 2–5 epoch sanity run on small subset; (b) 50–150 epochs full run with early stop on val mAP.
5) Validate with tiled inference; track mAP@50 and mAP@50-95, plus recall on rare classes.
6) Save best checkpoint + inference config (tile size/stride, thresholds).

## Time (rough, single 24GB GPU)
- Sanity subset (200 tiles, 5 epochs): ~20–30 min.
- Full MUSCIMA-scale (5k–10k tiles, 80–120 epochs, batch 8, AMP): ~8–14 h (R18/R50); +30–40% for R101.

## Unit Test Ideas
- Dataloader smoke: batch loads, boxes within image bounds.
- Tiling round-trip: tile→merge preserves box coords within tolerance.
- Training step: forward/backward two steps keep finite losses.
- Checkpoint save/reload: loss stable within 1e-5 after reload.
- Inference regression: tiny fixed set yields mAP@50 > floor (e.g., 0.1) to catch broken heads.


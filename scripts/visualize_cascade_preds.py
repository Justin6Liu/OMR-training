"""
Run inference with a Cascade checkpoint and save visualized detections.

Usage (cluster example):
  python3 scripts/visualize_cascade_preds.py \
    --config configs/cascade_omr.py \
    --checkpoint /home/users/jl1430/jl1430/OMR-training/artifacts/run_20260414_001918/epoch_12.pth \
    --ann-file datasets/muscima_coco/val.json \
    --img-root /home/users/jl1430/muscima-pp/v2.0/data/images \
    --out-dir /home/users/jl1430/jl1430/OMR-training/artifacts/run_20260414_001918/vis_manual \
    --num 30
"""

import argparse
import json
import os
import random
from pathlib import Path

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmdet.visualization import DetLocalVisualizer


def random_palette(n):
    random.seed(42)
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ann-file", required=True, help="COCO val json to select images")
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num", type=int, default=20, help="Number of images to visualize")
    ap.add_argument("--score-thr", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = init_detector(args.config, args.checkpoint, device="cuda")
    classes = model.dataset_meta["classes"]
    vis = DetLocalVisualizer(name="vis")
    vis.dataset_meta = {"classes": classes, "palette": random_palette(len(classes))}

    coco = json.load(open(args.ann_file))
    images = coco["images"][: args.num]
    img_root = Path(args.img_root)

    for im in images:
        img_path = img_root / im["file_name"]
        res = inference_detector(model, str(img_path))
        # add_datasample expects ndarray BGR
        img = mmcv.imread(str(img_path))
        data_sample = res  # already DetDataSample
        vis.add_datasample(
            name=img_path.stem,
            image=img,
            data_sample=data_sample,
            draw_gt=False,
            out_file=str(Path(args.out_dir) / f"{img_path.stem}_pred.png"),
            pred_score_thr=args.score_thr,
        )
        print(f"saved {Path(args.out_dir) / f'{img_path.stem}_pred.png'}")


if __name__ == "__main__":
    main()

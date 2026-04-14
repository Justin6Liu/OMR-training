"""
Pseudo-label all images in a folder (no GT) using a YOLO teacher and write a COCO json.

Usage (cluster example):
  python3 scripts/pseudo_label_folder_yolo.py \
    --model-path ~/Documents/GitHub/Schenkerian_OMR/trained_models/yolo11l_muscima.pt \
    --categories-json datasets/muscima_coco/train.json \
    --images-dir CvcMuscima-Distortions \
    --out-json datasets/muscima_coco/distortions_pseudo_yolo11.json \
    --score-thr 0.5 --device cuda --batch 1 --imgsz 640

Notes:
- categories are loaded from the provided COCO json (just to get category ids/names).
- Each image gets a new image_id incrementing from max existing in categories json.
- No GT annotations are included—only teacher pseudo boxes.
"""

import argparse
import json
import os
from pathlib import Path

from ultralytics import YOLO
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--categories-json", required=True, help="COCO json to read categories (and base for id start)")
    ap.add_argument("--images-dir", required=True, help="Root folder to scan for images")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    # Load categories and establish starting ids
    base = json.load(open(args.categories_json))
    categories = base["categories"]
    cat_name_to_id = {c["name"]: c["id"] for c in categories}
    start_img_id = max([im["id"] for im in base["images"]]) + 1 if base.get("images") else 1
    next_ann_id = 1

    # Collect images to process
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    img_paths = sorted([p for p in Path(args.images_dir).rglob("*") if p.suffix.lower() in exts])
    images = []
    for idx, p in enumerate(img_paths):
        images.append({"id": start_img_id + idx, "file_name": str(p.relative_to(Path(args.images_dir)))})

    model = YOLO(args.model_path)
    results = model.predict(
        [str(p) for p in img_paths],
        imgsz=args.imgsz,
        conf=args.score_thr,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
    )

    annotations = []
    for img_info, res in tqdm(zip(images, results), total=len(images), desc="Pseudo-labeling"):
        for b in res.boxes:
            cls_idx = int(b.cls.item())
            cls_name = model.names.get(cls_idx, str(cls_idx))
            if cls_name not in cat_name_to_id:
                continue
            score = float(b.conf.item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": img_info["id"],
                    "category_id": cat_name_to_id[cls_name],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": score,
                }
            )
            next_ann_id += 1

    out = {"images": images, "annotations": annotations, "categories": categories}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Saved {len(annotations)} pseudo annotations over {len(images)} images to {args.out_json}")


if __name__ == "__main__":
    main()


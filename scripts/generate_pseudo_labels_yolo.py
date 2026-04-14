"""
Generate COCO-format pseudo labels using a YOLO teacher.

Usage (example on cluster):
    python3 scripts/generate_pseudo_labels_yolo.py \
        --model-path /home/users/jl1430/Documents/GitHub/Schenkerian_OMR/trained_models/yolo11l_muscima.pt \
        --ann-json datasets/muscima_coco/train.json \
        --images-root /home/users/jl1430/muscima-pp/v2.0/data/images \
        --out-json datasets/muscima_coco/train_pseudo_yolo11.json \
        --score-thr 0.5 \
        --merge-gt

Notes:
- Requires `ultralytics` installed (`pip install ultralytics`).
- Assumes YOLO class names align with COCO categories in `ann-json`; detections
  whose class name is not found are skipped.
- By default, ground-truth annotations are merged into the output (use
  `--no-merge-gt` to output pseudo labels only).
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
from tqdm import tqdm


def load_coco(ann_path: Path):
    coco = json.loads(Path(ann_path).read_text())
    images = coco["images"]
    categories = coco["categories"]
    annotations = coco.get("annotations", [])
    return images, categories, annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Path to YOLO teacher .pt")
    ap.add_argument("--ann-json", required=True, help="Existing COCO json (images/categories)")
    ap.add_argument("--images-root", required=True, help="Root dir for images")
    ap.add_argument("--out-json", required=True, help="Output COCO json with pseudo labels")
    ap.add_argument("--score-thr", type=float, default=0.5, help="Confidence threshold for pseudo labels")
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    ap.add_argument("--merge-gt", action=argparse.BooleanOptionalAction, default=True,
                    help="Include ground-truth annotations in the output")
    args = ap.parse_args()

    images, categories, gt_anns = load_coco(args.ann_json)
    cat_name_to_id = {c["name"]: c["id"] for c in categories}
    next_ann_id = max([a["id"] for a in gt_anns], default=0) + 1

    model = YOLO(args.model_path)
    img_root = Path(args.images_root)

    pseudo_anns = []
    # Use list of image paths to keep order aligned with COCO images
    image_by_id = {im["id"]: im for im in images}
    image_paths = [img_root / im["file_name"] for im in images]

    results = model.predict(
        image_paths,
        imgsz=args.imgsz,
        conf=args.score_thr,
        stream=True,
        verbose=False,
    )

    for im, res in tqdm(zip(images, results), total=len(images), desc="Pseudo-labeling"):
        image_id = im["id"]
        for b in res.boxes:
            cls_idx = int(b.cls.item())
            cls_name = model.names.get(cls_idx, str(cls_idx))
            if cls_name not in cat_name_to_id:
                continue
            score = float(b.conf.item())
            xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            pseudo_anns.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": cat_name_to_id[cls_name],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": score,
                }
            )
            next_ann_id += 1

    out = {
        "images": images,
        "categories": categories,
        "annotations": (gt_anns if args.merge_gt else []) + pseudo_anns,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Saved {len(pseudo_anns)} pseudo annotations to {args.out_json}")


if __name__ == "__main__":
    main()


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
- Optional post-filtering can tighten pseudo labels with min-area, top-k, and
  class-wise NMS before writing the merged COCO json.
"""

import argparse
import json
from pathlib import Path

import torch
from torchvision.ops import nms
from ultralytics import YOLO
from tqdm import tqdm


def load_coco(ann_path: Path):
    coco = json.loads(Path(ann_path).read_text())
    images = coco["images"]
    categories = coco["categories"]
    annotations = coco.get("annotations", [])
    return images, categories, annotations


def load_thresholds(path):
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def apply_filters(ann_list, score_thr, min_area, topk, nms_iou):
    kept = [
        ann for ann in ann_list
        if ann["score"] >= score_thr and ann["area"] >= min_area
    ]
    if not kept:
        return []

    if nms_iou is not None:
        boxes = torch.tensor(
            [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in kept],
            dtype=torch.float32,
        )
        scores = torch.tensor([a["score"] for a in kept], dtype=torch.float32)
        keep_idx = nms(boxes, scores, nms_iou).tolist()
        kept = [kept[i] for i in keep_idx]

    kept.sort(key=lambda ann: ann["score"], reverse=True)
    if topk is not None:
        kept = kept[:topk]
    return kept


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
    ap.add_argument("--device", default="cuda", help="Device for inference (e.g., 'cuda', 'cuda:0', or 'cpu')")
    ap.add_argument("--batch", type=int, default=1, help="Batch size for YOLO predict")
    ap.add_argument("--per-class-thr-json", help="Optional JSON mapping class name to minimum score.")
    ap.add_argument("--min-area", type=float, default=4.0, help="Discard pseudo boxes smaller than this area.")
    ap.add_argument("--topk-per-image", type=int, help="Keep only the top-K pseudo boxes per image after filtering.")
    ap.add_argument("--class-wise-nms-iou", type=float, default=0.5,
                    help="Apply class-wise NMS with this IoU. Use negative value to disable.")
    ap.add_argument("--stats-json", help="Optional path to write summary stats.")
    args = ap.parse_args()

    images, categories, gt_anns = load_coco(args.ann_json)
    cat_name_to_id = {c["name"]: c["id"] for c in categories}
    next_ann_id = max([a["id"] for a in gt_anns], default=0) + 1
    per_class_thr = load_thresholds(args.per_class_thr_json)

    model = YOLO(args.model_path)
    img_root = Path(args.images_root)

    pseudo_anns = []
    kept_per_image = []
    # Use list of image paths to keep order aligned with COCO images
    image_by_id = {im["id"]: im for im in images}
    image_paths = [img_root / im["file_name"] for im in images]

    results = model.predict(
        image_paths,
        imgsz=args.imgsz,
        conf=args.score_thr,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
    )

    for im, res in tqdm(zip(images, results), total=len(images), desc="Pseudo-labeling"):
        image_id = im["id"]
        by_class = {}
        for b in res.boxes:
            cls_idx = int(b.cls.item())
            cls_name = model.names.get(cls_idx, str(cls_idx))
            if cls_name not in cat_name_to_id:
                continue
            score = float(b.conf.item())
            score_thr = per_class_thr.get(cls_name, args.score_thr)
            if score < score_thr:
                continue
            xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            by_class.setdefault(cls_name, []).append(
                {
                    "image_id": image_id,
                    "category_id": cat_name_to_id[cls_name],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": score,
                }
            )

        filtered = []
        for cls_name, cls_anns in by_class.items():
            filtered.extend(
                apply_filters(
                    cls_anns,
                    score_thr=per_class_thr.get(cls_name, args.score_thr),
                    min_area=args.min_area,
                    topk=args.topk_per_image,
                    nms_iou=args.class_wise_nms_iou if args.class_wise_nms_iou >= 0 else None,
                )
            )

        filtered.sort(key=lambda ann: ann["score"], reverse=True)
        if args.topk_per_image is not None:
            filtered = filtered[: args.topk_per_image]

        for ann in filtered:
            ann["id"] = next_ann_id
            pseudo_anns.append(ann)
            next_ann_id += 1
        kept_per_image.append(len(filtered))

    out = {
        "images": images,
        "categories": categories,
        "annotations": (gt_anns if args.merge_gt else []) + pseudo_anns,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Saved {len(pseudo_anns)} pseudo annotations to {args.out_json}")
    if args.stats_json:
        stats = {
            "num_images": len(images),
            "num_pseudo_annotations": len(pseudo_anns),
            "avg_pseudo_per_image": (sum(kept_per_image) / len(kept_per_image)) if kept_per_image else 0.0,
            "max_pseudo_per_image": max(kept_per_image) if kept_per_image else 0,
            "min_pseudo_per_image": min(kept_per_image) if kept_per_image else 0,
            "score_thr": args.score_thr,
            "per_class_thr_json": args.per_class_thr_json,
            "min_area": args.min_area,
            "topk_per_image": args.topk_per_image,
            "class_wise_nms_iou": args.class_wise_nms_iou,
        }
        Path(args.stats_json).write_text(json.dumps(stats, indent=2))
        print(f"Saved pseudo-label stats to {args.stats_json}")


if __name__ == "__main__":
    main()

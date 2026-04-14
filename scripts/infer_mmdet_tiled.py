"""
Run tiled MMDetection inference over a COCO image set and save merged detections.

Usage (cluster example):
  python scripts/infer_mmdet_tiled.py \
    --config configs/cascade_omr.py \
    --checkpoint /usr/project/xtmp/jl1430/omr_runs/cascade_r101_pseudo05/epoch_24.pth \
    --ann-file /home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/val.json \
    --img-root /home/users/jl1430/muscima-pp/v2.0/data/images \
    --out-json /home/users/jl1430/jl1430/OMR-training/artifacts/cascade_r101_tiled_preds.json
"""

import argparse
import json
from pathlib import Path

import mmcv
import torch
from mmdet.apis import inference_detector, init_detector
from torchvision.ops import nms
from tqdm import tqdm


def tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, max(length - tile, 1), stride))
    last = max(length - tile, 0)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def classwise_nms(detections, iou_thr):
    if not detections:
        return []
    merged = []
    by_class = {}
    for det in detections:
        by_class.setdefault(det["category_id"], []).append(det)
    for _, cls_dets in by_class.items():
        boxes = torch.tensor(
            [[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]] for d in cls_dets],
            dtype=torch.float32,
        )
        scores = torch.tensor([d["score"] for d in cls_dets], dtype=torch.float32)
        keep = nms(boxes, scores, iou_thr).tolist()
        merged.extend(cls_dets[i] for i in keep)
    return sorted(merged, key=lambda d: d["score"], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ann-file", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--tile-width", type=int, default=1024)
    ap.add_argument("--tile-height", type=int, default=1024)
    ap.add_argument("--tile-overlap", type=int, default=256)
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--max-per-img", type=int, default=300)
    ap.add_argument("--num", type=int, default=0, help="Optional image limit for smoke tests.")
    args = ap.parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)
    class_id_offset = 1

    coco = json.loads(Path(args.ann_file).read_text())
    images = coco["images"]
    if args.num > 0:
        images = images[: args.num]
    img_root = Path(args.img_root)
    predictions = []

    stride_x = max(args.tile_width - args.tile_overlap, 1)
    stride_y = max(args.tile_height - args.tile_overlap, 1)

    for im in tqdm(images, desc="Tiled inference"):
        img_path = img_root / im["file_name"]
        image = mmcv.imread(str(img_path))
        height, width = image.shape[:2]

        dets = []
        for x0 in tile_starts(width, args.tile_width, stride_x):
            for y0 in tile_starts(height, args.tile_height, stride_y):
                tile = image[y0:y0 + args.tile_height, x0:x0 + args.tile_width]
                result = inference_detector(model, tile)
                pred = result.pred_instances
                if len(pred) == 0:
                    continue
                bboxes = pred.bboxes.detach().cpu()
                scores = pred.scores.detach().cpu()
                labels = pred.labels.detach().cpu()

                for bbox, score, label in zip(bboxes, scores, labels):
                    score = float(score.item())
                    if score < args.score_thr:
                        continue
                    x1, y1, x2, y2 = bbox.tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if w <= 0 or h <= 0:
                        continue
                    dets.append(
                        {
                            "image_id": im["id"],
                            "category_id": int(label.item()) + class_id_offset,
                            "bbox": [x1 + x0, y1 + y0, w, h],
                            "score": score,
                        }
                    )

        dets = classwise_nms(dets, args.merge_iou)
        predictions.extend(dets[: args.max_per_img])

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(predictions, indent=2))
    print(f"Saved {len(predictions)} detections to {out_path}")


if __name__ == "__main__":
    main()

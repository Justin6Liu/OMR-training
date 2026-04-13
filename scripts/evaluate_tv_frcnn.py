"""Evaluate TorchVision Faster R-CNN checkpoints on MUSCIMA++ COCO val set.

Usage (from repo root, with venv active):
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
    python scripts/evaluate_tv_frcnn.py \
        --checkpoint /usr/project/xtmp/$USER/omr_runs/tv_frcnn/epoch_4.pth

If --checkpoint is omitted, all epoch_*.pth files in OUT_DIR are evaluated and
metrics are saved to metrics.json beside the checkpoints.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ---- Paths (cluster defaults; override with CLI if needed) ----
DEFAULT_IMG_ROOT = "/home/users/jl1430/muscima-pp/v2.0/data/images"
DEFAULT_VAL_JSON = "/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/val.json"
DEFAULT_OUT_DIR = f"/usr/project/xtmp/{os.environ.get('USER', 'user')}/omr_runs/tv_frcnn"


class CocoValDataset(torch.utils.data.Dataset):
    def __init__(self, coco: COCO, img_root: Path):
        self.coco = coco
        self.img_root = Path(img_root)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.img_root / img_info["file_name"]
        # torchvision returns CxHxW uint8 tensor
        img = torchvision.io.read_image(str(img_path)).float() / 255.0
        return img, img_id


def make_model(num_classes: int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def evaluate_checkpoint(ckpt_path: Path, coco: COCO, dataloader: DataLoader, device: torch.device):
    num_classes = max(cat["id"] for cat in coco.cats.values()) + 1
    model = make_model(num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    results = []
    with torch.inference_mode():
        for imgs, img_ids in dataloader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            out = outputs[0]
            boxes_xywh = box_convert(out["boxes"].cpu(), in_fmt="xyxy", out_fmt="xywh").tolist()
            scores = out["scores"].cpu().tolist()
            labels = out["labels"].cpu().tolist()
            img_id = int(img_ids[0])
            for bbox, score, label in zip(boxes_xywh, scores, labels):
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [float(x) for x in bbox],
                        "score": float(score),
                    }
                )

    coco_dt = coco.loadRes(results) if results else coco.loadRes([])
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = {
        "mAP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
    }
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to a single checkpoint .pth file")
    parser.add_argument("--img-root", type=str, default=DEFAULT_IMG_ROOT)
    parser.add_argument("--val-json", type=str, default=DEFAULT_VAL_JSON)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory containing checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coco = COCO(args.val_json)
    ds = CocoValDataset(coco, Path(args.img_root))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    checkpoints = []
    if args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    else:
        out_dir = Path(args.out_dir)
        checkpoints = sorted(out_dir.glob("epoch_*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {out_dir}")

    metrics = {}
    for ckpt in checkpoints:
        print(f"Evaluating {ckpt} on {device}...")
        stats = evaluate_checkpoint(ckpt, coco, loader, device)
        metrics[ckpt.name] = stats
        print(f"{ckpt.name}: mAP {stats['mAP']:.4f} | AP50 {stats['AP50']:.4f} | AP75 {stats['AP75']:.4f}")

    # Save metrics if multiple checkpoints were evaluated
    if len(checkpoints) > 1:
        out_path = Path(args.out_dir) / "metrics.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        best = max(metrics.items(), key=lambda kv: kv[1]["mAP"])
        print(f"Saved metrics to {out_path}\nBest: {best[0]} mAP={best[1]['mAP']:.4f}")


if __name__ == "__main__":
    main()


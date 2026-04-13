"""
Minimal RT-DETRv3 training scaffold for MUSCIMA++ tiles.

Requirements (install manually):
  pip install torch torchvision transformers==4.38.0 pycocotools pillow
  # If an official RT-DETRv3 checkpoint is available on HF, set --checkpoint accordingly.
  # Fallback: use RT-DETR v2/v1 checkpoint; dense-head features will be skipped gracefully.

This script:
  - loads COCO-format annotations (tiling optional if you pre-generated tiles),
  - feeds batches through a HF-style RT-DETR model,
  - supports small sanity runs via --max-steps/--subset,
  - saves checkpoints and evaluation metrics.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import (
        RTDetrForObjectDetection,
        AutoImageProcessor,
        get_cosine_schedule_with_warmup,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Install transformers >=4.38.0 to run this script. pip install transformers==4.38.0"
    ) from exc


# -----------------------
# Dataset
# -----------------------


@dataclass
class COCOTarget:
    boxes: torch.Tensor
    labels: torch.Tensor


class CocoDetectionDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, image_processor, subset: int | None = None):
        self.img_dir = Path(img_dir)
        self.coco = self._load_coco(ann_file)
        self.ids = list(self.coco["images"])
        if subset:
            self.ids = self.ids[:subset]
        self.processor = image_processor

        # index annotations by image_id for quick lookup
        self.ann_index: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.coco["annotations"]:
            self.ann_index.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_meta = self.ids[idx]
        img_path = self.img_dir / img_meta["file_name"]
        image = Image.open(img_path).convert("RGB")

        anns = self.ann_index.get(img_meta["id"], [])
        boxes, labels = [], []
        for ann in anns:
            bbox = ann["bbox"]  # [x, y, w, h]
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(ann["category_id"])

        target = {
            "image_id": torch.tensor([img_meta["id"]]),
            "annotations": [
                {
                    "bbox": boxes,
                    "category_id": labels,
                }
            ],
        }
        inputs = self.processor(images=image, annotations=target, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

    @staticmethod
    def _load_coco(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# -----------------------
# Training / Eval
# -----------------------


def collate_fn(batch):
    # HuggingFace processors handle padding inside; here we just stack dict lists
    collated = {}
    for key in batch[0].keys():
        collated[key] = [item[key] for item in batch]
    return collated


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, list):
            out[k] = [{kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in item.items()} for item in v]
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_processor = AutoImageProcessor.from_pretrained(args.checkpoint)
    model = RTDetrForObjectDetection.from_pretrained(
        args.checkpoint,
        ignore_mismatched_sizes=True,  # allow different class counts
        num_labels=args.num_classes,
    ).to(device)

    train_ds = CocoDetectionDataset(args.train_images, args.train_annotations, image_processor, subset=args.subset)
    val_ds = CocoDetectionDataset(args.val_images, args.val_annotations, image_processor, subset=args.subset_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    num_training_steps = args.epochs * math.ceil(len(train_loader))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    global_step = 0
    best_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            running += loss.item()
            global_step += 1

            if args.max_steps and global_step >= args.max_steps:
                break

        train_loss = running / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = Path(args.output_dir) / "best.pt"
            model.save_pretrained(ckpt_path)
            image_processor.save_pretrained(ckpt_path)

        if args.max_steps and global_step >= args.max_steps:
            break


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        batch = to_device(batch, device)
        outputs = model(**batch)
        losses.append(outputs.loss.item())
    return float(sum(losses) / max(1, len(losses)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", required=True, help="Path to train image directory (tiled).")
    ap.add_argument("--train-annotations", required=True, help="Path to train COCO json.")
    ap.add_argument("--val-images", required=True, help="Path to val image directory (tiled).")
    ap.add_argument("--val-annotations", required=True, help="Path to val COCO json.")
    ap.add_argument("--checkpoint", required=True, help="HF checkpoint name or path (RT-DETRv3/v2).")
    ap.add_argument("--num-classes", type=int, required=True, help="Number of detection classes.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--subset", type=int, default=None, help="Use first N train images (sanity).")
    ap.add_argument("--subset-val", type=int, default=None, help="Use first N val images.")
    ap.add_argument("--max-steps", type=int, default=None, help="Stop after N steps (sanity).")
    ap.add_argument("--output-dir", default="outputs/rtdetrv3")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)


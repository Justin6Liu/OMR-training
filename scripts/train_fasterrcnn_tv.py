"""
Minimal TorchVision Faster R-CNN fine-tune on MUSCIMA++ COCO data.
Uses torchvision COCO-pretrained weights (downloaded from download.pytorch.org).
Writes checkpoints to /usr/project/xtmp/$USER/omr_runs/tv_frcnn.
Tunable via env vars: EPOCHS (default 80), LR (0.0015), BATCH (1), NUM_WORKERS (2).
Augmentation: color jitter, random resize (short side 800–1200, max 1500), horizontal flip.
Scheduler: CosineAnnealingLR. Anchors tuned for skinny/tiny symbols.

Paths assume cluster layout; adjust as needed.
"""

import os
import json
import random
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms.functional import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR


# ---- Paths (cluster defaults) ----
IMG_ROOT = "/home/users/jl1430/muscima-pp/v2.0/data/images"
TRAIN_JSON = "/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/train.json"
VAL_JSON = "/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/val.json"
OUT_DIR = f"/usr/project/xtmp/{os.environ.get('USER', 'user')}/omr_runs/tv_frcnn"


def load_coco(ann_path):
    coco = json.load(open(ann_path))
    id2fname = {im["id"]: im["file_name"] for im in coco["images"]}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    return id2fname, anns, coco["categories"]


def jitter_color(img):
    # Light color jitter; avoids heavy color shifts for sheet music
    b = 0.1
    c = 0.1
    s = 0.1
    h = 0.02
    img = TF.adjust_brightness(img, 1 + random.uniform(-b, b))
    img = TF.adjust_contrast(img, 1 + random.uniform(-c, c))
    img = TF.adjust_saturation(img, 1 + random.uniform(-s, s))
    img = TF.adjust_hue(img, random.uniform(-h, h))
    return img


def random_resize(img, boxes, short_min=800, short_max=1200, max_size=1500):
    """Resize keeping aspect ratio; scales boxes accordingly."""
    w, h = img.size
    short_side = min(w, h)
    target_short = random.randint(short_min, short_max)
    scale = target_short / short_side
    # Respect max size
    if max(w, h) * scale > max_size:
        scale = max_size / max(w, h)
    if scale == 1.0:
        return img, boxes
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = TF.resize(img, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
    boxes = boxes * scale
    return img, boxes


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, ann_path, img_root, augment=False):
        self.id2fname, self.anns, _ = load_coco(ann_path)
        self.ids = list(self.id2fname.keys())
        self.img_root = Path(img_root)
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.img_root / self.id2fname[img_id]
        img = Image.open(img_path).convert("RGB")
        boxes, labels = [], []
        W, H = img.size
        for a in self.anns.get(img_id, []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            # Clamp to image bounds and re-order just in case
            x1, x2 = sorted((max(0.0, x1), min(W, x2)))
            y1, y2 = sorted((max(0.0, y1), min(H, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(a["category_id"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        if self.augment and len(target["boxes"]) > 0:
            # Color jitter (mild)
            img = jitter_color(img)
            # Random resize (scale jitter)
            if random.random() < 0.9:
                b = target["boxes"]
                img, b = random_resize(img, b)
                target["boxes"] = b
            # Horizontal flip
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
                w = img.width
                b = target["boxes"].clone()
                b[:, [0, 2]] = w - b[:, [2, 0]]
                target["boxes"] = b

        return TF.to_tensor(img), target


def collate(batch):
    return tuple(zip(*batch))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(os.environ.get("BATCH", "1"))
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))
    epochs = int(os.environ.get("EPOCHS", "80"))
    lr = float(os.environ.get("LR", "0.0015"))

    train_ds = CocoDataset(TRAIN_JSON, IMG_ROOT, augment=True)
    val_ds = CocoDataset(VAL_JSON, IMG_ROOT, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    # Load COCO-pretrained detector with tuned anchors
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=(0.2, 0.5, 1.0, 2.0, 5.0),
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,  # custom anchors; avoid RPN shape mismatch
        weights_backbone="IMAGENET1K_V2",
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=300,
    )
    torch.backends.cudnn.benchmark = False
    # Adjust classifier head for dataset classes
    num_classes = max(max((t["labels"].max().item() if len(t["labels"]) else 0) for _, t in train_ds), 114) + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # Sampler and head tweaks
    model.rpn.batch_size_per_image = 512
    model.roi_heads.batch_size_per_image = 1024
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.6

    for epoch in range(epochs):
        model.train()
        for imgs, targets in train_loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        torch.cuda.empty_cache()
        scheduler.step()
        ckpt_path = Path(OUT_DIR) / f"epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"saved {ckpt_path}")

    print("Done. Checkpoints in", OUT_DIR)


if __name__ == "__main__":
    main()

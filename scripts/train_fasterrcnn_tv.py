"""
Minimal TorchVision Faster R-CNN fine-tune on MUSCIMA++ COCO data.
Uses torchvision COCO-pretrained weights (downloaded from download.pytorch.org).
Writes checkpoints to /usr/project/xtmp/$USER/omr_runs/tv_frcnn.
Tunable via env vars: EPOCHS (default 20), LR (0.002), BATCH (1), NUM_WORKERS (1).
Includes simple horizontal flip augmentation and a StepLR scheduler.

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
        for a in self.anns.get(img_id, []):
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        # Simple horizontal flip augmentation for training
        if self.augment and random.random() < 0.5:
            img = ImageOps.mirror(img)
            w = img.width
            b = target["boxes"].clone()
            b[:, 0] = w - b[:, 2]
            b[:, 2] = w - b[:, 0]
            target["boxes"] = b

        return TF.to_tensor(img), target


def collate(batch):
    return tuple(zip(*batch))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(os.environ.get("BATCH", "1"))
    num_workers = int(os.environ.get("NUM_WORKERS", "1"))
    epochs = int(os.environ.get("EPOCHS", "20"))
    lr = float(os.environ.get("LR", "0.002"))

    train_ds = CocoDataset(TRAIN_JSON, IMG_ROOT, augment=True)
    val_ds = CocoDataset(VAL_JSON, IMG_ROOT, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    # Load COCO-pretrained detector
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    torch.backends.cudnn.benchmark = False
    # Adjust classifier head for dataset classes
    num_classes = max(max((t["labels"].max().item() if len(t["labels"]) else 0) for _, t in train_ds), 114) + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)

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

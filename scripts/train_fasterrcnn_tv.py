"""
Minimal TorchVision Faster R-CNN fine-tune on MUSCIMA++ COCO data.
Uses torchvision COCO-pretrained weights (downloaded from download.pytorch.org).
Writes checkpoints to /usr/project/xtmp/$USER/omr_runs/tv_frcnn.

Paths assume cluster layout; adjust as needed.
"""

import os
import json
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from PIL import Image


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
    def __init__(self, ann_path, img_root):
        self.id2fname, self.anns, _ = load_coco(ann_path)
        self.ids = list(self.id2fname.keys())
        self.img_root = Path(img_root)

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
        return torchvision.transforms.functional.to_tensor(img), target


def collate(batch):
    return tuple(zip(*batch))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CocoDataset(TRAIN_JSON, IMG_ROOT)
    val_ds = CocoDataset(VAL_JSON, IMG_ROOT)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate)

    # Load COCO-pretrained detector
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Adjust classifier head for dataset classes
    num_classes = max(max((t["labels"].max().item() if len(t["labels"]) else 0) for _, t in train_ds), 114) + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for imgs, targets in train_loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ckpt_path = Path(OUT_DIR) / f"epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"saved {ckpt_path}")

    print("Done. Checkpoints in", OUT_DIR)


if __name__ == "__main__":
    main()

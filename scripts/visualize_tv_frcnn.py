"""Run inference with a TorchVision Faster R-CNN checkpoint and save PNGs with boxes.

Usage:
    python scripts/visualize_tv_frcnn.py \
        --checkpoint /usr/project/xtmp/$USER/omr_runs/tv_frcnn/epoch_4.pth \
        --num 12 --score-thr 0.3 --out /usr/project/xtmp/$USER/omr_runs/tv_frcnn/viz
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, save_image
from torchvision.ops import box_convert
from torchvision.models.detection.rpn import AnchorGenerator
from pycocotools.coco import COCO


DEFAULT_IMG_ROOT = "/home/users/jl1430/muscima-pp/v2.0/data/images"
DEFAULT_VAL_JSON = "/home/users/jl1430/jl1430/OMR-training/datasets/muscima_coco/val.json"


def make_model(num_classes: int):
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=(0.2, 0.5, 1.0, 2.0, 5.0),
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone="IMAGENET1K_V2",
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=300,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    model.rpn.batch_size_per_image = 512
    model.roi_heads.batch_size_per_image = 1024
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.6
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-root", default=DEFAULT_IMG_ROOT)
    parser.add_argument("--val-json", default=DEFAULT_VAL_JSON)
    parser.add_argument("--num", type=int, default=8, help="Number of val images to visualize")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--out", default=None, help="Output directory for PNGs (defaults beside checkpoint)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coco = COCO(args.val_json)
    num_classes = max(cat["id"] for cat in coco.cats.values()) + 1
    cat_id_to_name = {cat_id: cat["name"] for cat_id, cat in coco.cats.items()}

    model = make_model(num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    img_root = Path(args.img_root)
    ids = list(coco.imgs.keys())[: args.num]

    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for img_id in ids:
            info = coco.imgs[img_id]
            img_path = img_root / info["file_name"]
            img = torchvision.io.read_image(str(img_path))  # uint8 CxHxW
            inputs = [img.float() / 255.0]
            outputs = model([i.to(device) for i in inputs])[0]

            keep = outputs["scores"] >= args.score_thr
            boxes = outputs["boxes"][keep].cpu()
            scores = outputs["scores"][keep].cpu()
            labels = outputs["labels"][keep].cpu()

            if boxes.numel() == 0:
                annotated = img
            else:
                labels_text = [f"{cat_id_to_name.get(int(l), int(l))}:{s:.2f}" for l, s in zip(labels, scores)]
                annotated = draw_bounding_boxes(img, boxes, labels=labels_text, colors="red", width=2, font_size=12)

            out_path = out_dir / f"{Path(info['file_name']).stem}_viz.png"
            save_image(annotated.float() / 255.0, str(out_path))
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()

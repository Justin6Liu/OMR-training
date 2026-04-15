"""
Convert COCO train/val annotations into an Ultralytics YOLO detection dataset.

Example:
  python scripts/convert_coco_to_yolo.py \
    --train-json datasets/muscima_coco/train.json \
    --val-json datasets/muscima_coco/val.json \
    --images-root /home/users/jl1430/muscima-pp/v2.0/data/images \
    --out-dir outputs/muscima_yolo
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_coco(path):
    data = json.loads(Path(path).read_text())
    images = {img["id"]: img for img in data["images"]}
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)
    categories = sorted(data["categories"], key=lambda c: c["id"])
    return images, anns_by_image, categories


def normalize_bbox(bbox, width, height):
    x, y, w, h = bbox
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    ww = w / width
    hh = h / height
    return xc, yc, ww, hh


def ensure_image(src, dst, link_images):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_images:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def write_split(split_name, ann_path, images_root, out_dir, link_images):
    images, anns_by_image, categories = load_coco(ann_path)
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}

    img_dir = out_dir / split_name / "images"
    lbl_dir = out_dir / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image in images.items():
        src = Path(images_root) / image["file_name"]
        if src.suffix.lower() not in IMAGE_EXTS:
            continue
        dst = img_dir / Path(image["file_name"]).name
        ensure_image(src, dst, link_images)

        label_path = lbl_dir / f"{dst.stem}.txt"
        lines = []
        for ann in anns_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            xc, yc, ww, hh = normalize_bbox(ann["bbox"], image["width"], image["height"])
            lines.append(
                f"{cat_id_to_idx[ann['category_id']]} "
                f"{xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}"
            )
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    return categories


def write_data_yaml(out_dir, categories):
    lines = [
        f"path: {out_dir}",
        "train: train/images",
        "val: val/images",
        "names:",
    ]
    for idx, cat in enumerate(categories):
        lines.append(f"  {idx}: {cat['name']}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--link-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symlink images into the YOLO dataset instead of copying them.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = write_split("train", args.train_json, args.images_root, out_dir, args.link_images)
    write_split("val", args.val_json, args.images_root, out_dir, args.link_images)
    write_data_yaml(out_dir, categories)
    print(f"Wrote YOLO dataset to {out_dir}")
    print(f"Data YAML: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

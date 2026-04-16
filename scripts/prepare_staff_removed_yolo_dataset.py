"""
Build a YOLO detection dataset for staff-removed MUSCIMA++ images using a
cleaned match JSON produced from reconstructed GT pairings.

Each record in the match JSON is expected to contain:
  - target_image: original MUSCIMA++ page path
  - matched_gt: paired staff-removed image path

The annotations are taken from an existing COCO export for the original
MUSCIMA++ images, while the output images come from the paired staff-removed
files.

Example:
  python scripts/prepare_staff_removed_yolo_dataset.py \
    --match-json outputs/muscima_gt_match_filtered_clean.json \
    --coco-jsons datasets/muscima_coco/train.json datasets/muscima_coco/val.json \
    --out-dir outputs/muscima_staff_removed_yolo \
    --val-ratio 0.2
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-json", required=True, help="Cleaned match JSON.")
    ap.add_argument(
        "--coco-jsons",
        nargs="+",
        required=True,
        help="One or more COCO JSONs covering the original MUSCIMA++ images.",
    )
    ap.add_argument("--out-dir", required=True, help="Output YOLO dataset directory.")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    ap.add_argument(
        "--link-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symlink staff-removed images instead of copying them.",
    )
    return ap.parse_args()


def load_coco_sources(paths):
    images_by_name = {}
    anns_by_image = defaultdict(list)
    categories = None

    for path_str in paths:
        path = Path(path_str)
        data = json.loads(path.read_text())
        current_categories = sorted(data["categories"], key=lambda c: c["id"])

        if categories is None:
            categories = current_categories
        elif current_categories != categories:
            raise ValueError(f"Category mismatch in COCO file: {path}")

        for image in data["images"]:
            name = Path(image["file_name"]).name
            if name in images_by_name:
                raise ValueError(f"Duplicate image name across COCO files: {name}")
            images_by_name[name] = image

        for ann in data.get("annotations", []):
            anns_by_image[ann["image_id"]].append(ann)

    if categories is None:
        raise ValueError("No categories found in COCO inputs.")

    return images_by_name, anns_by_image, categories


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


def write_labels(label_path, anns, src_width, src_height, cat_id_to_idx):
    lines = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        xc, yc, ww, hh = normalize_bbox(ann["bbox"], src_width, src_height)
        lines.append(
            f"{cat_id_to_idx[ann['category_id']]} "
            f"{xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


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
    args = parse_args()
    random.seed(args.seed)

    images_by_name, anns_by_image, categories = load_coco_sources(args.coco_jsons)
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
    match_records = json.loads(Path(args.match_json).read_text())

    examples = []
    seen_names = set()
    missing = []

    for record in match_records:
        target_name = Path(record["target_image"]).name
        staff_removed_path = Path(record["matched_gt"]).resolve()

        if target_name in seen_names:
            raise ValueError(f"Duplicate target image in match JSON: {target_name}")
        seen_names.add(target_name)

        image_info = images_by_name.get(target_name)
        if image_info is None:
            missing.append(target_name)
            continue

        if not staff_removed_path.exists():
            raise FileNotFoundError(f"Missing staff-removed image: {staff_removed_path}")

        with Image.open(staff_removed_path) as im:
            staff_width, staff_height = im.size

        if (staff_width, staff_height) != (image_info["width"], image_info["height"]):
            raise ValueError(
                "Dimension mismatch for "
                f"{target_name}: original={(image_info['width'], image_info['height'])} "
                f"staff_removed={(staff_width, staff_height)}"
            )

        examples.append(
            {
                "target_name": target_name,
                "staff_removed_path": staff_removed_path,
                "image_info": image_info,
                "annotations": anns_by_image.get(image_info["id"], []),
            }
        )

    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"{len(missing)} matched images were not found in the provided COCO JSONs. "
            f"Examples: {preview}"
        )

    if not examples:
        raise ValueError("No matched examples available after loading inputs.")

    random.shuffle(examples)
    val_count = max(1, int(round(len(examples) * args.val_ratio))) if len(examples) > 1 else 0
    val_names = {item["target_name"] for item in examples[:val_count]}

    out_dir = Path(args.out_dir)
    for split_name in ("train", "val"):
        (out_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    manifest = []
    split_counts = {"train": 0, "val": 0}

    for item in examples:
        split_name = "val" if item["target_name"] in val_names else "train"
        image_dst = out_dir / split_name / "images" / item["target_name"]
        label_dst = out_dir / split_name / "labels" / f"{Path(item['target_name']).stem}.txt"

        ensure_image(item["staff_removed_path"], image_dst, args.link_images)
        write_labels(
            label_dst,
            item["annotations"],
            item["image_info"]["width"],
            item["image_info"]["height"],
            cat_id_to_idx,
        )

        manifest.append(
            {
                "split": split_name,
                "target_name": item["target_name"],
                "staff_removed_path": str(item["staff_removed_path"]),
                "annotation_count": len(item["annotations"]),
            }
        )
        split_counts[split_name] += 1

    write_data_yaml(out_dir, categories)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote YOLO dataset to {out_dir}")
    print(f"train images: {split_counts['train']}")
    print(f"val images: {split_counts['val']}")
    print(f"data yaml: {out_dir / 'data.yaml'}")
    print(f"manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

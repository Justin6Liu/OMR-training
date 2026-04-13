"""
Quick MUSCIMA++ XML -> COCO converter (detection only).

Assumptions:
- Image files in one folder, named like CVC-MUSCIMA_W-01_N-10_D-ideal.png
- Matching XML in annotations folder with same stem and .xml extension.
- XML contains <cropObject> nodes with attributes: id, top, left, height, width, className.

Outputs:
- COCO JSON with all categories found.
- Simple train/val split by sorted file list (configurable ratio).

Usage:
  python convert_muscima_to_coco.py \
    --images /path/to/images \
    --annotations /path/to/xmls \
    --output /path/to/coco.json \
    --val-ratio 0.2
"""

import argparse
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Directory with MUSCIMA++ images (png).")
    ap.add_argument("--annotations", required=True, help="Directory with MUSCIMA++ XMLs.")
    ap.add_argument("--output-train", required=True, help="Output COCO train json.")
    ap.add_argument("--output-val", required=True, help="Output COCO val json.")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_annotations(img_path: Path, xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    anns = []
    # MUSCIMA++ Node format: each <Node> has children ClassName, Top, Left, Width, Height
    for obj in root.findall("./Node"):
        cls_el = obj.find("ClassName")
        top_el = obj.find("Top")
        left_el = obj.find("Left")
        w_el = obj.find("Width")
        h_el = obj.find("Height")
        if None in (cls_el, top_el, left_el, w_el, h_el):
            continue
        cls = cls_el.text.strip()
        try:
            top = float(top_el.text)
            left = float(left_el.text)
            width = float(w_el.text)
            height = float(h_el.text)
        except (TypeError, ValueError):
            continue
        bbox = [left, top, width, height]
        anns.append((cls, bbox))
    return anns


def main():
    args = parse_args()
    random.seed(args.seed)

    img_dir = Path(args.images)
    ann_dir = Path(args.annotations)

    images = sorted([p for p in img_dir.glob("*.png")])
    assert images, "No images found."

    # gather categories
    cat_set = set()
    img_records = []
    ann_records = []
    ann_id = 1
    for img_id, img_path in enumerate(images, start=1):
        stem = img_path.stem
        xml_path = ann_dir / f"{stem}.xml"
        if not xml_path.exists():
            print(f"skip {stem}: no xml")
            continue
        anns = load_annotations(img_path, xml_path)
        width, height = None, None
        # lazy read dimensions via PIL only when needed
        from PIL import Image
        with Image.open(img_path) as im:
            width, height = im.size
        img_records.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )
        for cls, bbox in anns:
            cat_set.add(cls)
            ann_records.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "bbox": bbox,
                    "category_id": cls,  # temporarily store name
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": i + 1, "name": c} for i, c in enumerate(sorted(cat_set))]
    name_to_id = {c["name"]: c["id"] for c in categories}
    for ann in ann_records:
        ann["category_id"] = name_to_id[ann["category_id"]]

    # split train/val
    random.shuffle(img_records)
    val_size = int(len(img_records) * args.val_ratio)
    val_ids = set([r["id"] for r in img_records[:val_size]])
    train_ids = set([r["id"] for r in img_records[val_size:]])

    def dump(json_path, keep_ids):
        imgs = [r for r in img_records if r["id"] in keep_ids]
        anns = [a for a in ann_records if a["image_id"] in keep_ids]
        coco = {
            "images": imgs,
            "annotations": anns,
            "categories": categories,
        }
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        print(f"wrote {json_path} images={len(imgs)} anns={len(anns)} cats={len(categories)}")

    dump(args.output_val, val_ids)
    dump(args.output_train, train_ids)


if __name__ == "__main__":
    main()

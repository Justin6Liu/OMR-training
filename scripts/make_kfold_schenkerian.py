"""
Builds a K-fold YOLO-format dataset from the 5 annotated pages in
/Users/justinliu/Documents/GitHub/Schenkerian_OMR/Complete Annotations.

What it does:
- Finds all mung.xml files and paired images (image.png/image.jpg) under the source root.
- Parses bounding boxes from MuNG (Left, Top, Width, Height) and class names.
- Normalizes to YOLO txt labels and writes per-fold train/val splits.
- Emits a data YAML per fold for ultralytics YOLO training.

Usage (from repo root):
    python scripts/make_kfold_schenkerian.py \
        --data-root /Users/justinliu/Documents/GitHub/Schenkerian_OMR/Complete\\ Annotations \
        --out-dir outputs/schenkerian_kfold \
        --k 5

After generation, train a fold with:
    pip install ultralytics  # if not already
    yolo train model=yolov8l.pt data=outputs/schenkerian_kfold/fold_0/data.yaml epochs=300 imgsz=960

Notes:
- Images are copied/converted to PNG; labels to YOLO txt.
- Class list is derived from the 5 MuNG files and shared across folds.
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

from PIL import Image
import xml.etree.ElementTree as ET


def find_docs(data_root: Path) -> List[Tuple[str, Path, Path]]:
    docs = []
    for xml_path in data_root.glob("**/mung.xml"):
        doc_name = xml_path.parent.name
        # Prefer PNG, fall back to JPG
        img_path = xml_path.parent / "image.png"
        if not img_path.exists():
            img_path = xml_path.parent / "image.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"No image.png or image.jpg next to {xml_path}")
        docs.append((doc_name, xml_path, img_path))
    return sorted(docs, key=lambda t: t[0])


def parse_mung(xml_path: Path) -> List[Dict]:
    root = ET.parse(xml_path).getroot()
    nodes = []
    for node_el in root.findall("Node"):
        cls = node_el.findtext("ClassName")
        top = float(node_el.findtext("Top"))
        left = float(node_el.findtext("Left"))
        w = float(node_el.findtext("Width"))
        h = float(node_el.findtext("Height"))
        if w <= 0 or h <= 0:
            continue
        nodes.append({"class": cls, "left": left, "top": top, "width": w, "height": h})
    return nodes


def write_yolo_label(out_txt: Path, boxes: List[Dict], class_to_id: Dict[str, int], img_w: int, img_h: int):
    with out_txt.open("w") as f:
        for b in boxes:
            xc = (b["left"] + b["width"] / 2) / img_w
            yc = (b["top"] + b["height"] / 2) / img_h
            ww = b["width"] / img_w
            hh = b["height"] / img_h
            f.write(f"{class_to_id[b['class']]} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")


def save_yaml(names: Dict[int, str], out_path: Path, base_dir: Path):
    lines = [
        f"path: {base_dir}",
        "train: train/images",
        "val: val/images",
        "names:",
    ]
    for k in sorted(names.keys()):
        lines.append(f"  {k}: {names[k]}")
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to 'Complete Annotations' directory")
    ap.add_argument("--out-dir", default="outputs/schenkerian_kfold", help="Output root for folds")
    ap.add_argument("--k", type=int, default=5, help="Number of folds (<= number of docs)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    docs = find_docs(data_root)
    if len(docs) < args.k:
        raise ValueError(f"Requested {args.k} folds but only found {len(docs)} documents.")

    # Collect all classes
    class_names = {}
    for _, xml_path, _ in docs:
        for b in parse_mung(xml_path):
            class_names[b["class"]] = True
    class_list = sorted(class_names.keys())
    class_to_id = {c: i for i, c in enumerate(class_list)}
    id_to_class = {i: c for c, i in class_to_id.items()}

    # Build folds (round-robin assignment of val doc)
    for fold_idx in range(args.k):
        fold_dir = out_root / f"fold_{fold_idx}"
        for split in ["train/images", "train/labels", "val/images", "val/labels"]:
            (fold_dir / split).mkdir(parents=True, exist_ok=True)

        for doc_idx, (doc_name, xml_path, img_path) in enumerate(docs):
            split = "val" if doc_idx % args.k == fold_idx else "train"
            # Load boxes
            boxes = parse_mung(xml_path)
            # Copy/convert image to PNG
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                out_img = fold_dir / split / "images" / f"{doc_name}.png"
                img.save(out_img)
                img_w, img_h = img.size
            # Write labels
            out_lbl = fold_dir / split / "labels" / f"{doc_name}.txt"
            write_yolo_label(out_lbl, boxes, class_to_id, img_w, img_h)

        # Write YAML
        save_yaml(id_to_class, fold_dir / "data.yaml", fold_dir)
        print(f"Fold {fold_idx}: train/val prepared at {fold_dir}")

    print(f"Done. Classes ({len(class_list)}): {class_list}")


if __name__ == "__main__":
    main()


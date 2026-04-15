"""
Match MUSCIMA++ page images to flat CVC-MUSCIMA staff-removed GT_XXXX images.

This is useful when you only have the flat `GT_XXXX.png` export rather than the
original hierarchical CVC-MUSCIMA directory tree expected by
`get_images_from_muscima.py`.

The script computes a simple image-distance score on resized grayscale pages and
greedily assigns each target image the best unused GT image.

Example:
  python scripts/match_muscima_staff_removed.py \
    --targets /Users/justinliu/Documents/GitHub/muscima-pp/v2.0/data/images \
    --gt-root /Users/justinliu/Downloads/Training_GT \
    --out-json outputs/muscima_gt_match.json \
    --out-dir outputs/muscima_staff_removed_images \
    --link
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_resized_gray(path: Path, size):
    img = Image.open(path).convert("L").resize(size)
    return np.asarray(img, dtype=np.float32)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def collect_images(root: Path, pattern: str):
    return sorted(root.glob(pattern))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="Directory of MUSCIMA++ target images.")
    ap.add_argument("--gt-root", required=True, help="Directory containing flat GT_XXXX.png images.")
    ap.add_argument("--target-pattern", default="*.png")
    ap.add_argument("--gt-pattern", default="GT_*.png")
    ap.add_argument("--resize", default="256,256", help="Resize used for matching, e.g. 256,256")
    ap.add_argument("--topk", type=int, default=20, help="How many candidate matches to score per target output preview.")
    ap.add_argument("--out-json", required=True, help="Where to save the final mapping JSON.")
    ap.add_argument("--out-dir", help="Optional output dir to populate with matched staff-removed images.")
    ap.add_argument(
        "--link",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symlink matches into out-dir instead of copying them.",
    )
    args = ap.parse_args()

    resize = tuple(int(x) for x in args.resize.split(","))
    targets = collect_images(Path(args.targets), args.target_pattern)
    gt_images = collect_images(Path(args.gt_root), args.gt_pattern)

    if not targets:
        raise FileNotFoundError(f"No target images found in {args.targets}")
    if not gt_images:
        raise FileNotFoundError(f"No GT images found in {args.gt_root}")

    gt_cache = {p: load_resized_gray(p, resize) for p in tqdm(gt_images, desc="Caching GT")}
    used_gt = set()
    matches = []

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for target_path in tqdm(targets, desc="Matching targets"):
        target_arr = load_resized_gray(target_path, resize)
        scored = []
        for gt_path, gt_arr in gt_cache.items():
            if gt_path in used_gt:
                continue
            scored.append((mse(target_arr, gt_arr), gt_path))
        scored.sort(key=lambda x: x[0])
        best_score, best_gt = scored[0]
        used_gt.add(best_gt)

        record = {
            "target_image": str(target_path),
            "matched_gt": str(best_gt),
            "score": best_score,
            "top_candidates": [
                {"path": str(path), "score": score}
                for score, path in scored[: args.topk]
            ],
        }
        matches.append(record)

        if out_dir:
            dst = out_dir / target_path.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.link:
                os.symlink(best_gt, dst)
            else:
                Image.open(best_gt).save(dst)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matches, indent=2))
    print(f"Saved {len(matches)} matches to {out_path}")
    if out_dir:
        print(f"Populated {out_dir} with matched staff-removed images")


if __name__ == "__main__":
    main()

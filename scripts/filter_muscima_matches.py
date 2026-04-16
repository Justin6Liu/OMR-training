"""
Filter a MUSCIMA match JSON by dropping selected record indices and optionally
rebuild the symlinked staff-removed image folder.

Example:
  python scripts/filter_muscima_matches.py \
    --match-json /Users/justinliu/Desktop/muscima_gt_match_filtered_1000_2000.json \
    --drop-indices 8,12,19,22,37,42,54,65,77,80,84,88,96,98 \
    --out-json /Users/justinliu/Desktop/muscima_gt_match_filtered_clean.json \
    --out-dir /Users/justinliu/Desktop/muscima_staff_removed_images_filtered_clean
"""

import argparse
import json
import os
from pathlib import Path


def parse_indices(spec: str):
    return {int(x.strip()) for x in spec.split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-json", required=True)
    ap.add_argument("--drop-indices", required=True, help="Comma-separated 0-based indices to remove.")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-dir", help="Optional directory to populate with symlinks for surviving matches.")
    args = ap.parse_args()

    drop = parse_indices(args.drop_indices)
    data = json.loads(Path(args.match_json).read_text())
    kept = [item for idx, item in enumerate(data) if idx not in drop]

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(kept, indent=2))
    print(f"Kept {len(kept)} / {len(data)} matches")
    print(f"Saved filtered JSON to {out_json}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.iterdir():
            if old.is_symlink() or old.is_file():
                old.unlink()
        for item in kept:
            target_name = Path(item["target_image"]).name
            matched_gt = Path(item["matched_gt"])
            dst = out_dir / target_name
            os.symlink(matched_gt, dst)
        print(f"Rebuilt symlink folder at {out_dir}")


if __name__ == "__main__":
    main()

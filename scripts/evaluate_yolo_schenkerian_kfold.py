"""
Evaluate a YOLO checkpoint over Schenkerian k-fold splits and report per-fold metrics.

Example:
  python scripts/evaluate_yolo_schenkerian_kfold.py \
    --model /home/users/jl1430/jl1430/Schenkerian_OMR/trained_models/yolo11l_muscima.pt \
    --fold-root outputs/schenkerian_kfold \
    --k 5 \
    --imgsz 960 \
    --device 0
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Checkpoint to evaluate.")
    ap.add_argument("--fold-root", required=True, help="Directory containing fold_*/data.yaml.")
    ap.add_argument("--k", type=int, default=5, help="Number of folds to evaluate.")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project", help="Optional Ultralytics project dir for val outputs.")
    ap.add_argument("--name-prefix", default="eval_fold")
    ap.add_argument("--out-json", help="Optional path to save fold metrics as JSON.")
    args = ap.parse_args()

    fold_root = Path(args.fold_root)
    model = YOLO(args.model)
    results = []

    for fold_idx in range(args.k):
        data_yaml = fold_root / f"fold_{fold_idx}" / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Missing {data_yaml}")

        kwargs = dict(
            data=str(data_yaml),
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            split="val",
            verbose=False,
        )
        if args.project:
            kwargs["project"] = args.project
            kwargs["name"] = f"{args.name_prefix}_{fold_idx}"

        metrics = model.val(**kwargs)
        fold_result = {
            "fold": fold_idx,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        results.append(fold_result)
        print(
            f"fold {fold_idx}: "
            f"P={fold_result['precision']:.4f} "
            f"R={fold_result['recall']:.4f} "
            f"mAP50={fold_result['map50']:.4f} "
            f"mAP50-95={fold_result['map50_95']:.4f}"
        )

    summary = {
        "model": args.model,
        "folds": results,
        "mean_map50": sum(r["map50"] for r in results) / len(results),
        "mean_map50_95": sum(r["map50_95"] for r in results) / len(results),
        "mean_precision": sum(r["precision"] for r in results) / len(results),
        "mean_recall": sum(r["recall"] for r in results) / len(results),
    }

    print(
        "mean: "
        f"P={summary['mean_precision']:.4f} "
        f"R={summary['mean_recall']:.4f} "
        f"mAP50={summary['mean_map50']:.4f} "
        f"mAP50-95={summary['mean_map50_95']:.4f}"
    )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"saved summary to {out_path}")


if __name__ == "__main__":
    main()

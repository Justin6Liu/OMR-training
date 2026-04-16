"""
Build side-by-side visual QA sheets for MUSCIMA++ original pages and matched
staff-removed images.

Example:
  python scripts/visualize_muscima_pairings.py \
    --match-json /Users/justinliu/Desktop/muscima_gt_match_filtered_1000_2000.json \
    --out-dir /Users/justinliu/Desktop/muscima_pair_review \
    --limit 108
"""

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw


def make_side_by_side(left_path: Path, right_path: Path, out_path: Path, max_height: int):
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    left = ImageOps.contain(left, (10_000, max_height))
    right = ImageOps.contain(right, (10_000, max_height))

    gap = 24
    header_h = 42
    canvas = Image.new(
        "RGB",
        (left.width + right.width + gap, max(left.height, right.height) + header_h),
        "white",
    )
    canvas.paste(left, (0, header_h))
    canvas.paste(right, (left.width + gap, header_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), f"Original: {left_path.name}", fill="black")
    draw.text((left.width + gap + 8, 10), f"Matched GT: {right_path.name}", fill="black")
    draw.line((left.width + gap // 2, 0, left.width + gap // 2, canvas.height), fill=(180, 180, 180), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_index(records, out_dir: Path):
    rows = []
    for rec in records:
        rel = rec["comparison_file"]
        rows.append(
            f"<tr>"
            f"<td>{html.escape(rec['target_name'])}</td>"
            f"<td>{html.escape(rec['matched_name'])}</td>"
            f"<td>{rec['score']:.2f}</td>"
            f"<td><a href='{html.escape(rel)}'>open</a></td>"
            f"</tr>"
        )

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MUSCIMA Pair Review</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    th {{ background: #f3f3f3; }}
  </style>
</head>
<body>
  <h1>MUSCIMA Pair Review</h1>
  <p>Generated {len(records)} side-by-side comparisons.</p>
  <table>
    <thead>
      <tr><th>Original</th><th>Matched GT</th><th>Score</th><th>Comparison</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-json", required=True, help="JSON produced by match_muscima_staff_removed.py")
    ap.add_argument("--out-dir", required=True, help="Directory for side-by-side sheets and HTML index")
    ap.add_argument("--max-height", type=int, default=900, help="Max rendered height for each page")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of comparisons")
    args = ap.parse_args()

    data = json.loads(Path(args.match_json).read_text())
    if args.limit > 0:
        data = data[: args.limit]

    out_dir = Path(args.out_dir)
    comp_dir = out_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, item in enumerate(data):
        target = Path(item["target_image"])
        matched = Path(item["matched_gt"])
        out_file = comp_dir / f"{idx:03d}_{target.stem}.png"
        make_side_by_side(target, matched, out_file, args.max_height)
        records.append(
            {
                "target_name": target.name,
                "matched_name": matched.name,
                "score": float(item.get("score", -1)),
                "comparison_file": str(Path("comparisons") / out_file.name),
            }
        )

    write_index(records, out_dir)
    print(f"Generated {len(records)} comparisons in {out_dir}")
    print(f"Open {out_dir / 'index.html'} in a browser to review pairs quickly")


if __name__ == "__main__":
    main()

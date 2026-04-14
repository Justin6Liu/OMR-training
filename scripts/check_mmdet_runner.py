#!/usr/bin/env python3
"""Build an MMDetection runner and write the full traceback to a file on failure."""

import argparse
import sys
import traceback
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--error-file",
        default="/tmp/mmdet_runner_build.err",
        help="Where to write the full traceback if runner build fails.",
    )
    args = parser.parse_args()

    error_path = Path(args.error_file)

    try:
        cfg = Config.fromfile(args.config)
        print("config loaded")
        Runner.from_cfg(cfg)
        print("runner built")
        return 0
    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb)
        last_line = tb.strip().splitlines()[-1] if tb.strip() else "Unknown error"
        print(f"runner build failed: {last_line}", file=sys.stderr)
        print(f"full traceback written to {error_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

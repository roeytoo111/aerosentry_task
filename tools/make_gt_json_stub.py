#!/usr/bin/env python3
"""Create a per-frame GT JSON stub (empty boxes) aligned with a video's frame count.

Edit the output file to fill ``xyxy`` / ``class_ids`` for each frame (pixel coordinates),
or use it only to verify tooling. Schema matches ``tools/benchmark_video_fp_compare.py``.

Example::

    PYTHONPATH=. python3 tools/make_gt_json_stub.py \\
      --video \"Video Analytics/Test Footage/Arsuf F1 09_04_2025 - Made with Clipchamp.mp4\" \\
      --out outputs/arsuf_gt_stub.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    args = p.parse_args()

    vid = args.video.expanduser().resolve()
    if not vid.is_file():
        print(f"Video not found: {vid}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        print(f"Could not open: {vid}", file=sys.stderr)
        return 2
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    if n <= 0:
        print("Frame count unknown or zero; using 1 empty frame entry.", file=sys.stderr)
        n = 1

    payload = {
        "iou_threshold": float(args.iou_threshold),
        "frames": [{"xyxy": [], "class_ids": []} for _ in range(n)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {n} empty frame entries to {args.out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a trained YOLO checkpoint on a video with optional FP suppressor (real-time style loop)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from src.core.data_contracts import Detection, FrameData
from src.evaluation.tactical_evaluator import UltralyticsYoloDetector
from src.tracking.fp_suppressor import FalsePositiveSuppressor


def _opencv_gui_available() -> bool:
    """``opencv-python-headless`` builds omit HighGUI — ``imshow`` is not implemented."""
    try:
        cv2.namedWindow("__aerosentry_gui_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__aerosentry_gui_probe__")
        return True
    except cv2.error:
        return False


def _safe_destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def _xywhn_to_xyxy_pixel(d: Detection, w: int, h: int) -> Tuple[int, int, int, int]:
    cx, cy, bw, bh = (float(d.bbox[i]) for i in range(4))
    x1 = int(round((cx - bw / 2.0) * w))
    y1 = int(round((cy - bh / 2.0) * h))
    x2 = int(round((cx + bw / 2.0) * w))
    y2 = int(round((cy + bh / 2.0) * h))
    return x1, y1, x2, y2


def _draw(
    frame_bgr: np.ndarray,
    dets: List[Detection],
    *,
    thickness: int = 2,
) -> np.ndarray:
    out = frame_bgr
    h, w = out.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = _xywhn_to_xyxy_pixel(d, w, h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        label = f"{d.class_label}"
        if d.track_id is not None:
            label += f" id={d.track_id}"
        label += f" {d.confidence:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), thickness)
        cv2.putText(
            out,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True, help="Video file path (mp4, avi, …).")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--fp-suppressor",
        action="store_true",
        help="Apply FalsePositiveSuppressor after each frame (temporal / ego gate).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write annotated video here (e.g. out.mp4). Recommended if no display.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show live window (needs opencv with GUI, not headless; or use --out).",
    )
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = entire clip).")
    p.add_argument(
        "--progress-every",
        type=int,
        default=30,
        help="Print progress every N frames (0 = disable periodic progress).",
    )
    p.add_argument(
        "--progress-seconds",
        type=float,
        default=3.0,
        help="Also print if this many seconds passed since last log (0 = time-based off).",
    )
    p.add_argument("--quiet", action="store_true", help="No progress lines (only final summary).")
    p.add_argument(
        "--debug-detections",
        action="store_true",
        help="On each progress line, print raw YOLO count + max conf and count after FP gate (if any).",
    )
    p.add_argument(
        "--geo-debug",
        action="store_true",
        help="When using --fp-suppressor, print [GeometricEgoMotion] / skip logs (or set AEROSENTRY_GEO_DEBUG=1).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if getattr(args, "geo_debug", False):
        os.environ["AEROSENTRY_GEO_DEBUG"] = "1"
    terminal_dbg = args.debug_detections and not args.quiet
    if not args.show and args.out is None and not terminal_dbg:
        print("Specify --show and/or --out (otherwise there is no output).", flush=True)
        return 2
    if not args.source.is_file():
        print(f"Video not found: {args.source}", flush=True)
        return 2

    use_show = bool(args.show)
    if use_show and not _opencv_gui_available():
        print(
            "OpenCV was built without GUI support (typical with opencv-python-headless).\n"
            "  • Save video: add  --out outputs/run.mp4  (works with headless)\n"
            "  • Or: pip install opencv-python  (replaces headless; needs Qt/GTK on Linux)",
            flush=True,
        )
        use_show = False
        if args.out is None:
            return 2

    det = UltralyticsYoloDetector(
        args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
    )
    suppressor: Optional[FalsePositiveSuppressor] = (
        FalsePositiveSuppressor() if args.fp_suppressor else None
    )

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        print(f"Could not open video: {args.source}", flush=True)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total < 1:
        n_total = 0

    dur_min = (n_total / fps / 60.0) if n_total and fps else 0.0
    meta = f"{w}x{h} @ {fps:.2f} FPS"
    if n_total:
        meta += f"  ~{n_total} frames (~{dur_min:.1f} min source)"
    else:
        meta += "  (total frame count unknown)"
    print(f"Video: {meta}", flush=True)
    print(
        "Running inference — progress prints every "
        f"{args.progress_every} frame(s) or every {args.progress_seconds:g}s "
        "(first batch can be slow). Ctrl+C stops early; partial file is still closed.",
        flush=True,
    )

    writer: Optional[cv2.VideoWriter] = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))
        if not writer.isOpened():
            print("VideoWriter failed; try a different --out suffix or check codecs.", flush=True)
            cap.release()
            return 2

    frame_id = 0
    t0 = time.perf_counter()
    t_last_log = t0
    interrupted = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            ts = time.perf_counter()
            preds = det.predict(frame)
            n_raw = len(preds)
            max_raw = max((float(p.confidence) for p in preds), default=0.0)
            fd = FrameData(
                frame=frame,
                frame_id=frame_id,
                timestamp=ts,
                detections=list(preds),
            )
            if suppressor is not None:
                fd = suppressor.process(fd)
            n_out = len(fd.detections)
            vis = _draw(frame, fd.detections)

            if writer is not None:
                writer.write(vis)
            if use_show:
                cv2.imshow("aerosentry_infer", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            if not args.quiet:
                by_frame = args.progress_every > 0 and frame_id % args.progress_every == 0
                by_time = args.progress_seconds > 0 and (ts - t_last_log) >= args.progress_seconds
                if by_frame or by_time:
                    elapsed = ts - t0
                    rate = frame_id / max(elapsed, 1e-6)
                    tail = ""
                    if n_total and rate > 0:
                        eta_s = (n_total - frame_id) / rate
                        pct = 100.0 * frame_id / n_total
                        tail = f"  {pct:.1f}%  ETA ~{eta_s/60:.1f} min"
                    line = (
                        f"  frame {frame_id}"
                        + (f"/{n_total}" if n_total else "")
                        + f"  {rate:.2f} FPS (processing){tail}"
                    )
                    if args.debug_detections:
                        line += (
                            f"  | raw_dets={n_raw} max_conf={max_raw:.3f} after_fp={n_out}"
                        )
                    print(line, flush=True)
                    t_last_log = ts

            if args.max_frames and frame_id >= args.max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nStopped by user after {frame_id} frames.", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if use_show:
            _safe_destroy_windows()

    elapsed = time.perf_counter() - t0
    print(
        f"Processed {frame_id} frames in {elapsed:.2f}s "
        f"({frame_id / max(elapsed, 1e-6):.2f} FPS effective ingest+infer).",
        flush=True,
    )
    if args.out:
        print(f"Wrote {args.out.resolve()}", flush=True)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())

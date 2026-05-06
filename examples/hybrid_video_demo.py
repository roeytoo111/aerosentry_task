#!/usr/bin/env python3
"""Demonstration harness for :class:`~src.models.hybrid_detector.HybridDetector`.

Shows the intended data flow: VideoCapture → hybrid ``detect`` → optional
:class:`~src.tracking.fp_suppressor.FalsePositiveSuppressor`` → lightweight tracker shim.
Not used by production ``run.py``; copy patterns into your loop.

Example with full FP layer (TrackManager + geometry), same as ``run.py infer --fp-suppressor``::

    PYTHONPATH=. python3 examples/hybrid_video_demo.py \\
      --video path/to/clip.mp4 --yolo runs/.../best.pt --rtdetr rtdetr-l.pt \\
      --out outputs/hybrid_fp.mp4 --max-frames 0 --fp-suppressor

With ``--gt-json``, prints a **micro-aggregated TP/FP/FN** summary at the end, comparing
**cascaded hybrid** vs **YOLO-only** on the same frames (same IoU rules as
``tools/benchmark_video_fp_compare.py``). GT must list one entry per frame in order; see that
module's docstring for the JSON schema.

Example::

    PYTHONPATH=. python3 examples/hybrid_video_demo.py \\
      --video path/to/clip.mp4 \\
      --yolo yolo11n.pt \\
      --rtdetr rtdetr-l.pt

Example with annotated video (no labels needed)::

    PYTHONPATH=. python3 examples/hybrid_video_demo.py \\
      --video path/to/clip.mp4 --yolo runs/.../best.pt --rtdetr rtdetr-l.pt \\
      --out outputs/hybrid_annotated.mp4 --max-frames 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.data_contracts import Detection, FrameData
from src.models.evaluate_detector import match_image
from src.models.hybrid_detector import HybridDetector, xyxy6_to_detections
from src.tracking.fp_config import (
    build_false_positive_suppressor_from_mapping,
    load_tracking_fp_yaml_dict,
    parse_hybrid_demo_mock_tracker_miss_threshold,
    parse_hybrid_demo_yolo_conf,
    resolve_tracking_fp_yaml,
)
from src.tracking.fp_suppressor import FalsePositiveSuppressor
from tools.benchmark_video_fp_compare import load_gt_json


class MockTracker:
    """Minimal stand-in: infer *track loss* from consecutive empty detection arrays."""

    def __init__(self, *, miss_threshold: int = 3) -> None:
        self._miss = 0
        self._miss_threshold = int(miss_threshold)

    def update(self, _frame_id: int, dets_xyxy6: np.ndarray) -> None:
        if dets_xyxy6.shape[0] == 0:
            self._miss += 1
        else:
            self._miss = 0

    def is_losing_track(self) -> bool:
        """True when the mock tracker has seen too many empty returns in a row."""
        return self._miss >= self._miss_threshold


def _detections_to_xyxy6(dets: List[Detection], w: int, h: int) -> np.ndarray:
    """Pixel ``(N,6)`` from normalized :class:`Detection` list (inverse of ``xyxy6_to_detections``)."""
    if not dets:
        return np.zeros((0, 6), dtype=np.float32)
    rows: List[List[float]] = []
    for d in dets:
        cx, cy, bw, bh = (float(d.bbox[i]) for i in range(4))
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        rows.append([x1, y1, x2, y2, float(d.confidence), float(d.class_id)])
    return np.asarray(rows, dtype=np.float32)


def _opencv_gui_available() -> bool:
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


def _draw_xyxy6_overlay(
    frame_bgr: np.ndarray,
    dets: np.ndarray,
    *,
    used_rtdetr: bool,
    state_name: str,
    cooldown: int,
    fp_applied: bool,
    thickness: int = 2,
) -> np.ndarray:
    """Draw ``(N,6)`` pixel boxes and a small mode banner (YOLO vs RT-DETR path)."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    banner = f"{'RT-DETR' if used_rtdetr else 'YOLO'}  |  {state_name}  cd={cooldown}"
    if fp_applied:
        banner += "  |  FP"
    cv2.putText(
        out,
        banner,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255) if used_rtdetr else (0, 200, 0),
        2,
        cv2.LINE_AA,
    )
    if dets.size == 0:
        return out
    for i in range(dets.shape[0]):
        x1, y1, x2, y2, conf, cid = (float(v) for v in dets[i])
        xi1 = int(max(0, min(w - 1, round(x1))))
        yi1 = int(max(0, min(h - 1, round(y1))))
        xi2 = int(max(0, min(w - 1, round(x2))))
        yi2 = int(max(0, min(h - 1, round(y2))))
        color = (255, 128, 0) if used_rtdetr else (0, 165, 255)
        cv2.rectangle(out, (xi1, yi1), (xi2, yi2), color, thickness)
        lab = f"cls={int(round(cid))} {conf:.2f}"
        cv2.putText(
            out,
            lab,
            (xi1, max(18, yi1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _get_gt_frame(
    gt_list: list[Tuple[np.ndarray, np.ndarray]], idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    if idx < len(gt_list):
        return gt_list[idx]
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )


def _split_xyxy6(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``(N,6)`` into xyxy, class ids, confidences for :func:`match_image`."""
    if arr.size == 0:
        z = np.zeros((0, 4), dtype=np.float32)
        return z, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return (
        arr[:, :4].astype(np.float32),
        arr[:, 5].astype(np.int64),
        arr[:, 4].astype(np.float32),
    )


def _micro_prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    r = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--yolo", type=str, default="yolo11n.pt")
    p.add_argument("--rtdetr", type=str, default="rtdetr-l.pt")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--yolo-conf",
        type=float,
        default=None,
        metavar="CONF",
        help="YOLO confidence threshold (default: hybrid_demo_detector.yolo_conf in YAML, else 0.25).",
    )
    p.add_argument("--uncertainty-thresh", type=float, default=0.55)
    p.add_argument("--cooldown", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=120, help="0 = entire clip.")
    p.add_argument(
        "--gt-json",
        type=Path,
        default=None,
        help="Per-frame GT (pixel xyxy); if set, prints TP/FP/FN vs YOLO-only at end.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write annotated video (BGR boxes + YOLO/RT-DETR banner).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show a live window (needs OpenCV with GUI; use --out on headless).",
    )
    fp_grp = p.add_mutually_exclusive_group()
    fp_grp.add_argument(
        "--fp-suppressor",
        action="store_true",
        help="Apply FalsePositiveSuppressor after hybrid (TrackManager + geometric gate).",
    )
    fp_grp.add_argument(
        "--fp-geo-only",
        action="store_true",
        help="Geometry-only FP gate (no TrackManager); ablation vs full --fp-suppressor.",
    )
    p.add_argument(
        "--geo-debug",
        action="store_true",
        help="Verbose geometric gate logs (sets AEROSENTRY_GEO_DEBUG=1).",
    )
    p.add_argument(
        "--fp-config",
        type=Path,
        default=None,
        help=(
            "YAML: tracking_fp + hybrid_demo_* sections. "
            "Default loads config/tracking_fp.yaml under the repo when present."
        ),
    )
    p.add_argument(
        "--fp-no-config",
        action="store_true",
        help="Ignore YAML; use built-in constructor defaults only.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if getattr(args, "geo_debug", False):
        os.environ["AEROSENTRY_GEO_DEBUG"] = "1"

    vid = args.video.expanduser().resolve()
    if not vid.is_file():
        print(f"Video not found: {vid}", file=sys.stderr)
        return 2

    gt_frames: Optional[list[Tuple[np.ndarray, np.ndarray]]] = None
    gt_iou = 0.5
    if args.gt_json is not None:
        gtp = args.gt_json.expanduser().resolve()
        if not gtp.is_file():
            print(f"GT JSON not found: {gtp}", file=sys.stderr)
            return 2
        gt_iou, gt_frames = load_gt_json(gtp)
        print(
            f"Loaded GT: {len(gt_frames)} frame entries, iou_threshold={gt_iou}",
            flush=True,
        )

    yaml_path = resolve_tracking_fp_yaml(
        args.fp_config,
        repo_root=_REPO_ROOT,
        no_config=args.fp_no_config,
    )
    fp_cfg = load_tracking_fp_yaml_dict(yaml_path)
    if yaml_path is not None:
        print(f"Loaded FP tracking config: {yaml_path}", flush=True)

    yolo_conf = args.yolo_conf
    if yolo_conf is None:
        yolo_conf = parse_hybrid_demo_yolo_conf(fp_cfg)
    if yolo_conf is None:
        yolo_conf = 0.25

    hybrid = HybridDetector(
        yolo_weights=args.yolo,
        rtdetr_weights=args.rtdetr,
        device=args.device,
        imgsz=args.imgsz,
        yolo_conf=yolo_conf,
        uncertainty_thresh=args.uncertainty_thresh,
        cooldown_frames=args.cooldown,
    )

    tracker = MockTracker(
        miss_threshold=parse_hybrid_demo_mock_tracker_miss_threshold(fp_cfg)
    )
    suppressor: Optional[FalsePositiveSuppressor] = None
    if getattr(args, "fp_geo_only", False):
        suppressor = build_false_positive_suppressor_from_mapping(fp_cfg, geo_only=True)
    elif args.fp_suppressor:
        suppressor = build_false_positive_suppressor_from_mapping(fp_cfg, geo_only=False)

    tp_h = fp_h = fn_h = 0
    tp_y = fp_y = fn_y = 0

    use_show = bool(args.show)
    if use_show and not _opencv_gui_available():
        print(
            "OpenCV was built without GUI (--show disabled). Use --out to save an MP4.",
            flush=True,
        )
        use_show = False
    if args.out is None and not use_show and args.gt_json is None:
        print(
            "Specify --out and/or --show to visualize hybrid boxes "
            "(or --gt-json for TP summary without video).",
            file=sys.stderr,
        )
        return 2

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        print(f"Could not open: {vid}", file=sys.stderr)
        return 2

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: Optional[cv2.VideoWriter] = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.out), fourcc, fps, (vw, vh))
        if not writer.isOpened():
            print(
                f"VideoWriter failed for {args.out}; try another path or codec.",
                file=sys.stderr,
            )
            cap.release()
            return 2

    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            lost = tracker.is_losing_track()
            dets_raw = hybrid.detect(frame, is_track_lost=lost)
            h, w = frame.shape[:2]
            if suppressor is not None:
                det_list = xyxy6_to_detections(dets_raw, frame.shape)
                ts = time.perf_counter()
                fd = FrameData(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=ts,
                    detections=det_list,
                )
                fd = suppressor.process(fd)
                dets = _detections_to_xyxy6(fd.detections, w, h)
            else:
                dets = dets_raw

            tracker.update(frame_id, dets)

            if gt_frames is not None:
                g_box, g_cls = _get_gt_frame(gt_frames, frame_id)
                y_box, y_cls, y_conf = _split_xyxy6(dets)
                a, b, c = match_image(
                    g_box, g_cls, y_box, y_cls, y_conf, 0.0, gt_iou
                )
                tp_h, fp_h, fn_h = tp_h + a, fp_h + b, fn_h + c

                y_only = hybrid.predict_yolo_only(frame)
                yy_box, yy_cls, yy_conf = _split_xyxy6(y_only)
                a2, b2, c2 = match_image(
                    g_box, g_cls, yy_box, yy_cls, yy_conf, 0.0, gt_iou
                )
                tp_y, fp_y, fn_y = tp_y + a2, fp_y + b2, fn_y + c2

            vis = _draw_xyxy6_overlay(
                frame,
                dets,
                used_rtdetr=hybrid.used_rtdetr_last,
                state_name=hybrid.last_state.name,
                cooldown=hybrid.cooldown_remaining,
                fp_applied=suppressor is not None,
            )
            if writer is not None:
                writer.write(vis)
            if use_show:
                cv2.imshow("aerosentry_hybrid_demo", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_id % 30 == 0:
                print(
                    f"frame={frame_id} state={hybrid.last_state.name} "
                    f"cooldown={hybrid.cooldown_remaining} n_dets={dets.shape[0]} "
                    f"lost_flag={lost} rtdetr={hybrid.used_rtdetr_last}",
                    flush=True,
                )

            frame_id += 1
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        _safe_destroy_windows()

    if args.out is not None:
        print(f"Wrote annotated video: {args.out.resolve()}", flush=True)

    print(f"Done: processed {frame_id} frames.", flush=True)

    if gt_frames is not None:
        ph, rh, f1h = _micro_prf1(tp_h, fp_h, fn_h)
        py, ry, f1y = _micro_prf1(tp_y, fp_y, fn_y)
        print("", flush=True)
        print("=== Ground-truth summary (micro over processed frames) ===", flush=True)
        print(
            f"Hybrid (cascade):  TP={tp_h} FP={fp_h} FN={fn_h}  "
            f"P={ph:.4f} R={rh:.4f} F1={f1h:.4f}",
            flush=True,
        )
        print(
            f"YOLO-only (same W):  TP={tp_y} FP={fp_y} FN={fn_y}  "
            f"P={py:.4f} R={ry:.4f} F1={f1y:.4f}",
            flush=True,
        )
        print(
            f"Delta TP (hybrid − YOLO): {tp_h - tp_y:+d}  "
            f"(positive ⇒ more true positives with cascade on this GT)",
            flush=True,
        )
        print(
            "Note: ``class_ids`` in GT must match detector class indices "
            "(RT-DETR COCO ids differ from a custom UAV head unless remapped).",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

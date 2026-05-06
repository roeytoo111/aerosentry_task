#!/usr/bin/env python3
"""Compare YOLO on one video: raw vs full FP suppressor vs geometry-only FP.

For each checkpoint, runs three passes over the clip. Writes Markdown + CSV + JSON.

**Ground truth (optional):** pass ``--gt-json`` with per-frame boxes in **pixel** ``xyxy`` to
obtain micro-averaged **Precision / Recall** and **TP / FP / FN** totals for each pipeline mode.
Without GT, only operational counts (detection sums, timing) are reported; P/R columns show ``—``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from src.core.data_contracts import Detection, FrameData
from src.evaluation.tactical_evaluator import UltralyticsYoloDetector
from src.models.evaluate_detector import _xywhn_to_xyxy, match_image
from src.tracking.fp_config import (
    build_false_positive_suppressor_from_mapping,
    load_tracking_fp_yaml_dict,
    resolve_tracking_fp_yaml,
)
from src.tracking.fp_suppressor import FalsePositiveSuppressor


FpMode = Literal["none", "full", "geo_only"]


def _default_label(weights: Path) -> str:
    w = weights.expanduser().resolve()
    if w.parent.name == "weights" and w.parent.parent.name:
        return w.parent.parent.name
    return w.stem


def _discover_run_weights(root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for p in sorted(root.glob("*/weights/best.pt")):
        if p.is_file():
            out.append((p.parent.parent.name, p.resolve()))
    return out


def load_gt_json(path: Path) -> Tuple[float, List[Tuple[np.ndarray, np.ndarray]]]:
    """Load optional GT: per-frame ``xyxy`` pixels + ``class_ids``.

    Schema::

        {
          "iou_threshold": 0.5,
          "frames": [
            { "xyxy": [[x1,y1,x2,y2], ...], "class_ids": [0, ...] },
            ...
          ]
        }

    ``class_ids`` may be omitted (defaults to class 0 per box).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    iou = float(raw.get("iou_threshold", 0.5))
    frames_in = raw.get("frames")
    if frames_in is None:
        raise ValueError("gt-json must contain 'frames' array")
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for fr in frames_in:
        xy = fr.get("xyxy") or []
        if not xy:
            out.append(
                (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                )
            )
            continue
        arr = np.asarray(xy, dtype=np.float32).reshape(-1, 4)
        cids = fr.get("class_ids")
        if cids is None:
            cls = np.zeros((arr.shape[0],), dtype=np.int64)
        else:
            cls = np.asarray(cids, dtype=np.int64).reshape(-1)
        if cls.shape[0] != arr.shape[0]:
            raise ValueError(
                f"class_ids length {cls.shape[0]} != xyxy rows {arr.shape[0]}"
            )
        out.append((arr, cls))
    return iou, out


def _get_gt_frame(
    gt_list: List[Tuple[np.ndarray, np.ndarray]], idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    if idx < len(gt_list):
        return gt_list[idx]
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    )


def _dets_to_arrays(
    dets: List[Detection], w: int, h: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not dets:
        z = np.zeros((0, 4), dtype=np.float32)
        return z, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    xywhn = np.asarray([list(d.bbox) for d in dets], dtype=np.float32)
    pred_xyxy = _xywhn_to_xyxy(xywhn, w, h)
    pred_cls = np.asarray([d.class_id for d in dets], dtype=np.int64)
    pred_conf = np.asarray([d.confidence for d in dets], dtype=np.float32)
    return pred_xyxy, pred_cls, pred_conf


@dataclass
class PassStats:
    frames: int
    det_sum: int
    det_max: int
    wall_s: float
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None

    @property
    def mean_per_frame(self) -> float:
        return float(self.det_sum) / max(self.frames, 1)

    @property
    def precision(self) -> Optional[float]:
        if self.tp is None:
            return None
        d = self.tp + (self.fp or 0)
        return float(self.tp) / d if d > 0 else 0.0

    @property
    def recall(self) -> Optional[float]:
        if self.tp is None:
            return None
        d = self.tp + (self.fn or 0)
        return float(self.tp) / d if d > 0 else 0.0

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        if p + r <= 0:
            return 0.0
        return 2.0 * p * r / (p + r)


@dataclass
class ModelVideoBench:
    model: str
    weights: str
    frames: int
    raw: PassStats
    with_fp: PassStats
    geo_only: PassStats


def _run_video_pass(
    cap_path: Path,
    det: UltralyticsYoloDetector,
    *,
    fp_mode: FpMode,
    max_frames: int,
    gt_frames: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    gt_iou: float = 0.5,
    fp_cfg: Optional[Dict[str, Any]] = None,
) -> PassStats:
    cfg = fp_cfg if fp_cfg is not None else {}
    sup: Optional[FalsePositiveSuppressor] = None
    if fp_mode == "full":
        sup = build_false_positive_suppressor_from_mapping(cfg, geo_only=False)
    elif fp_mode == "geo_only":
        sup = build_false_positive_suppressor_from_mapping(cfg, geo_only=True)

    cap = cv2.VideoCapture(str(cap_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {cap_path}")
    det_sum = 0
    det_max = 0
    n_frames = 0
    tp_a = fp_a = fn_a = 0
    use_gt = gt_frames is not None
    t0 = time.perf_counter()
    try:
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            ts = time.perf_counter()
            preds: List[Detection] = det.predict(fr)
            if sup is not None:
                fd = FrameData(
                    frame=fr,
                    frame_id=n_frames,
                    timestamp=ts,
                    detections=list(preds),
                )
                fd = sup.process(fd)
                preds = fd.detections

            n = len(preds)
            det_sum += n
            det_max = max(det_max, n)

            if use_gt:
                h, w = fr.shape[:2]
                pbox, pcls, pconf = _dets_to_arrays(preds, w, h)
                gbox, gcls = _get_gt_frame(gt_frames, n_frames)
                a, b, c = match_image(
                    gbox,
                    gcls,
                    pbox,
                    pcls,
                    pconf,
                    conf_thresh=0.0,
                    iou_thresh=gt_iou,
                )
                tp_a, fp_a, fn_a = tp_a + a, fp_a + b, fn_a + c

            n_frames += 1
            if max_frames > 0 and n_frames >= max_frames:
                break
    finally:
        cap.release()
    wall = time.perf_counter() - t0
    if use_gt:
        return PassStats(
            frames=n_frames,
            det_sum=det_sum,
            det_max=det_max,
            wall_s=wall,
            tp=tp_a,
            fp=fp_a,
            fn=fn_a,
        )
    return PassStats(
        frames=n_frames,
        det_sum=det_sum,
        det_max=det_max,
        wall_s=wall,
        tp=None,
        fp=None,
        fn=None,
    )


def _bench_one_model(
    label: str,
    weights: Path,
    video: Path,
    *,
    device: str,
    imgsz: int,
    conf: float,
    max_frames: int,
    gt_frames: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    gt_iou: float,
    fp_cfg: Dict[str, Any],
) -> ModelVideoBench:
    det_raw = UltralyticsYoloDetector(weights, device=device, imgsz=imgsz, conf=conf)
    raw = _run_video_pass(
        video,
        det_raw,
        fp_mode="none",
        max_frames=max_frames,
        gt_frames=gt_frames,
        gt_iou=gt_iou,
        fp_cfg=fp_cfg,
    )

    det_fp = UltralyticsYoloDetector(weights, device=device, imgsz=imgsz, conf=conf)
    gated = _run_video_pass(
        video,
        det_fp,
        fp_mode="full",
        max_frames=max_frames,
        gt_frames=gt_frames,
        gt_iou=gt_iou,
        fp_cfg=fp_cfg,
    )

    det_geo = UltralyticsYoloDetector(weights, device=device, imgsz=imgsz, conf=conf)
    geo = _run_video_pass(
        video,
        det_geo,
        fp_mode="geo_only",
        max_frames=max_frames,
        gt_frames=gt_frames,
        gt_iou=gt_iou,
        fp_cfg=fp_cfg,
    )

    frames = raw.frames
    return ModelVideoBench(
        model=label,
        weights=str(weights.resolve()),
        frames=frames,
        raw=raw,
        with_fp=gated,
        geo_only=geo,
    )


def _pct_drop(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return 100.0 * (1.0 - float(after) / float(before))


def _fmt_pr(s: PassStats) -> str:
    if s.precision is None:
        return "—"
    return f"P={s.precision:.3f} R={s.recall:.3f} F1={s.f1:.3f}"


def _fmt_counts(s: PassStats) -> str:
    if s.tp is None:
        return "—"
    return f"TP={s.tp} FP={s.fp} FN={s.fn}"


def render_markdown(
    rows: Sequence[ModelVideoBench], *, video: Path, conf: float, has_gt: bool
) -> str:
    lines = [
        "# Video FP comparison (raw vs full FP vs geometry-only)",
        "",
        f"- **Video:** `{video}`",
        f"- **conf (detector):** {conf}",
        f"- **GT metrics:** {'yes (micro P/R over frames)' if has_gt else 'no — precision/recall columns are operational N/A'}",
        "",
        "## Detection counts & timing",
        "",
        "| Model | Frames | Raw Σ | Raw μ/f | Full FP Σ | Full μ/f | Δ% vs raw | Geo-only Σ | Geo μ/f | Δ% vs raw | Wall raw (s) | Wall full (s) | Wall geo (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        df = _pct_drop(r.raw.det_sum, r.with_fp.det_sum)
        dg = _pct_drop(r.raw.det_sum, r.geo_only.det_sum)
        lines.append(
            f"| {r.model} | {r.frames} | {r.raw.det_sum} | {r.raw.mean_per_frame:.3f} | "
            f"{r.with_fp.det_sum} | {r.with_fp.mean_per_frame:.3f} | {df:.1f} | "
            f"{r.geo_only.det_sum} | {r.geo_only.mean_per_frame:.3f} | {dg:.1f} | "
            f"{r.raw.wall_s:.2f} | {r.with_fp.wall_s:.2f} | {r.geo_only.wall_s:.2f} |"
        )

    lines.extend(
        [
            "",
            "- **Δ% vs raw**: `100 * (1 - after/raw)` on total detection counts (operational proxy, not GT FPR).",
            "",
        ]
    )

    if has_gt:
        lines.extend(
            [
                "## Precision / Recall / TP-FP-FN (requires `--gt-json`)",
                "",
                "| Model | Raw | Full FP | Geo-only |",
                "| --- | --- | --- | --- |",
            ]
        )
        for r in rows:
            lines.append(
                f"| {r.model} | {_fmt_pr(r.raw)}; {_fmt_counts(r.raw)} | "
                f"{_fmt_pr(r.with_fp)}; {_fmt_counts(r.with_fp)} | "
                f"{_fmt_pr(r.geo_only)}; {_fmt_counts(r.geo_only)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- **Raw**: YOLO only.",
            "- **Full FP**: `FalsePositiveSuppressor` (TrackManager + geometry on confirmed tracks).",
            "- **Geo-only**: `FalsePositiveSuppressor(geo_only=True)` — geometry on every frame’s boxes.",
            "- Each mode reloads weights and rescans the video (three passes per row).",
            "- **GT file**: one JSON entry per **processed** frame (`frame_id` order); shorter lists pad with empty GT, longer lists truncate.",
            "",
        ]
    )
    return "\n".join(lines)


def _pass_dict(prefix: str, s: PassStats) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        f"{prefix}_det_sum": s.det_sum,
        f"{prefix}_mean_per_frame": round(s.mean_per_frame, 6),
        f"{prefix}_det_max": s.det_max,
        f"{prefix}_wall_s": round(s.wall_s, 4),
    }
    if s.tp is not None:
        d[f"{prefix}_tp"] = s.tp
        d[f"{prefix}_fp"] = s.fp
        d[f"{prefix}_fn"] = s.fn
        d[f"{prefix}_precision"] = round(s.precision or 0.0, 6)
        d[f"{prefix}_recall"] = round(s.recall or 0.0, 6)
        d[f"{prefix}_f1"] = round(s.f1 or 0.0, 6)
    else:
        d[f"{prefix}_tp"] = None
        d[f"{prefix}_fp"] = None
        d[f"{prefix}_fn"] = None
        d[f"{prefix}_precision"] = None
        d[f"{prefix}_recall"] = None
        d[f"{prefix}_f1"] = None
    return d


def _row_dict(r: ModelVideoBench) -> dict:
    out = {
        "model": r.model,
        "weights": r.weights,
        "frames": r.frames,
        "fp_full_delta_pct_vs_raw": round(
            _pct_drop(r.raw.det_sum, r.with_fp.det_sum), 4
        ),
        "geo_only_delta_pct_vs_raw": round(
            _pct_drop(r.raw.det_sum, r.geo_only.det_sum), 4
        ),
    }
    out.update(_pass_dict("raw", r.raw))
    out.update(_pass_dict("full_fp", r.with_fp))
    out.update(_pass_dict("geo_only", r.geo_only))
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        help="One or more checkpoints (best.pt paths).",
    )
    src.add_argument(
        "--discover-runs",
        type=Path,
        default=None,
        metavar="DIR",
        help="Glob DIR/*/weights/best.pt (e.g. runs/detect/aerosentry).",
    )
    p.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Labels for --weights (same count as --weights); default: parent run folder name.",
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-frames", type=int, default=0, help="0 = full clip.")
    p.add_argument(
        "--gt-json",
        type=Path,
        default=None,
        help="Per-frame GT boxes (pixel xyxy) for TP/FP/FN and P/R; see module docstring.",
    )
    p.add_argument("--out-md", type=Path, default=Path("outputs/fp_video_compare.md"))
    p.add_argument("--out-csv", type=Path, default=Path("outputs/fp_video_compare.csv"))
    p.add_argument("--out-json", type=Path, default=Path("outputs/fp_video_compare.json"))
    p.add_argument("--no-json", action="store_true")
    p.add_argument(
        "--fp-config",
        type=Path,
        default=None,
        help="YAML for full/geo FP modes (TrackManager + GeometricEgoMotion). Default: repo config/tracking_fp.yaml.",
    )
    p.add_argument(
        "--fp-no-config",
        action="store_true",
        help="Ignore YAML for FP passes; use built-in defaults.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 2

    gt_frames: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
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
        print(f"Loaded FP tracking config (full + geo passes): {yaml_path}", flush=True)

    pairs: List[Tuple[str, Path]]
    if args.discover_runs is not None:
        root = args.discover_runs.expanduser().resolve()
        if not root.is_dir():
            print(f"Not a directory: {root}", file=sys.stderr)
            return 2
        pairs = _discover_run_weights(root)
        if not pairs:
            print(f"No best.pt under {root}/*/weights/", file=sys.stderr)
            return 2
    else:
        wts = [w.expanduser().resolve() for w in args.weights]
        for w in wts:
            if not w.is_file():
                print(f"Weights not found: {w}", file=sys.stderr)
                return 2
        names = args.names
        if names is None:
            pairs = [(_default_label(w), w) for w in wts]
        else:
            if len(names) != len(wts):
                print("--names must have same length as --weights", file=sys.stderr)
                return 2
            pairs = list(zip(names, wts))

    rows: List[ModelVideoBench] = []
    for label, wpath in pairs:
        print(f"Benchmarking {label} ({wpath.name}) — 3 passes …", flush=True)
        rows.append(
            _bench_one_model(
                label,
                wpath,
                video,
                device=args.device,
                imgsz=args.imgsz,
                conf=args.conf,
                max_frames=args.max_frames,
                gt_frames=gt_frames,
                gt_iou=gt_iou,
                fp_cfg=fp_cfg,
            )
        )

    has_gt = gt_frames is not None
    md = render_markdown(
        rows, video=video, conf=args.conf, has_gt=has_gt
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out_md.resolve()}", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(_row_dict(rows[0]).keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(_row_dict(r))
        print(f"Wrote {args.out_csv.resolve()}", flush=True)

    if not args.no_json:
        payload = {
            "video": str(video),
            "conf": args.conf,
            "max_frames": args.max_frames,
            "gt_json": str(args.gt_json.resolve()) if args.gt_json else None,
            "gt_iou_threshold": gt_iou if has_gt else None,
            "models": [_row_dict(r) for r in rows],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_json.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

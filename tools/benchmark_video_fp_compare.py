#!/usr/bin/env python3
"""Compare YOLO + optional :class:`FalsePositiveSuppressor` on one video across multiple checkpoints.

For each ``best.pt``, runs the clip twice: raw detector counts vs FP-gated counts (stateful
suppressor). Writes Markdown + CSV for a comparison table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2

from src.core.data_contracts import Detection, FrameData
from src.evaluation.tactical_evaluator import UltralyticsYoloDetector
from src.tracking.fp_suppressor import FalsePositiveSuppressor


def _default_label(weights: Path) -> str:
    w = weights.expanduser().resolve()
    if w.parent.name == "weights" and w.parent.parent.name:
        return w.parent.parent.name
    return w.stem


def _discover_run_weights(root: Path) -> List[Tuple[str, Path]]:
    """Return (label, best.pt) for each ``<root>/<run>/weights/best.pt``."""
    out: List[Tuple[str, Path]] = []
    for p in sorted(root.glob("*/weights/best.pt")):
        if p.is_file():
            out.append((p.parent.parent.name, p.resolve()))
    return out


@dataclass
class PassStats:
    frames: int
    det_sum: int
    det_max: int
    wall_s: float

    @property
    def mean_per_frame(self) -> float:
        return float(self.det_sum) / max(self.frames, 1)


@dataclass
class ModelVideoBench:
    model: str
    weights: str
    frames: int
    raw: PassStats
    with_fp: PassStats


def _run_video_pass(
    cap_path: Path,
    det: UltralyticsYoloDetector,
    *,
    use_fp: bool,
    max_frames: int,
) -> PassStats:
    sup = FalsePositiveSuppressor() if use_fp else None
    cap = cv2.VideoCapture(str(cap_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {cap_path}")
    det_sum = 0
    det_max = 0
    frames = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            ts = time.perf_counter()
            preds: List[Detection] = det.predict(fr)
            n = len(preds)
            if use_fp:
                assert sup is not None
                fd = FrameData(
                    frame=fr,
                    frame_id=frames,
                    timestamp=ts,
                    detections=list(preds),
                )
                fd = sup.process(fd)
                n = len(fd.detections)
            det_sum += n
            det_max = max(det_max, n)
            frames += 1
            if max_frames > 0 and frames >= max_frames:
                break
    finally:
        cap.release()
    wall = time.perf_counter() - t0
    return PassStats(frames=frames, det_sum=det_sum, det_max=det_max, wall_s=wall)


def _bench_one_model(
    label: str,
    weights: Path,
    video: Path,
    *,
    device: str,
    imgsz: int,
    conf: float,
    max_frames: int,
) -> ModelVideoBench:
    det = UltralyticsYoloDetector(weights, device=device, imgsz=imgsz, conf=conf)
    raw = _run_video_pass(video, det, use_fp=False, max_frames=max_frames)
    det2 = UltralyticsYoloDetector(weights, device=device, imgsz=imgsz, conf=conf)
    gated = _run_video_pass(video, det2, use_fp=True, max_frames=max_frames)
    if raw.frames != gated.frames:
        # Should not happen unless file changes; keep note in JSON only
        pass
    return ModelVideoBench(
        model=label,
        weights=str(weights.resolve()),
        frames=raw.frames,
        raw=raw,
        with_fp=gated,
    )


def _fp_reduction_pct(raw_sum: int, fp_sum: int) -> float:
    if raw_sum <= 0:
        return 0.0
    return 100.0 * (1.0 - float(fp_sum) / float(raw_sum))


def render_markdown(rows: Sequence[ModelVideoBench], *, video: Path, conf: float) -> str:
    lines = [
        "# Video FP comparison",
        "",
        f"- **Video:** `{video}`",
        f"- **conf:** {conf}",
        "",
        "| Model | Frames | Raw Σ | Raw μ/frame | After FP Σ | After FP μ/frame | FP reduction % | Wall raw (s) | Wall +FP (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        red = _fp_reduction_pct(r.raw.det_sum, r.with_fp.det_sum)
        lines.append(
            f"| {r.model} | {r.frames} | {r.raw.det_sum} | {r.raw.mean_per_frame:.3f} | "
            f"{r.with_fp.det_sum} | {r.with_fp.mean_per_frame:.3f} | {red:.1f} | "
            f"{r.raw.wall_s:.2f} | {r.with_fp.wall_s:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **Raw Σ**: sum of detection counts per frame (YOLO only).",
            "- **After FP Σ**: sum after :class:`FalsePositiveSuppressor` (M-of-N + geometry on confirmed tracks).",
            "- **FP reduction %**: `100 * (1 - after/raw)` on totals; only meaningful when raw > 0.",
            "- Timings are wall-clock per full pass over the clip (loads model twice per row).",
            "",
        ]
    )
    return "\n".join(lines)


def _row_dict(r: ModelVideoBench) -> dict:
    return {
        "model": r.model,
        "weights": r.weights,
        "frames": r.frames,
        "raw_det_sum": r.raw.det_sum,
        "raw_mean_pf": r.raw.mean_per_frame,
        "fp_det_sum": r.with_fp.det_sum,
        "fp_mean_pf": r.with_fp.mean_per_frame,
        "fp_reduction_pct": _fp_reduction_pct(r.raw.det_sum, r.with_fp.det_sum),
        "wall_raw_s": r.raw.wall_s,
        "wall_fp_s": r.with_fp.wall_s,
    }


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
    p.add_argument("--out-md", type=Path, default=Path("outputs/fp_video_compare.md"))
    p.add_argument("--out-csv", type=Path, default=Path("outputs/fp_video_compare.csv"))
    p.add_argument("--out-json", type=Path, default=Path("outputs/fp_video_compare.json"))
    p.add_argument("--no-json", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 2

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
        print(f"Benchmarking {label} ({wpath.name}) …", flush=True)
        rows.append(
            _bench_one_model(
                label,
                wpath,
                video,
                device=args.device,
                imgsz=args.imgsz,
                conf=args.conf,
                max_frames=args.max_frames,
            )
        )

    md = render_markdown(rows, video=video, conf=args.conf)
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
            "models": [_row_dict(r) for r in rows],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_json.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

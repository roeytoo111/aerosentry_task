#!/usr/bin/env python3
"""Run tactical benchmarks and write ``TACTICAL_REPORT.md`` (FP32 / FP16 / INT8 comparison table)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.evaluation.tactical_evaluator import (
    TacticalEvaluator,
    TacticalMetricRow,
    UltralyticsYoloDetector,
)
from src.tracking.fp_suppressor import FalsePositiveSuppressor


def _load_metrics_json(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Metrics JSON must be an object keyed by mode name")
    return {str(k): dict(v) for k, v in raw.items()}  # type: ignore[arg-type]


def _default_row() -> Dict[str, float]:
    return {
        "precision": float("nan"),
        "recall": float("nan"),
        "fdr": float("nan"),
        "tnr_distractor": float("nan"),
        "latency_p95_ms": float("nan"),
        "fps": float("nan"),
    }


def build_report_rows(
    metrics_by_mode: Dict[str, Dict[str, Any]],
) -> Dict[str, TacticalMetricRow]:
    """Normalize JSON / dict payloads to :class:`TacticalMetricRow`."""
    rows: Dict[str, TacticalMetricRow] = {}
    for mode, vals in metrics_by_mode.items():
        base = _default_row()
        base.update({k: float(vals[k]) for k in base if k in vals})
        rows[mode] = TacticalMetricRow(
            precision=base["precision"],
            recall=base["recall"],
            fdr=base["fdr"],
            tnr_distractor=base["tnr_distractor"],
            latency_p95_ms=base["latency_p95_ms"],
            fps=base["fps"],
        )
    return rows


def render_markdown_table(rows: Dict[str, TacticalMetricRow]) -> str:
    """Render GitHub-flavored markdown table."""
    headers = [
        "Mode",
        "Precision",
        "Recall",
        "FDR",
        "TNR (distractors)",
        "E2E Latency p95 (ms)",
        "FPS",
    ]
    lines = [
        "# Tactical evaluation report",
        "",
        "Automated comparison across deployment precisions (Jetson Orin Nano–class targets).",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    order = ["FP32", "FP16", "INT8"]
    keys = [k for k in order if k in rows] + [k for k in sorted(rows) if k not in order]
    for mode in keys:
        r = rows[mode]
        line = (
            f"| {mode} | {r.precision:.4f} | {r.recall:.4f} | {r.fdr:.4f} | "
            f"{r.tnr_distractor:.4f} | {r.latency_p95_ms:.3f} | {r.fps:.2f} |"
        )
        lines.append(line)
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **Precision / Recall / FDR**: populate via offline val (e.g. ``evaluate_detector``) and merge into the metrics JSON.",
            "- **TNR**: fraction of hard-negative images with **zero** outputs after :class:`FalsePositiveSuppressor`.",
            "- **Latency**: detector + FP suppressor, ``torch.cuda.synchronize()`` bracketing (CUDA only).",
            "",
        ]
    )
    return "\n".join(lines)


def run_live_bench(
    weights: Path,
    video_path: Path,
    distractor_dir: Optional[Path],
    device: str,
    imgsz: int,
) -> Dict[str, Dict[str, float]]:
    """Execute a single-mode benchmark (FP32 path) to fill partial metrics."""
    import cv2

    det = UltralyticsYoloDetector(weights, device=device, imgsz=imgsz)

    def factory() -> FalsePositiveSuppressor:
        return FalsePositiveSuppressor()

    ev = TacticalEvaluator(det, factory, device_tag=device)
    vp = video_path.expanduser().resolve()
    if not vp.is_file():
        raise FileNotFoundError(
            f"Video not found (check path relative to cwd): {vp}"
        )
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"OpenCV could not open video (missing codec or bad file): {vp}"
        )
    frames: List[np.ndarray] = []
    try:
        for _ in range(64):
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            frames.append(fr)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(
            f"Read 0 frames from {vp} — file empty, wrong format, or path was not a video."
        )

    lat = ev.evaluate_end_to_end_latency(frames, warmup=10, iterations=80, reuse_suppressor=True)
    out: Dict[str, float] = {
        "latency_p95_ms": lat.p95_ms,
        "fps": lat.fps_mean,
    }
    if distractor_dir is not None and distractor_dir.is_dir():
        paths = sorted(
            p
            for ext in (".jpg", ".jpeg", ".png", ".bmp")
            for p in distractor_dir.glob(f"*{ext}")
        )
        drep = ev.evaluate_distractor_rejection(paths)
        out["tnr_distractor"] = drep.tnr
    return {"FP32": out}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Precomputed metrics per mode; merged with --live output keys.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("TACTICAL_REPORT.md"),
        help="Output markdown path.",
    )
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--distractor-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    merged: Dict[str, Dict[str, Any]] = {
        "FP32": _default_row(),
        "FP16": _default_row(),
        "INT8": _default_row(),
    }
    if args.metrics_json and args.metrics_json.is_file():
        file_m = _load_metrics_json(args.metrics_json)
        for mode, vals in file_m.items():
            if mode not in merged:
                merged[mode] = _default_row()
            merged[mode].update(vals)

    if args.weights and args.video:
        live = run_live_bench(
            args.weights,
            args.video,
            args.distractor_dir,
            args.device,
            args.imgsz,
        )
        for mode, vals in live.items():
            if mode not in merged:
                merged[mode] = _default_row()
            merged[mode].update(vals)

    rows = build_report_rows(merged)
    md = render_markdown_table(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(
        json.dumps({k: asdict(v) for k, v in rows.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.out} and {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

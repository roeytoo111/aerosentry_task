#!/usr/bin/env python3
"""Unified CLI entry point (assignment: ``python run.py <command> …``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable


def _checkpoint_path_or_exit(p: Path, flag: str = "--weights") -> Path:
    """Fail fast if README placeholders were copied literally or the file is missing."""
    s = str(p)
    if "..." in s:
        print(
            f"Invalid {flag} path (literal '...' is not a path): {p}\n"
            "Replace with your checkpoint path, e.g. after:\n"
            "  find runs -name best.pt",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not p.is_file():
        print(
            f"Weights file not found: {p.resolve()}\n"
            "Tip:  find runs -name best.pt\n"
            "     ls runs/detect/*/*/weights/",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return p


def _run_train(ns: argparse.Namespace) -> int:
    from src.models.train_detector import main as train_main

    argv = ["--experiment", ns.experiment.upper(), "--config", str(ns.config)]
    if ns.resume is not None:
        argv.extend(["--resume", str(ns.resume)])
    if ns.epochs is not None:
        argv.extend(["--epochs", str(ns.epochs)])
    return int(train_main(argv))


def _run_eval(ns: argparse.Namespace) -> int:
    from src.models.evaluate_detector import main as eval_main

    w = _checkpoint_path_or_exit(Path(ns.weights))
    argv = [
        "--weights",
        str(w),
        "--data",
        str(ns.data),
        "--device",
        ns.device,
        "--split",
        ns.split,
    ]
    argv.extend(["--iou-threshold", str(ns.iou_threshold)])
    argv.append("--conf-thresholds")
    argv.extend(str(t) for t in ns.conf_thresholds)
    if ns.out is not None:
        argv.extend(["--out", str(ns.out)])
    return int(eval_main(argv))


def _run_yolo_val_plots(ns: argparse.Namespace) -> int:
    from tools.val_yolo_plots import run_val_plots

    w = _checkpoint_path_or_exit(Path(ns.weights))
    sd = run_val_plots(
        weights=w,
        data=Path(ns.data),
        split=ns.split,
        imgsz=ns.imgsz,
        batch=ns.batch,
        device=ns.device,
        project=ns.project,
        name=(str(ns.name) if ns.name else None),
    )
    print(f"Saved plots and val metrics under:\n  {sd.resolve()}")
    return 0


def _run_report(ns: argparse.Namespace) -> int:
    from tools.generate_report import main as report_main

    argv: list[str] = []
    if ns.metrics_json:
        argv.extend(["--metrics-json", str(ns.metrics_json)])
    argv.extend(["--out", str(ns.out)])
    if ns.weights:
        w = _checkpoint_path_or_exit(Path(ns.weights))
        argv.extend(["--weights", str(w)])
    if ns.video:
        argv.extend(["--video", str(ns.video)])
    if ns.distractor_dir:
        argv.extend(["--distractor-dir", str(ns.distractor_dir)])
    argv.extend(["--device", ns.device, "--imgsz", str(ns.imgsz)])
    return int(report_main(argv))


def _run_export(ns: argparse.Namespace) -> int:
    from tools.export_engine import main as export_main

    w = _checkpoint_path_or_exit(Path(ns.weights))
    argv = [
        "--weights",
        str(w),
        "--out",
        str(ns.out),
        "--imgsz",
        str(ns.imgsz),
        "--workspace-gb",
        str(ns.workspace_gb),
    ]
    if ns.fp16:
        argv.append("--fp16")
    if ns.int8:
        argv.append("--int8")
    if ns.calibration_data:
        argv.extend(["--calibration-data", str(ns.calibration_data)])
    if ns.nms_e2e:
        argv.append("--nms-e2e")
    if ns.skip_onnx:
        argv.append("--skip-onnx")
    return int(export_main(argv))


def _run_split(ns: argparse.Namespace) -> int:
    from tools.split_dataset import main as split_main

    argv = [
        "--source",
        str(ns.source),
        "--output",
        str(ns.output),
        "--train-ratio",
        str(ns.train_ratio),
        "--val-ratio",
        str(ns.val_ratio),
        "--test-ratio",
        str(ns.test_ratio),
    ]
    if ns.symlink:
        argv.append("--symlink")
    return int(split_main(argv))


def _run_infer(ns: argparse.Namespace) -> int:
    from tools.infer_video import main as infer_main

    argv = [
        "--weights",
        str(_checkpoint_path_or_exit(Path(ns.weights))),
        "--source",
        str(ns.source),
        "--device",
        ns.device,
        "--imgsz",
        str(ns.imgsz),
        "--conf",
        str(ns.conf),
    ]
    if getattr(ns, "fp_geo_only", False):
        argv.append("--fp-geo-only")
    elif ns.fp_suppressor:
        argv.append("--fp-suppressor")
    if ns.out:
        argv.extend(["--out", str(ns.out)])
    if ns.show:
        argv.append("--show")
    if ns.max_frames:
        argv.extend(["--max-frames", str(ns.max_frames)])
    argv.extend(
        [
            "--progress-every",
            str(ns.progress_every),
            "--progress-seconds",
            str(ns.progress_seconds),
        ]
    )
    if ns.quiet:
        argv.append("--quiet")
    if getattr(ns, "debug_detections", False):
        argv.append("--debug-detections")
    if getattr(ns, "geo_debug", False):
        argv.append("--geo-debug")
    return int(infer_main(argv))


def _run_demo(ns: argparse.Namespace) -> int:
    from src.pipeline import PipelineManager

    key = ns.model_name.strip().lower()
    PipelineManager(model_name=key).process_video(str(ns.video))
    return 0


def _run_compare_fp_video(ns: argparse.Namespace) -> int:
    from tools.benchmark_video_fp_compare import main as compare_main

    argv: list[str] = [
        "--video",
        str(ns.video),
        "--device",
        ns.device,
        "--imgsz",
        str(ns.imgsz),
        "--conf",
        str(ns.conf),
        "--max-frames",
        str(ns.max_frames),
        "--out-md",
        str(ns.out_md),
        "--out-csv",
        str(ns.out_csv),
        "--out-json",
        str(ns.out_json),
    ]
    if ns.no_json:
        argv.append("--no-json")
    if ns.discover_runs is not None:
        argv.extend(["--discover-runs", str(ns.discover_runs)])
    else:
        w = [Path(p) for p in ns.weights]
        argv.append("--weights")
        argv.extend(str(p) for p in w)
        if ns.names:
            argv.append("--names")
            argv.extend(ns.names)
    if getattr(ns, "gt_json", None) is not None:
        argv.extend(["--gt-json", str(ns.gt_json)])
    return int(compare_main(argv))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="AeroSentry — train, evaluate, export, and demo entry point.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train YOLO11 via Ultralytics (experiments A/B/T/U).")
    p_train.add_argument(
        "--experiment",
        "-e",
        choices=["A", "B", "T", "U", "a", "b", "t", "u"],
        required=True,
    )
    p_train.add_argument("--config", type=Path, default=Path("config/experiments.yaml"))
    p_train.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue from this checkpoint (usually .../weights/last.pt); sets Ultralytics resume=True.",
    )
    p_train.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total epoch target (e.g. 100 after a 50-epoch run). Overrides YAML for this run.",
    )
    p_train.set_defaults(_handler=_run_train)

    p_eval = sub.add_parser("eval", help="Offline metrics on train/val/test image splits.")
    p_eval.add_argument("--weights", type=Path, required=True)
    p_eval.add_argument("--data", type=Path, default=Path("config/dataset_aerosentry.yaml"))
    p_eval.add_argument("--split", default="val", choices=["train", "val", "test"])
    p_eval.add_argument(
        "--device",
        default="0",
        help="Ultralytics device id (e.g. 0, cpu).",
    )
    p_eval.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching a prediction to GT (see evaluate_detector).",
    )
    p_eval.add_argument(
        "--conf-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Confidence cutoffs for the printed P/R/F1 rows (one row per value).",
    )
    p_eval.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional text file for metric lines (still prints to stdout).",
    )
    p_eval.set_defaults(_handler=_run_eval)

    p_yplots = sub.add_parser(
        "yolo-val-plots",
        help="Ultralytics val plots (PR/F1/confusion matrix) on train/val/test — same PNGs as training val.",
    )
    p_yplots.add_argument("--weights", type=Path, required=True)
    p_yplots.add_argument("--data", type=Path, default=Path("config/dataset_aerosentry.yaml"))
    p_yplots.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_yplots.add_argument("--device", default="0")
    p_yplots.add_argument("--imgsz", type=int, default=640)
    p_yplots.add_argument("--batch", type=int, default=8)
    p_yplots.add_argument(
        "--project",
        default="aerosentry_eval",
        help="Ultralytics project folder (under cwd).",
    )
    p_yplots.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run subfolder; default derives from checkpoint run name + split.",
    )
    p_yplots.set_defaults(_handler=_run_yolo_val_plots)

    p_rep = sub.add_parser("report", help="Write TACTICAL_REPORT.md (+ JSON) from live bench + optional metrics.")
    p_rep.add_argument("--weights", type=Path, default=None)
    p_rep.add_argument("--video", type=Path, default=None)
    p_rep.add_argument("--distractor-dir", type=Path, default=None)
    p_rep.add_argument("--metrics-json", type=Path, default=None)
    p_rep.add_argument("--out", type=Path, default=Path("TACTICAL_REPORT.md"))
    p_rep.add_argument("--device", default="0")
    p_rep.add_argument("--imgsz", type=int, default=640)
    p_rep.set_defaults(_handler=_run_report)

    p_exp = sub.add_parser("export", help="Export ONNX / TensorRT engine (Jetson path).")
    p_exp.add_argument("--weights", type=Path, required=True)
    p_exp.add_argument("--out", type=Path, default=Path("exports/tensorrt"))
    p_exp.add_argument("--imgsz", type=int, default=640)
    p_exp.add_argument("--fp16", action="store_true")
    p_exp.add_argument("--int8", action="store_true")
    p_exp.add_argument("--calibration-data", type=Path, default=None)
    p_exp.add_argument("--workspace-gb", type=float, default=4.0)
    p_exp.add_argument("--nms-e2e", action="store_true")
    p_exp.add_argument("--skip-onnx", action="store_true")
    p_exp.set_defaults(_handler=_run_export)

    p_sp = sub.add_parser("split", help="Sequence-aware dataset split (anti–temporal leakage).")
    p_sp.add_argument("--source", type=Path, required=True)
    p_sp.add_argument("--output", type=Path, required=True)
    p_sp.add_argument("--train-ratio", type=float, default=0.7)
    p_sp.add_argument("--val-ratio", type=float, default=0.15)
    p_sp.add_argument("--test-ratio", type=float, default=0.15)
    p_sp.add_argument("--symlink", action="store_true")
    p_sp.set_defaults(_handler=_run_split)

    p_inf = sub.add_parser(
        "infer",
        help="Trained YOLO on a video file — real weights + optional FP suppressor; use --show and/or --out.",
    )
    p_inf.add_argument("--weights", type=Path, required=True)
    p_inf.add_argument("--source", type=Path, required=True, help="Path to .mp4 / .avi / …")
    p_inf.add_argument("--device", default="0")
    p_inf.add_argument("--imgsz", type=int, default=640)
    p_inf.add_argument("--conf", type=float, default=0.25)
    p_inf.add_argument("--fp-suppressor", action="store_true")
    p_inf.add_argument(
        "--fp-geo-only",
        action="store_true",
        help="FP post-process: only GeometricEgoMotion (no TrackManager). Overrides --fp-suppressor.",
    )
    p_inf.add_argument("--out", type=Path, default=None)
    p_inf.add_argument("--show", action="store_true")
    p_inf.add_argument("--max-frames", type=int, default=0)
    p_inf.add_argument("--progress-every", type=int, default=30)
    p_inf.add_argument("--progress-seconds", type=float, default=3.0)
    p_inf.add_argument("--quiet", action="store_true")
    p_inf.add_argument(
        "--debug-detections",
        action="store_true",
        help="Log raw vs post-FP detection counts on progress lines.",
    )
    p_inf.add_argument(
        "--geo-debug",
        action="store_true",
        help="With --fp-suppressor / --fp-geo-only: GeometricEgoMotion / FP skip logs.",
    )
    p_inf.set_defaults(_handler=_run_infer)

    p_demo = sub.add_parser("demo", help="Run stub pipeline on a video path (ingest → detect stub → FP suppressor).")
    p_demo.add_argument("--video", type=Path, required=True)
    p_demo.add_argument("--model-name", default="yolov11")
    p_demo.set_defaults(_handler=_run_demo)

    p_cmp = sub.add_parser(
        "compare-fp-video",
        help="Benchmark weights on one video: raw vs full FP vs geo-only FP; MD/CSV/JSON.",
    )
    wg = p_cmp.add_mutually_exclusive_group(required=True)
    wg.add_argument(
        "--weights",
        type=str,
        nargs="+",
        metavar="PT",
        help="One or more best.pt paths.",
    )
    wg.add_argument(
        "--discover-runs",
        type=Path,
        metavar="DIR",
        help="Use every DIR/*/weights/best.pt (e.g. runs/detect/aerosentry).",
    )
    p_cmp.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help="Row labels for --weights (same count); default = run folder name.",
    )
    p_cmp.add_argument("--video", type=Path, required=True)
    p_cmp.add_argument("--device", default="0")
    p_cmp.add_argument("--imgsz", type=int, default=640)
    p_cmp.add_argument("--conf", type=float, default=0.25)
    p_cmp.add_argument("--max-frames", type=int, default=0)
    p_cmp.add_argument("--out-md", type=Path, default=Path("outputs/fp_video_compare.md"))
    p_cmp.add_argument("--out-csv", type=Path, default=Path("outputs/fp_video_compare.csv"))
    p_cmp.add_argument("--out-json", type=Path, default=Path("outputs/fp_video_compare.json"))
    p_cmp.add_argument(
        "--gt-json",
        type=Path,
        default=None,
        help="Optional per-frame GT (pixel xyxy) for TP/FP/FN and P/R; see tools/benchmark_video_fp_compare.py doc.",
    )
    p_cmp.add_argument("--no-json", action="store_true")
    p_cmp.set_defaults(_handler=_run_compare_fp_video)

    args = parser.parse_args(argv)
    handler: Callable[[Any], int] = getattr(args, "_handler")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

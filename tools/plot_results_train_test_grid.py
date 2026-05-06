#!/usr/bin/env python3
"""Ultralytics-style 2×5 ``results.png`` grid: train row from CSV + **test** row from checkpoints.

**Methodology**

* **Row 1** comes from ``results.csv`` logged during training:

  * ``train/*`` losses — training set.
  * ``metrics/precision(B)``, ``metrics/recall(B)`` — whatever split Ultralytics used each epoch
    (usually ``valid/images`` when ``split: val``). These reflect *development* monitoring, not final test.

* **Row 2** is **recomputed offline** on the **test** split (``test/images`` in your yaml) for each
  saved ``weights/epoch{k}.pt``. That avoids using test during training while still reporting true
  generalization trends **if** you saved checkpoints (``save_period: 1`` or N in training config).

* If there are **no** ``epoch*.pt`` files, the script falls back to ``last.pt`` only and plots **one**
  marker per subplot at the final epoch (with a note to set ``save_period`` for a full curve).

**Row 2 — metrics columns (last two plots):**

* ``ultralytics`` (default): mAP50 / mAP50-95 via ``YOLO().val(split=test)`` — same family as the training log.
* ``evaluate-detector``: :mod:`src.models.evaluate_detector` greedy IoU matching + conf sweep at a **single**
  confidence (``--eval-conf``), so P/R/F1 match the rest AeroSentry tooling. **Training losses** in row 2
  still come from ``compute_split_mean_losses`` (``evaluate_detector`` only runs inference, not ``loss``).

Examples::

    PYTHONPATH=. python3 tools/plot_results_train_test_grid.py \\
        --results runs/detect/aerosentry/yolo11_baseline-4/results.csv \\
        --out runs/detect/aerosentry/yolo11_baseline-4/results_train_test.png
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_TOOLS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plot_yolo_loss_curves as pylc  # noqa: E402


def _smooth(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float64)
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(y, sigma=3)
    except ImportError:
        w = max(3, min(9, len(y) // 5 * 2 + 1))
        if len(y) < w or w < 1:
            return y
        k = np.ones(w, dtype=np.float64) / w
        return np.convolve(y, k, mode="same")


def _resolve_data_yaml_from_run(run_dir: Path) -> Path:
    args_yaml = run_dir / "args.yaml"
    with args_yaml.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return pylc._resolve_data_yaml(args_yaml, cfg["data"])


def _collect_checkpoints(weights_dir: Path) -> list[tuple[int, Path]]:
    """(trainer_epoch_0based, path); CSV epoch ≈ trainer_epoch + 1."""
    found: list[tuple[int, Path]] = []
    for p in weights_dir.glob("epoch*.pt"):
        m = re.fullmatch(r"epoch(\d+)\.pt", p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return found


def _eval_test_losses_only(
    weights_pt: Path,
    data_yaml: Path,
    *,
    device: str,
    batch: int,
) -> tuple[float, float, float]:
    return pylc.compute_split_mean_losses(
        weights_pt, data_yaml, split="test", batch=batch, device=device
    )


def _eval_test_metrics_ultralytics(
    weights_pt: Path,
    data_yaml: Path,
    *,
    device: str,
    batch: int,
    imgsz: int,
) -> dict[str, float]:
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    from ultralytics.models.yolo.model import YOLO
    from ultralytics.utils import LOGGER as UL_LOGGER

    UL_LOGGER.setLevel(logging.ERROR)
    m = YOLO(str(weights_pt), verbose=False)
    met = m.val(
        data=str(data_yaml),
        split="test",
        plots=False,
        verbose=False,
        device=device,
        batch=batch,
        imgsz=imgsz,
    )
    rd = met.results_dict
    return {
        "metrics/mAP50(B)": float(rd["metrics/mAP50(B)"]),
        "metrics/mAP50-95(B)": float(rd["metrics/mAP50-95(B)"]),
    }


def _eval_test_metrics_evaluate_detector(
    weights_pt: Path,
    data_yaml: Path,
    *,
    device: str,
    imgsz: int,
    conf: float,
    iou_thresh: float,
) -> dict[str, float]:
    from src.models.evaluate_detector import aggregate_metrics, gather_predictions

    per_image = gather_predictions(
        weights_pt,
        data_yaml,
        split="test",
        imgsz=imgsz,
        device=str(device),
    )
    rows = aggregate_metrics(per_image, [conf], iou_thresh)
    key = f"{conf:.2f}"
    if key not in rows:
        raise KeyError(f"No metrics for conf={conf}; keys={sorted(rows)}")
    r = rows[key]
    return {
        "eval/precision": float(r["precision"]),
        "eval/recall": float(r["recall"]),
        "_eval/f1": float(r["f1"]),
    }


def _eval_test_for_checkpoint(
    weights_pt: Path,
    data_yaml: Path,
    *,
    device: str,
    batch: int,
    imgsz: int,
    row2_metrics: str,
    eval_conf: float,
    eval_iou: float,
) -> dict[str, float]:
    """Loss triple + last two metric panels (Ultralytics mAP **or** project P/R from evaluate_detector)."""
    box, cls_, dfl = _eval_test_losses_only(weights_pt, data_yaml, device=device, batch=batch)
    out: dict[str, float] = {
        "test/box_loss": box,
        "test/cls_loss": cls_,
        "test/dfl_loss": dfl,
    }
    if row2_metrics == "ultralytics":
        out.update(_eval_test_metrics_ultralytics(weights_pt, data_yaml, device=device, batch=batch, imgsz=imgsz))
    else:
        out.update(
            _eval_test_metrics_evaluate_detector(
                weights_pt,
                data_yaml,
                device=device,
                imgsz=imgsz,
                conf=eval_conf,
                iou_thresh=eval_iou,
            )
        )
    return out


def plot_grid(
    results_csv: Path,
    out_png: Path,
    *,
    data_yaml: Path | None,
    weights_dir: Path | None,
    device: str,
    batch: int,
    row2_metrics: str,
    eval_conf: float,
    eval_iou: float,
) -> None:
    pylc._quiet_ultralytics()
    run_dir = results_csv.parent
    if data_yaml is None:
        data_yaml = _resolve_data_yaml_from_run(run_dir)
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    wdir = weights_dir or (run_dir / "weights")
    if not wdir.is_dir():
        raise FileNotFoundError(wdir)

    df = pd.read_csv(results_csv)
    x = df["epoch"].astype(float).to_numpy()

    with (run_dir / "args.yaml").open(encoding="utf-8") as f:
        train_args = yaml.safe_load(f)
    imgsz = int(train_args.get("imgsz", 640))
    if batch <= 0:
        batch = int(train_args.get("batch", 8))

    ckpts = _collect_checkpoints(wdir)
    if ckpts:
        eval_plan: list[tuple[float, Path]] = [(float(k + 1), p) for k, p in ckpts]
        methodology_footer = (
            f"Bottom row: recomputed on test split for {len(eval_plan)} checkpoint(s) "
            f"(epoch*.pt in {wdir.name})."
        )
    else:
        last_pt = wdir / "last.pt"
        if not last_pt.is_file():
            raise FileNotFoundError(
                f"No ``epoch*.pt`` and no ``last.pt`` in {wdir}. Cannot build test row."
            )
        final_e = float(x.max()) if len(x) else 1.0
        eval_plan = [(final_e, last_pt)]
        methodology_footer = (
            "Bottom row: **single** eval on ``last.pt`` at final epoch — set ``save_period`` "
            "during training for per-epoch test curves (recommended)."
        )

    if row2_metrics == "evaluate-detector":
        methodology_footer += (
            f" | Bottom tail: ``evaluate_detector`` P/R on test (conf={eval_conf}, IoU={eval_iou}); "
            "F1 echoed to stdout per checkpoint."
        )

    if row2_metrics == "ultralytics":
        tail_keys = ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    else:
        # Five panels total: keep P/R here; F1 is harmonic mean (printed after run).
        tail_keys = ["eval/precision", "eval/recall"]

    test_rows: dict[str, list[float]] = {k: [] for k in ["test/box_loss", "test/cls_loss", "test/dfl_loss"] + tail_keys}
    xt: list[float] = []
    f1_per_ckpt: list[tuple[float, float]] = []
    for epoch_disp, wpath in eval_plan:
        xt.append(epoch_disp)
        stats = _eval_test_for_checkpoint(
            wpath,
            data_yaml,
            device=device,
            batch=batch,
            imgsz=imgsz,
            row2_metrics=row2_metrics,
            eval_conf=eval_conf,
            eval_iou=eval_iou,
        )
        f1_snap = None
        if row2_metrics == "evaluate-detector":
            f1_snap = stats.pop("_eval/f1", None)
        for k in test_rows:
            test_rows[k].append(stats[k])
        if f1_snap is not None:
            f1_per_ckpt.append((epoch_disp, f1_snap))

    if f1_per_ckpt and row2_metrics == "evaluate-detector":
        for ep, f1v in f1_per_ckpt:
            print(f"  epoch≈{ep:.0f}  evaluate_detector F1@conf={eval_conf}  {f1v:.4f}")

    # Layout matches Ultralytics ``results.png``: train losses + val-epoch metrics, then losses + mAP
    row1_cols = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
    ]
    row2_cols = [
        "test/box_loss",
        "test/cls_loss",
        "test/dfl_loss",
    ] + tail_keys

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), tight_layout=True)
    axes_flat = axes.ravel()

    def draw_series(ax, xv: np.ndarray, yv: np.ndarray, title: str) -> None:
        ax.plot(xv, yv, marker=".", label="results", linewidth=2, markersize=6)
        if len(yv) >= 3:
            ax.plot(xv, _smooth(yv), ":", label="smooth", linewidth=2)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("epoch")

    # Row 1: CSV
    for j, col in enumerate(row1_cols):
        yv = df[col].to_numpy(dtype=np.float64)
        draw_series(axes_flat[j], x, yv, col)
    axes_flat[1].legend(fontsize=8)

    def _csv_reference_column(row2_metric_name: str) -> str | None:
        """Use training-time val curves to set a sane y-axis when only one test checkpoint exists."""
        m = {
            "test/box_loss": "val/box_loss",
            "test/cls_loss": "val/cls_loss",
            "test/dfl_loss": "val/dfl_loss",
            "metrics/mAP50(B)": "metrics/mAP50(B)",
            "metrics/mAP50-95(B)": "metrics/mAP50-95(B)",
            "eval/precision": "metrics/precision(B)",
            "eval/recall": "metrics/recall(B)",
        }
        return m.get(row2_metric_name)

    # Row 2: test (sparse x)
    xt_arr = np.array(xt, dtype=np.float64)
    single_ckpt = len(xt_arr) == 1
    x_full = (float(x.min()), float(x.max())) if len(x) else (0.0, 1.0)
    for j, col in enumerate(row2_cols):
        yv = np.array(test_rows[col], dtype=np.float64)
        ax = axes_flat[5 + j]
        ax.set_xlim(x_full[0], x_full[1])
        if single_ckpt:
            y0 = float(yv[0])
            ax.axhline(y0, color="C0", linestyle="--", linewidth=1.5, alpha=0.75, label="test (constant)")
            ax.plot(xt_arr, yv, marker="o", color="C0", linestyle="None", markersize=11, label="test checkpoint", zorder=5)
            ref_c = _csv_reference_column(col)
            if ref_c is not None and ref_c in df.columns:
                ref_y = df[ref_c].to_numpy(dtype=np.float64)
                lo, hi = float(np.nanmin(ref_y)), float(np.nanmax(ref_y))
                lo2, hi2 = min(lo, y0), max(hi, y0)
                pad = (hi2 - lo2) * 0.12 + 1e-6
                ax.set_ylim(lo2 - pad, hi2 + pad)
            else:
                if y0 > 0:
                    ax.set_ylim(0.0, max(y0 * 1.25, y0 + 0.01))
                else:
                    ax.set_ylim(-0.05, 0.2)
            ax.text(
                0.02,
                0.98,
                "one weight file\n(no epoch*.pt)",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
            )
        else:
            ax.plot(xt_arr, yv, marker=".", color="C0", label="test eval", linewidth=2, markersize=8)
            if len(yv) >= 3:
                ax.plot(xt_arr, _smooth(yv), ":", color="C1", label="smooth", linewidth=2)
        subtitle = "(test split)"
        if row2_metrics != "ultralytics" and col.startswith("eval/"):
            subtitle = f"(test, evaluate_detector conf={eval_conf})"
        ax.set_title(f"{col}\n{subtitle}", fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Top: training log from ``results.csv`` (train losses + epoch monitoring metrics). "
        "Bottom: **held-out test** replay from checkpoints (no test leakage into training). "
        + methodology_footer,
        fontsize=10,
        y=1.02,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description="2×5 results grid: CSV train row + recomputed test row.")
    p.add_argument("--results", type=Path, required=True, help="Path to results.csv")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: <run-dir>/results_train_test.png)",
    )
    p.add_argument("--data-yaml", type=Path, default=None, help="Override dataset yaml")
    p.add_argument("--weights-dir", type=Path, default=None, help="Directory with epoch*.pt / last.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=0, help="Val batch (0 = from args.yaml)")
    p.add_argument(
        "--row2-metrics",
        choices=("ultralytics", "evaluate-detector"),
        default="ultralytics",
        help="How to fill the last 2–3 bottom panels (mAP vs project eval).",
    )
    p.add_argument(
        "--eval-conf",
        type=float,
        default=0.25,
        help="Fixed conf threshold for evaluate-detector tail metrics.",
    )
    p.add_argument("--eval-iou", type=float, default=0.5, help="IoU threshold for evaluate_detector matching.")
    args = p.parse_args()

    results_csv = args.results.expanduser().resolve()
    run_dir = results_csv.parent
    out_png = args.out.expanduser().resolve() if args.out else (run_dir / "results_train_test.png")

    plot_grid(
        results_csv,
        out_png,
        data_yaml=args.data_yaml.expanduser().resolve() if args.data_yaml else None,
        weights_dir=args.weights_dir.expanduser().resolve() if args.weights_dir else None,
        device=args.device,
        batch=args.batch,
        row2_metrics=args.row2_metrics,
        eval_conf=args.eval_conf,
        eval_iou=args.eval_iou,
    )
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

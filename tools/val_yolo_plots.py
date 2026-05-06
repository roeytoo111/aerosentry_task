#!/usr/bin/env python3
"""Run Ultralytics ``YOLO.val(..., plots=True)`` on ``train`` / ``val`` / ``test``.

This writes the same artifacts as training-time validation (when enabled): e.g.
``BoxPR_curve.png``, ``BoxF1_curve.png``, ``BoxP_curve.png``, ``BoxR_curve.png``,
``confusion_matrix.png``, ``confusion_matrix_normalized.png``, ``results.csv`` for
that **single** eval — *not* the multi-epoch ``results.png`` from training.

Outputs go under ``{project}/{name}/`` (Ultralytics layout), defaulting to a new
folder so your original run dir is unchanged.

Example (test split, baseline weights)::

    PYTHONPATH=. python3 tools/val_yolo_plots.py \\
        --weights runs/detect/aerosentry/yolo11_baseline-4/weights/best.pt \\
        --data config/dataset_aerosentry.yaml \\
        --split test

Or via ``python3 run.py yolo-val-plots ...``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def run_val_plots(
    *,
    weights: Path,
    data: Path,
    split: str = "test",
    imgsz: int = 640,
    batch: int = 8,
    device: str = "0",
    project: str = "aerosentry_eval",
    name: str | None = None,
) -> Path:
    weights = weights.expanduser().resolve()
    data = data.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not data.is_file():
        raise FileNotFoundError(data)

    if name is None:
        name = f"{weights.parent.parent.name}_split_{split}"

    os.environ.setdefault("WANDB_DISABLED", "true")

    from ultralytics.models.yolo.model import YOLO

    model = YOLO(str(weights), verbose=True)
    metrics = model.val(
        data=str(data),
        split=split,
        plots=True,
        project=project,
        name=name,
        exist_ok=True,
        imgsz=imgsz,
        batch=batch,
        device=device,
    )
    save_dir = Path(getattr(metrics, "save_dir", "") or "")
    if not save_dir.is_dir():
        v = getattr(model, "validator", None)
        if v is not None and getattr(v, "save_dir", None):
            save_dir = Path(v.save_dir)
    if not save_dir.is_dir():
        raise RuntimeError("Could not resolve Ultralytics save_dir after val().")
    return save_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ultralytics val + plot curves on a dataset split.")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--project",
        default="aerosentry_eval",
        help="Top-level folder under cwd for this eval (Ultralytics ``project``).",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Subfolder name. Default: ``<run_folder>_split_<split>`` from checkpoint path.",
    )
    args = p.parse_args(argv)

    sd = run_val_plots(
        weights=args.weights,
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print(f"Saved plots and val metrics under:\n  {sd.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot YOLO training / hold-out losses from ``results.csv`` and checkpoints.

**Epoch metrics in CSV:** Ultralytics always writes the per-epoch hold-out columns as ``val/*``,
whether the underlying split was ``valid/images`` (``split: val``) or ``test/images``
(``split: test`` in ``config/experiments.yaml``). Use ``plot_yolo_loss_curves.py --csv-holdout test``
after training with ``split: test`` so plot legends match the data.

Example: ``python3 tools/plot_yolo_loss_curves.py --mode val --csv-holdout test --csv runs/detect/.../results.csv --with-val-plots``
(loss figure + optional ``pr_cm_plots/``; omit ``--with-val-plots`` for loss only).

With ``--with-val-plots``, also runs a one-shot ``YOLO.val(..., plots=True)`` on the chosen split
and copies only ``BoxPR_curve.png`` and the two confusion-matrix PNGs into ``<run-dir>/pr_cm_plots/``
(override with ``--val-plots-out``); the Ultralytics staging folder under the run dir is removed afterward.

**``--mode test-curve`` / ``test-line``:** hold-out recomputed from ``.pt`` files, e.g.
- Retrain with ``save_period: 1`` (or N) so ``weights/epoch{k}.pt`` exist, then use
  ``--mode test-curve``, or
- Evaluate ``test/images`` once on ``last.pt`` (``--mode test-line``).

Heavy imports (torch / ultralytics) load only for modes that need checkpoints.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def _resolve_data_yaml(args_yaml: Path, data_field: str) -> Path:
    rel = Path(data_field)
    if rel.is_absolute() and rel.is_file():
        return rel.resolve()
    d = args_yaml.parent
    for _ in range(10):
        trial = d / rel
        if trial.is_file():
            return trial.resolve()
        d = d.parent
    raise FileNotFoundError(f"Could not resolve data yaml {data_field!r} from {args_yaml}")


def _resolve_torch_device(device: str | int) -> Any:
    import torch

    if device == "cpu":
        return torch.device("cpu")
    if isinstance(device, str) and device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _quiet_ultralytics() -> None:
    for name in (
        "ultralytics",
        "ultralytics.nn",
        "ultralytics.data",
        "ultralytics.models",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def compute_split_mean_losses(
    weights_pt: Path,
    data_yaml: Path,
    *,
    split: str = "test",
    batch: int = 8,
    device: str | int = 0,
) -> tuple[float, float, float]:
    """Mean (box, cls, dfl) loss items averaged over val-style batches (matches trainer style)."""
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, IterableSimpleNamespace, LOGGER

    _quiet_ultralytics()
    LOGGER.setLevel(logging.ERROR)

    dev = _resolve_torch_device(device)
    model, ckpt = load_checkpoint(str(weights_pt), device=dev, inplace=True, fuse=False)
    ta = ckpt.get("train_args")
    if not isinstance(ta, dict):
        raise ValueError(f"No train_args dict in {weights_pt}")
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **ta})
    model.criterion = None

    data = check_det_dataset(str(data_yaml))
    args = get_cfg(DEFAULT_CFG)
    args.data = str(data_yaml)
    args.imgsz = int(ta.get("imgsz", 640))
    args.batch = batch
    args.workers = min(int(ta.get("workers", 8)), 8)
    args.rect = False

    path = data[split]
    gs = max(int(model.stride.max()), 32)
    ds = build_yolo_dataset(args, path, batch, data, mode="val", stride=gs)
    loader = build_dataloader(
        ds, batch=batch, workers=args.workers, shuffle=False, rank=-1, drop_last=False
    )

    tloss: torch.Tensor | None = None
    nb = 0
    for b in loader:
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                b[k] = v.to(dev, non_blocking=True)
        b["img"] = b["img"].float() / 255.0
        with torch.no_grad():
            preds = model(b["img"])
            _, items = model.loss(b, preds)
        items = items.detach()
        tloss = items if tloss is None else tloss + items
        nb += 1
    if nb == 0:
        raise RuntimeError(f"No batches for split {split!r}")
    box, cls_, dfl = (tloss / nb).cpu().tolist()
    return float(box), float(cls_), float(dfl)


def _pick_eval_weights(run_dir: Path, override: Path | None) -> Path:
    if override is not None:
        p = override.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    for n in ("best.pt", "last.pt"):
        cand = run_dir / "weights" / n
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No best.pt or last.pt under {run_dir / 'weights'}")


def _minimal_pr_cm_plots(
    *,
    run_dir: Path,
    weights: Path,
    data_yaml: Path,
    split: str,
    dest_dir: Path,
    device: str,
    imgsz: int = 640,
    batch: int = 8,
) -> None:
    """Run Ultralytics val(plots=True), copy PR + confusion PNGs to ``dest_dir``, remove staging folder."""
    import importlib.util

    vpath = Path(__file__).resolve().parent / "val_yolo_plots.py"
    spec = importlib.util.spec_from_file_location("_aerosentry_val_yolo_plots", vpath)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load val_yolo_plots from {vpath}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_val_plots = mod.run_val_plots

    staging_name = "_val_plots_staging_tmp"
    staging = run_dir / staging_name
    if staging.is_dir():
        shutil.rmtree(staging)

    save_dir = run_val_plots(
        weights=weights,
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(run_dir.resolve()),
        name=staging_name,
    )
    save_dir = save_dir.resolve()

    dest_dir = dest_dir.expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = (
        "BoxPR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    )
    copied = 0
    for w in wanted:
        src = save_dir / w
        if src.is_file():
            shutil.copy2(src, dest_dir / w)
            copied += 1

    if copied == 0:
        raise RuntimeError(
            f"Expected PR/confusion PNGs under {save_dir} after val(); check Ultralytics version / val output."
        )

    if save_dir.is_dir():
        shutil.rmtree(save_dir, ignore_errors=True)


def _epoch_index_and_path(weights_dir: Path) -> list[tuple[int, Path]]:
    """Ultralytics saves ``epoch{k}.pt`` where ``k`` matches trainer epoch (0-based); CSV uses 1-based epoch."""
    found: list[tuple[int, Path]] = []
    for p in weights_dir.glob("epoch*.pt"):
        m = re.fullmatch(r"epoch(\d+)\.pt", p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return found


def plot_val_curves(
    results_csv: Path,
    out_png: Path,
    *,
    csv_holdout: str = "val",
) -> None:
    """Train + per-epoch hold-out curves from ``results.csv`` (columns ``train/*`` and ``val/*``).

    Note: Ultralytics **always** logs the epoch hold-out metrics under the prefix ``val/`` in the CSV,
    even when ``split: test`` was used during training (data from ``test/images``).
    """
    df = pd.read_csv(results_csv)
    epoch = df["epoch"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), constrained_layout=True)
    pairs = [
        ("train/box_loss", "val/box_loss", "Box loss"),
        ("train/cls_loss", "val/cls_loss", "Cls loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL loss"),
    ]
    if csv_holdout == "test":
        holdout_legend = "Test"
    else:
        holdout_legend = "Validation"

    for ax, (tr, va, name) in zip(axes, pairs):
        ax.plot(epoch, df[tr], label="Train", color="#1f77b4", linewidth=1.8)
        ax.plot(epoch, df[va], label=holdout_legend, color="#ff7f0e", linewidth=1.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_test_line(
    results_csv: Path,
    out_png: Path,
    *,
    weights_pt: Path,
    data_yaml: Path,
    device: str | int,
) -> None:
    df = pd.read_csv(results_csv)
    epoch = df["epoch"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), constrained_layout=True)
    trains = [
        ("train/box_loss", "Box loss"),
        ("train/cls_loss", "Cls loss"),
        ("train/dfl_loss", "DFL loss"),
    ]
    triple = compute_split_mean_losses(weights_pt, data_yaml, split="test", batch=8, device=device)
    wname = weights_pt.name

    for ax, (tr_col, name), i in zip(axes, trains, range(3)):
        ax.plot(epoch, df[tr_col], label="Train", color="#1f77b4", linewidth=1.8)
        ax.axhline(
            triple[i],
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            label=f"Test hold-out (mean, {wname})",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_test_curve(
    results_csv: Path,
    out_png: Path,
    *,
    weights_dir: Path,
    data_yaml: Path,
    device: str | int,
) -> None:
    """Train from CSV + test split loss recomputed for each ``epoch{k}.pt``."""
    checkpoints = _epoch_index_and_path(weights_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"No epoch*.pt under {weights_dir}. Retrain with save_period>=1 "
            "(Ultralytics writes epoch0.pt, epoch1.pt, …)."
        )

    df = pd.read_csv(results_csv)
    train_epoch = df["epoch"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), constrained_layout=True)
    trains = [
        ("train/box_loss", "Box loss"),
        ("train/cls_loss", "Cls loss"),
        ("train/dfl_loss", "DFL loss"),
    ]

    csv_epochs: list[float] = []
    boxes: list[float] = []
    clss: list[float] = []
    dfls: list[float] = []
    for k, wpt in checkpoints:
        csv_e = float(k + 1)
        csv_epochs.append(csv_e)
        box, cls_, dfl = compute_split_mean_losses(wpt, data_yaml, split="test", batch=8, device=device)
        boxes.append(box)
        clss.append(cls_)
        dfls.append(dfl)

    series = [boxes, clss, dfls]
    for ax, (tr_col, name), i in zip(axes, trains, range(3)):
        ax.plot(train_epoch, df[tr_col], label="Train", color="#1f77b4", linewidth=1.8)
        ax.plot(
            csv_epochs,
            series[i],
            label="Test hold-out (recomputed)",
            color="#d62728",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Plot Ultralytics loss curves: vanilla train/val from CSV, or test hold-out from checkpoints.",
        allow_abbrev=False,
    )
    p.add_argument(
        "--results",
        "--csv",
        type=Path,
        dest="results",
        required=True,
        help="Path to Ultralytics ``results.csv`` (alias: ``--csv``).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <run-dir>/loss_curves_train_holdout.png).",
    )
    p.add_argument(
        "--mode",
        choices=("val", "test-line", "test-curve"),
        default="val",
        help="'val' = train+val curves from CSV (standard). 'test-line' = train + one test level from --weights. "
        "'test-curve' = train + test recomputed for each weights/epoch*.pt (needs save_period).",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="For test-line: checkpoint (default: <results-dir>/weights/last.pt)",
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="For test-curve: directory with epoch*.pt (default: <results-dir>/weights)",
    )
    p.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Dataset yaml (default: ``data`` from <results-dir>/args.yaml)",
    )
    p.add_argument(
        "--csv-holdout",
        choices=("val", "test"),
        default="val",
        help="Which split was used for epoch metrics during training (``split`` in YAML). "
        "CSV columns are always ``val/*``; set ``test`` if you trained with ``split: test`` so legends match.",
    )
    p.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    p.add_argument(
        "--with-val-plots",
        action="store_true",
        help="After the loss figure, run one YOLO val(plots=True) and copy PR + confusion PNGs into a small folder.",
    )
    p.add_argument(
        "--val-plots-weights",
        type=Path,
        default=None,
        help="Weights for sidecar val plots (default: <run-dir>/weights/best.pt or last.pt).",
    )
    p.add_argument(
        "--val-plots-split",
        choices=("train", "val", "test"),
        default=None,
        help="Split for sidecar val (default: test if --csv-holdout test, else val).",
    )
    p.add_argument(
        "--val-plots-out",
        type=Path,
        default=None,
        help="Output folder for copied PNGs (default: <run-dir>/pr_cm_plots).",
    )
    p.add_argument(
        "--val-plots-imgsz",
        type=int,
        default=640,
        help="imgsz for sidecar val (default 640).",
    )
    p.add_argument(
        "--val-plots-batch",
        type=int,
        default=8,
        help="batch for sidecar val (default 8).",
    )
    args = p.parse_args()

    results_csv = args.results.expanduser().resolve()
    run_dir = results_csv.parent
    if args.out is None:
        args.out = run_dir / "loss_curves_train_holdout.png"
    args_yaml = run_dir / "args.yaml"
    with args_yaml.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_yaml = (
        args.data_yaml.expanduser().resolve()
        if args.data_yaml
        else _resolve_data_yaml(args_yaml, cfg["data"])
    )
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)

    out_png = args.out.expanduser().resolve()

    if args.mode == "val":
        plot_val_curves(results_csv, out_png, csv_holdout=args.csv_holdout)
    elif args.mode == "test-line":
        weights_pt = (
            args.weights.expanduser().resolve()
            if args.weights
            else (run_dir / "weights" / "last.pt")
        )
        if not weights_pt.is_file():
            raise FileNotFoundError(weights_pt)
        plot_test_line(results_csv, out_png, weights_pt=weights_pt, data_yaml=data_yaml, device=args.device)
    else:
        weights_dir = (
            args.weights_dir.expanduser().resolve()
            if args.weights_dir
            else (run_dir / "weights")
        )
        plot_test_curve(
            results_csv,
            out_png,
            weights_dir=weights_dir,
            data_yaml=data_yaml,
            device=args.device,
        )

    if args.with_val_plots:
        v_split = args.val_plots_split or ("test" if args.csv_holdout == "test" else "val")
        wpt = _pick_eval_weights(run_dir, args.val_plots_weights)
        dest = (args.val_plots_out or (run_dir / "pr_cm_plots")).expanduser().resolve()
        _minimal_pr_cm_plots(
            run_dir=run_dir,
            weights=wpt,
            data_yaml=data_yaml,
            split=v_split,
            dest_dir=dest,
            device=args.device,
            imgsz=args.val_plots_imgsz,
            batch=args.val_plots_batch,
        )
        print(f"PR + confusion matrix PNGs copied to:\n  {dest}")

    elif args.mode == "val" and args.csv_holdout == "test" and not args.with_val_plots:
        print(
            "plot_yolo_loss_curves: for <run-dir>/pr_cm_plots/ (PR + confusion), re-run with --with-val-plots",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

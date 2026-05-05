"""Ultralytics YOLOv11 training entrypoint with reproducible experiment presets (A/B/C)."""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

ExperimentId = str


def _max_epoch_from_results_csv(csv_path: Path) -> int:
    m = 0
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    e = int(float(row["epoch"]))
                except (KeyError, ValueError, TypeError):
                    continue
                m = max(m, e)
    except OSError:
        pass
    return m


def _last_time_from_results_csv(csv_path: Path) -> float:
    last = 0.0
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    last = float(row["time"])
                except (KeyError, ValueError, TypeError):
                    continue
    except OSError:
        pass
    return last


class SameDirFinetuneTrainer(DetectionTrainer):
    """Finetune into an existing run dir without wiping ``results.csv``; new rows continue epoch & time columns."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = dict(overrides or {})
        self._epoch_offset = 0
        self._time_offset = 0.0
        self._prior_csv_backup: Optional[Path] = None

        # Always preserve ``results.csv`` before ``BaseTrainer.__init__`` (it deletes the file when ``resume=False``).
        # ``get_cfg`` rejects HUB/internal keys like ``session``; BaseTrainer pops ``session`` separately.
        peek = dict(overrides)
        peek.pop("session", None)
        args_peek = get_cfg(cfg, peek)
        save_dir = get_save_dir(args_peek)
        csv_path = save_dir / "results.csv"
        if csv_path.is_file():
            self._epoch_offset = _max_epoch_from_results_csv(csv_path)
            self._time_offset = _last_time_from_results_csv(csv_path)
            self._prior_csv_backup = csv_path.parent / f"{csv_path.name}.aerosentry_bak"
            csv_path.rename(self._prior_csv_backup)

        super().__init__(cfg, overrides, _callbacks)

        if self._prior_csv_backup is not None:
            if self.csv.exists():
                self.csv.unlink()
            self._prior_csv_backup.rename(self.csv)

    def save_metrics(self, metrics):
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2
        t = time.time() - self.train_time_start + self._time_offset
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        epoch_out = self.epoch + 1 + self._epoch_offset
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with self.csv.open("a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (epoch_out, t, *vals)).rstrip(",") + "\n")


class InterceptorSameDirFinetuneTrainer(SameDirFinetuneTrainer):
    """Same as :class:`SameDirFinetuneTrainer` plus interceptor albumentations on train."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode == "train":
            setattr(self.args, "augmentations", build_interceptor_camera_augmentations())
        return DetectionTrainer.build_dataset(self, img_path, mode, batch)


def _set_global_seed(seed: int) -> None:
    """Fix RNG seeds for reproducible runs (best-effort on GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_interceptor_camera_augmentations() -> list:
    """Albumentations transforms mimicking interceptor optics and exposure stress.

    Wired into Ultralytics via ``cfg.augmentations`` (see ``v8_transforms`` / ``Albumentations``).

    Returns:
        List of instantiated albumentations transforms (non-spatial; no bbox_params needed).
    """
    import albumentations as A

    return [
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(7, 15), angle_range=(0, 360), p=1.0),
                A.GaussianBlur(blur_limit=(5, 9), p=1.0),
            ],
            p=0.65,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.45,
            contrast_limit=0.45,
            p=0.7,
        ),
        A.GaussNoise(var_limit=(20.0, 80.0), mean=0, p=0.55),
    ]


class InterceptorAlbumentationsTrainer(DetectionTrainer):
    """``DetectionTrainer`` that injects domain albumentations into ``hyp.augmentations``."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if mode == "train":
            setattr(self.args, "augmentations", build_interceptor_camera_augmentations())
        return super().build_dataset(img_path, mode, batch)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_train_kwargs(global_cfg: Dict[str, Any], experiment_cfg: Dict[str, Any]) -> Dict[str, Any]:
    train = dict(global_cfg.get("train", {}))
    exp_train = experiment_cfg.get("train", {})
    if isinstance(exp_train, dict):
        train.update(exp_train)
    return train


def _fallback_device_cpu_if_needed(train_kwargs: Dict[str, Any]) -> None:
    """If config asks for CUDA but this PyTorch build has no GPU, force ``device='cpu'``."""
    if torch.cuda.is_available():
        return
    d = train_kwargs.get("device")
    if d is None:
        train_kwargs["device"] = "cpu"
        return
    s = str(d).strip().lower()
    wants_cuda = (
        d in (0, 0.0)
        or s in ("0", "cuda", "cuda:0")
        or s.isdigit()
        or s.startswith("cuda")
    )
    if wants_cuda:
        import warnings

        warnings.warn(
            "CUDA is not available (torch.cuda.is_available() is False). "
            "Using CPU — training will be much slower. For GPU: install NVIDIA drivers "
            "and a CUDA-enabled PyTorch build (see https://pytorch.org/get-started/locally/). "
            "You can also set device: cpu in config/experiments.yaml.",
            UserWarning,
            stacklevel=2,
        )
        train_kwargs["device"] = "cpu"


def _apply_cpu_perf_defaults(train_kwargs: Dict[str, Any]) -> None:
    """Tune Ultralytics args for CPU / MPS (training is still slow vs GPU)."""
    dev = str(train_kwargs.get("device", "cpu")).lower()
    if dev not in ("cpu", "mps"):
        return
    # AMP on CPU adds overhead; Ultralytics still forces dataloader workers=0 on CPU (see their trainer).
    train_kwargs.setdefault("amp", False)


def _checkpoint_truly_resumable(path: Path) -> bool:
    """Ultralytics only continues the same run if epoch + optimizer state exist in the ``.pt`` file."""
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(ckpt, dict):
        return False
    return ckpt.get("epoch", -1) >= 0 and ckpt.get("optimizer") is not None


def run_training(
    config_path: Path,
    experiment: ExperimentId,
    extra_overrides: Optional[Dict[str, Any]] = None,
    *,
    resume_checkpoint: Optional[Path] = None,
) -> None:
    """Run a single training experiment (Ultralytics logs + plots under ``runs/detect/``).

    Sets ``WANDB_DISABLED=true`` so Ultralytics does not try to use Weights & Biases.

    Args:
        resume_checkpoint: If set, load weights from this ``.pt``. When the file contains a full
            training checkpoint (epoch ≥ 0 and optimizer state), Ultralytics continues the run
            with ``resume=True``. Otherwise a **fine-tune** is started (``resume=False``) in the
            **same** ``project`` / ``name`` folder as in YAML, with ``exist_ok=True``. For that
            fine-tune, ``results.csv`` is **appended** (epoch / time columns continue), and
            ``results.png`` is regenerated from the full CSV at the end. Learning-rate schedule
            still restarts unless the checkpoint is fully resumable.
    """
    os.environ["WANDB_DISABLED"] = "true"
    root = _load_yaml(config_path)
    experiments: Dict[str, Any] = root.get("experiments", {})
    key = experiment.strip().upper()
    if key not in experiments:
        raise KeyError(f"Unknown experiment '{experiment}'. Defined: {sorted(experiments)}")

    global_cfg: Dict[str, Any] = root.get("global", {})
    exp_block: Dict[str, Any] = experiments[key]

    seed = int(global_cfg.get("seed", 0))
    _set_global_seed(seed)

    data_yaml = str(Path(global_cfg["data"]).expanduser())
    exp_data = exp_block.get("data")
    if isinstance(exp_data, str) and exp_data.strip():
        data_yaml = str(Path(exp_data.strip()).expanduser())
    weights = str(Path(global_cfg.get("model", "yolo11n.pt")).expanduser())
    train_kwargs = _merge_train_kwargs(global_cfg, exp_block)
    train_kwargs.setdefault("seed", seed)
    train_kwargs.setdefault("plots", True)
    train_kwargs.setdefault("deterministic", True)

    proj = train_kwargs.pop("project", None) or "aerosentry"
    name = train_kwargs.pop("name", None) or f"exp_{key}_{Path(data_yaml).stem}"

    if extra_overrides:
        train_kwargs.update(extra_overrides)

    train_kwargs.pop("resume", None)

    _fallback_device_cpu_if_needed(train_kwargs)
    _apply_cpu_perf_defaults(train_kwargs)

    resume_path = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint else None
    resume_file_ok = bool(resume_path and resume_path.is_file())
    ckpt_truly_resumable = resume_file_ok and _checkpoint_truly_resumable(resume_path)  # type: ignore[arg-type]
    finetune_same_dir = resume_file_ok and not ckpt_truly_resumable

    trainer_cls: Optional[Type[DetectionTrainer]] = None
    if finetune_same_dir:
        if key == "B":
            trainer_cls = InterceptorSameDirFinetuneTrainer
        else:
            trainer_cls = SameDirFinetuneTrainer
    elif key == "B":
        trainer_cls = InterceptorAlbumentationsTrainer
    elif key == "C":
        # Hard negatives: ensure train set includes images with empty YOLO labels.
        # Ultralytics treats these as background; cls gain emphasizes score calibration vs background.
        pass

    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        model = YOLO(str(resume_path))
    else:
        model = YOLO(weights)

    if trainer_cls is not None:

        def _train_with_trainer(**kw: Any) -> Any:
            return model.train(trainer=trainer_cls, **kw)

        train_call = _train_with_trainer
    else:
        train_call = model.train

    if resume_path is not None:
        if ckpt_truly_resumable:
            train_call(data=data_yaml, resume=True, **train_kwargs)
        else:
            import warnings

            warnings.warn(
                f"{resume_path} has no optimizer/epoch training state — cannot resume in-place. "
                f"Fine-tuning in existing folder project={proj!s}, name={name!s} "
                f"(epochs={train_kwargs.get('epochs', '?')}; LR schedule starts over). "
                f"Appending metrics to results.csv in that folder.",
                UserWarning,
                stacklevel=2,
            )
            finetune_kw = dict(train_kwargs)
            finetune_kw["exist_ok"] = True
            train_call(
                data=data_yaml,
                project=str(proj),
                name=name,
                resume=False,
                **finetune_kw,
            )
    else:
        train_call(
            data=data_yaml,
            project=str(proj),
            name=name,
            **train_kwargs,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv11 with experiment presets A/B/C.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments.yaml"),
        help="Path to experiments YAML.",
    )
    p.add_argument(
        "--experiment",
        choices=["A", "B", "C", "a", "b", "c"],
        required=True,
        help="A=baseline, B=interceptor albumentations, C=hard-negative / FP-aware loss shaping.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Load weights from this .pt: full resume if checkpoint has optimizer+epoch; else fine-tune same run dir (append results.csv, refresh plots).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epoch count (e.g. 100 when resuming after 50). Merged into train kwargs.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extra: Dict[str, Any] = {}
    if args.epochs is not None:
        extra["epochs"] = int(args.epochs)
    run_training(
        args.config,
        args.experiment,
        extra if extra else None,
        resume_checkpoint=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""PyTorch → ONNX → TensorRT engine export with FP16 / INT8 hooks for Jetson-class deploy paths."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

import numpy as np


def _cuda_available() -> bool:
    """TensorRT via Ultralytics requires a working CUDA device in this workflow."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass
class CalibrationBatch:
    """One normalized batch for INT8 entropy calibration (NCHW float32 in ``[0,1]``)."""

    images: np.ndarray


class EntropyCalibratorStub:
    """Stub / design doc for TensorRT **post-training INT8 entropy calibration**.

    A production :class:`tensorrt.IInt8EntropyCalibrator2` implementation must:

    #. Implement ``get_batch_size``, ``get_batch`` (write bindings), ``read_calibration_cache`` /
       ``write_calibration_cache``.
    #. Feed **representative** frames (same pre-process as deploy) so histogram entropy minimization
       matches field statistics.

    Ultralytics already performs an internal calibration path when exporting ``format='engine'``
    with ``int8=True`` and a valid ``data`` YAML listing images. This class wraps a Python iterator
    so you can plug the same image list into a hand-rolled TensorRT ``trt.Builder`` workflow on
    Orin if you bypass Ultralytics.
    """

    def __init__(
        self,
        batch_iterator: Callable[[], Iterator[CalibrationBatch]],
        cache_file: Path,
    ) -> None:
        self._iter_factory = batch_iterator
        self.cache_file = Path(cache_file)
        self._iterator: Optional[Iterator[CalibrationBatch]] = None

    def reset_iterator(self) -> None:
        """Call once per calibration pass (TensorRT may run multiple epochs)."""
        self._iterator = self._iter_factory()

    def get_next_batch(self) -> Optional[np.ndarray]:
        """Return NCHW float batch or ``None`` when exhausted (stub API)."""
        if self._iterator is None:
            self.reset_iterator()
        assert self._iterator is not None
        try:
            batch = next(self._iterator)
        except StopIteration:
            return None
        return np.ascontiguousarray(batch.images, dtype=np.float32)


def _default_calibration_batches_from_dir(
    image_dir: Path,
    batch_size: int,
    imgsz: int,
) -> Iterator[CalibrationBatch]:
    """Yield calibration batches using OpenCV resize (CPU) — mirrors edge pre-process roughly."""
    import cv2

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in exts)
    batch: List[np.ndarray] = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        t = im.astype(np.float32) / 255.0
        t = np.transpose(t, (2, 0, 1))  # CHW
        batch.append(t)
        if len(batch) >= batch_size:
            yield CalibrationBatch(images=np.stack(batch, axis=0))
            batch = []
    if batch:
        yield CalibrationBatch(images=np.stack(batch, axis=0))


class EngineExporter:
    """Manage ``.pt`` → ONNX → ``.engine`` conversion (Ultralytics backend).

    Parameters ``fp16`` / ``int8`` map to TensorRT builder precision toggles where supported.
    ``nms=False`` requests graphs that defer NMS to CPU unless the backbone provides a fused
    end-to-end export path (YOLOv10 / specific heads — Ultralytics sets this per model family).

    Attributes:
        weights: Path to ``.pt`` checkpoint.
        output_dir: Directory for ONNX / engine artifacts.
    """

    def __init__(self, weights: Path, output_dir: Path) -> None:
        self.weights = Path(weights).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"Weights not found: {self.weights}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_onnx(
        self,
        *,
        imgsz: int = 640,
        opset: int = 17,
        simplify: bool = True,
        dynamic: bool = False,
        nms: bool = False,
    ) -> Path:
        """Export ONNX to ``output_dir``.

        Args:
            imgsz: Square inference size.
            opset: ONNX opset version.
            simplify: Run ONNX graph simplifier when ``True``.
            dynamic: Dynamic batch / shapes when supported.
            nms: Attempt to fold NMS in graph if exporter supports it.

        Returns:
            Path to the ``.onnx`` file.
        """
        from ultralytics import YOLO

        model = YOLO(str(self.weights))
        out = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            nms=nms,
            project=str(self.output_dir),
            name="onnx_export",
            exist_ok=True,
        )
        return Path(out)

    def export_tensorrt(
        self,
        *,
        imgsz: int = 640,
        fp16: bool = False,
        int8: bool = False,
        calibration_data_yaml: Optional[Path] = None,
        workspace_gb: float = 4.0,
        nms: bool = False,
        batch: int = 1,
        verbose: bool = False,
    ) -> Path:
        """Export TensorRT ``.engine`` (uses Ultralytics TensorRT builder / ``trtexec`` fallback).

        Args:
            imgsz: Square resolution baked into engine.
            fp16: Enable FP16 kernels.
            int8: Enable INT8 PTQ (requires ``calibration_data_yaml`` for calibration data).
            calibration_data_yaml: Ultralytics ``data.yaml`` listing calibration images.
            workspace_gb: Builder workspace budget (GiB).
            nms: Prefer end-to-end NMS-fused graph when backbone allows.
            batch: Static batch dimension when not dynamic.
            verbose: Verbose exporter logs.

        Returns:
            Path to ``.engine`` file.

        Raises:
            ValueError: If ``int8`` is ``True`` but no calibration YAML was provided.
        """
        if int8 and calibration_data_yaml is None:
            raise ValueError("INT8 export requires calibration_data_yaml (Ultralytics data YAML).")
        from ultralytics import YOLO

        model = YOLO(str(self.weights))
        kwargs: dict[str, Any] = dict(
            format="engine",
            imgsz=imgsz,
            half=fp16,
            int8=int8,
            workspace=workspace_gb,
            nms=nms,
            batch=batch,
            verbose=verbose,
            project=str(self.output_dir),
            name="trt_export",
            exist_ok=True,
        )
        if calibration_data_yaml is not None:
            kwargs["data"] = str(calibration_data_yaml)
        out = model.export(**kwargs)
        return Path(out)

    def export_full_pipeline(
        self,
        *,
        imgsz: int = 640,
        export_onnx_explicit: bool = True,
        fp16: bool = False,
        int8: bool = False,
        calibration_data_yaml: Optional[Path] = None,
        workspace_gb: float = 4.0,
        nms: bool = False,
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Optionally write ONNX, then build TensorRT engine.

        Returns:
            ``(onnx_path_or_none, engine_path_or_none)``. Engine is ``None`` if CUDA
            is unavailable (ONNX export can still run on CPU).
        """
        onnx_path: Optional[Path] = None
        if export_onnx_explicit:
            onnx_path = self.export_onnx(imgsz=imgsz, nms=nms)
        if not _cuda_available():
            print(
                "Note: CUDA is not available — skipping TensorRT `.engine` build.\n"
                "      You still have ONNX (use on Jetson / a GPU machine for "
                "`yolo export format=engine`, or deploy ONNX with ONNX Runtime / OpenVINO).",
                file=sys.stderr,
            )
            return onnx_path, None
        engine_path = self.export_tensorrt(
            imgsz=imgsz,
            fp16=fp16,
            int8=int8,
            calibration_data_yaml=calibration_data_yaml,
            workspace_gb=workspace_gb,
            nms=nms,
        )
        return onnx_path, engine_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("exports/tensorrt"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--calibration-data", type=Path, default=None, help="data.yaml for INT8 PTQ.")
    p.add_argument("--workspace-gb", type=float, default=4.0)
    p.add_argument("--nms-e2e", action="store_true", help="Request fused NMS in graph if supported.")
    p.add_argument("--skip-onnx", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ex = EngineExporter(args.weights, args.out)
    if args.skip_onnx:
        if not _cuda_available():
            print(
                "TensorRT export requires CUDA on this machine (`torch.cuda.is_available()` is False).",
                file=sys.stderr,
            )
            return 2
        path = ex.export_tensorrt(
            imgsz=args.imgsz,
            fp16=args.fp16,
            int8=args.int8,
            calibration_data_yaml=args.calibration_data,
            workspace_gb=args.workspace_gb,
            nms=args.nms_e2e,
        )
        print(f"Engine: {path}")
    else:
        onnx, eng = ex.export_full_pipeline(
            imgsz=args.imgsz,
            export_onnx_explicit=True,
            fp16=args.fp16,
            int8=args.int8,
            calibration_data_yaml=args.calibration_data,
            workspace_gb=args.workspace_gb,
            nms=args.nms_e2e,
        )
        if eng is None:
            print(f"ONNX: {onnx}\nEngine: (not built — no CUDA)")
        else:
            print(f"ONNX: {onnx}\nEngine: {eng}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

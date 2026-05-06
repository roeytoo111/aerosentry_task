"""Tactical benchmarking: end-to-end latency, tracking stability, and distractor TNR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from src.core.data_contracts import Detection, FrameData
from src.tracking.fp_suppressor import FalsePositiveSuppressor
from src.tracking.track_manager import iou_xyxy, xywhn_to_xyxy

if TYPE_CHECKING:
    pass


class DetectorProtocol(Protocol):
    """Minimal detector surface for tactical evaluation."""

    def predict(self, frame: np.ndarray) -> List[Detection]: ...


@dataclass
class TacticalLatencyReport:
    """Timing statistics in milliseconds (wall time with optional CUDA sync)."""

    mean_ms: float
    p95_ms: float
    std_ms: float
    fps_mean: float
    num_samples: int
    warmup: int
    device_note: str


@dataclass
class TrackingStabilityReport:
    """MOT-style identity switch count on a continuous clip."""

    identity_switches: int
    num_frames: int
    num_gt_tracks: int
    matched_frames: int


@dataclass
class DistractorReport:
    """Hard-negative purity after the FP suppressor."""

    tnr: float
    true_negatives: int
    false_positives: int
    total_images: int


@dataclass
class TacticalMetricRow:
    """One row in the tactical markdown table."""

    precision: float
    recall: float
    fdr: float
    tnr_distractor: float
    latency_p95_ms: float
    fps: float


class UltralyticsYoloDetector:
    """Ultralytics YOLO adapter producing :class:`Detection` in normalized xywh."""

    def __init__(
        self,
        weights: Path | str,
        device: str = "0",
        imgsz: int = 640,
        conf: float = 0.25,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        self._device = device
        self._imgsz = int(imgsz)
        self._conf = float(conf)

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run one forward and convert boxes to normalized ``cx,cy,w,h``."""
        res = self._model.predict(
            frame,
            imgsz=self._imgsz,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )[0]
        h, w = frame.shape[:2]
        out: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        names: Dict[int, str] = res.names
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            bw = float((x2 - x1) / max(w, 1))
            bh = float((y2 - y1) / max(h, 1))
            cx = float(((x1 + x2) / 2.0) / max(w, 1))
            cy = float(((y1 + y2) / 2.0) / max(h, 1))
            cid = int(cls[i])
            out.append(
                Detection(
                    class_id=cid,
                    class_label=str(names.get(cid, str(cid))),
                    confidence=float(confs[i]),
                    bbox=[cx, cy, bw, bh],
                )
            )
        return out


class TacticalEvaluator:
    """Full-chain evaluator: raw detector + :class:`FalsePositiveSuppressor`.

    Latency uses ``torch.cuda.synchronize()`` around timed regions on CUDA so asynchronous
    launches cannot hide true kernel + transfer cost (Jetson-class concern).

    Args:
        detector: Callable with ``predict(frame) -> List[Detection]``.
        suppressor_factory: Builds a **fresh** :class:`FalsePositiveSuppressor` per distractor image
            when evaluating TNR to avoid cross-image track bleed.
        device_tag: Human-readable device string for reports.
    """

    def __init__(
        self,
        detector: DetectorProtocol,
        suppressor_factory: Callable[[], FalsePositiveSuppressor],
        *,
        device_tag: str = "cuda:0",
    ) -> None:
        self._detector = detector
        self._suppressor_factory = suppressor_factory
        self._default_suppressor = suppressor_factory()
        self.device_tag = device_tag

    def _sync_cuda(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _run_e2e(self, frame: np.ndarray, suppressor: FalsePositiveSuppressor) -> FrameData:
        dets = self._detector.predict(frame)
        fd = FrameData(
            frame=frame,
            frame_id=0,
            timestamp=0.0,
            detections=dets,
        )
        return suppressor.process(fd)

    def evaluate_end_to_end_latency(
        self,
        frames: Sequence[np.ndarray],
        *,
        warmup: int = 20,
        iterations: int = 100,
        reuse_suppressor: bool = True,
    ) -> TacticalLatencyReport:
        """Measure detector + suppressor latency with CUDA sync.

        Args:
            frames: Looping source frames (BGR ``uint8``).
            warmup: Batches not timed.
            iterations: Timed iterations (frames cycle ``frames``).
            reuse_suppressor: If ``True``, one suppressor instance (stateful / realistic).

        Returns:
            :class:`TacticalLatencyReport` with **p95** latency, not only mean.
        """
        import time
        import torch

        if len(frames) == 0:
            raise ValueError("frames must be non-empty")
        sup = self._default_suppressor if reuse_suppressor else self._suppressor_factory()

        for i in range(warmup):
            self._sync_cuda()
            self._run_e2e(frames[i % len(frames)], sup)

        times_ms: List[float] = []
        for i in range(iterations):
            idx = i % len(frames)

            self._sync_cuda()
            t0 = time.perf_counter()
            self._run_e2e(frames[idx], sup)
            self._sync_cuda()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        arr = np.asarray(times_ms, dtype=np.float64)
        mean = float(arr.mean())
        p95 = float(np.percentile(arr, 95))
        std = float(arr.std())
        fps = 1000.0 / mean if mean > 1e-6 else 0.0
        dev = (
            "CUDA (synchronized)"
            if torch.cuda.is_available()
            else "CPU (no CUDA sync effect)"
        )
        return TacticalLatencyReport(
            mean_ms=mean,
            p95_ms=p95,
            std_ms=std,
            fps_mean=fps,
            num_samples=iterations,
            warmup=warmup,
            device_note=dev,
        )

    def evaluate_tracking_stability(
        self,
        frames: Sequence[np.ndarray],
        gt_per_frame: Sequence[Sequence[Tuple[int, np.ndarray]]],
        timestamps: Optional[Sequence[float]] = None,
    ) -> TrackingStabilityReport:
        """Count **identity switches** (IDSW) against sparse GT track IDs.

        Args:
            frames: Video frames.
            gt_per_frame: For each frame, a list of ``(gt_track_id, xywhn ndarray)`` entries.
            timestamps: Optional monotonic seconds per frame; defaults to frame index * 0.033.

        Returns:
            Report with cumulative IDSW tally.
        """
        if len(frames) != len(gt_per_frame):
            raise ValueError("frames and gt_per_frame length mismatch")

        sup = self._suppressor_factory()
        last_pred_tid: Dict[int, Optional[int]] = {}
        idsw = 0
        matched_frames = 0

        all_gt_ids = {gid for gtf in gt_per_frame for gid, _ in gtf}
        num_gt_tracks = len(all_gt_ids)

        for fi, frame in enumerate(frames):
            ts = float(fi * (1.0 / 30.0)) if timestamps is None else float(timestamps[fi])
            dets = self._detector.predict(frame)
            fd = FrameData(
                frame=frame,
                frame_id=fi,
                timestamp=ts,
                detections=dets,
            )
            out_fd = sup.process(fd)
            preds = out_fd.detections

            gt_list = list(gt_per_frame[fi])
            if not gt_list:
                continue

            pred_xyxy = [
                xywhn_to_xyxy(np.asarray(p.bbox, dtype=np.float64)) for p in preds
            ]
            gt_xyxy = [xywhn_to_xyxy(b) for _, b in gt_list]

            assigned_pred: Dict[int, int] = {}
            used_pred = set()
            for gi, (gt_id, _) in enumerate(gt_list):
                best_j = -1
                best_iou = 0.25
                for pj, pb in enumerate(pred_xyxy):
                    if pj in used_pred:
                        continue
                    iou_v = iou_xyxy(gt_xyxy[gi], pb)
                    if iou_v > best_iou:
                        best_iou = iou_v
                        best_j = pj
                if best_j >= 0:
                    used_pred.add(best_j)
                    assigned_pred[gt_id] = best_j

            for gt_id, pj in assigned_pred.items():
                pred_track_id = preds[pj].track_id
                if pred_track_id is None:
                    continue
                prev = last_pred_tid.get(gt_id)
                if prev is not None and prev != pred_track_id:
                    idsw += 1
                last_pred_tid[gt_id] = int(pred_track_id)

            if assigned_pred:
                matched_frames += 1

        return TrackingStabilityReport(
            identity_switches=idsw,
            num_frames=len(frames),
            num_gt_tracks=num_gt_tracks,
            matched_frames=matched_frames,
        )

    def evaluate_distractor_rejection(
        self,
        image_paths: Sequence[Path],
        *,
        reset_suppressor_per_frame: bool = True,
    ) -> DistractorReport:
        """True Negative Rate on a folder of hard-negative images (no target should persist).

        A **true negative** means the **post-suppressor** output has zero detections.

        Args:
            image_paths: Raster image paths (PNG/JPG).
            reset_suppressor_per_frame: Use a fresh suppressor per image (recommended).

        Returns:
            :class:`DistractorReport` with ``tnr = TN / total``.
        """
        import cv2

        tn = fp = 0
        sup = self._suppressor_factory()
        for p in image_paths:
            if reset_suppressor_per_frame:
                sup = self._suppressor_factory()
            im = cv2.imread(str(p))
            if im is None:
                continue
            out = self._run_e2e(im, sup)
            if len(out.detections) == 0:
                tn += 1
            else:
                fp += 1
        total = tn + fp
        tnr = float(tn / total) if total > 0 else 0.0
        return DistractorReport(
            tnr=tnr,
            true_negatives=tn,
            false_positives=fp,
            total_images=total,
        )


__all__ = [
    "DistractorReport",
    "TacticalEvaluator",
    "TacticalLatencyReport",
    "TacticalMetricRow",
    "TrackingStabilityReport",
    "UltralyticsYoloDetector",
]

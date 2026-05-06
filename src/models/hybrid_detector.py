"""Cascaded inference: YOLOv11 primary with RT-DETR fallback (confidence-guided switching).

This module implements a small state machine:
default YOLO pass, optional RT-DETR on the *same* frame when uncertainty or
track-loss conditions fire, then a cooldown to avoid model oscillation.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.data_contracts import Detection


class HybridDetectorState(Enum):
    """Internal high-level mode (FALLBACK is transient within :meth:`HybridDetector.detect`)."""

    YOLO = auto()
    COOLDOWN = auto()


class HybridDetector:
    """Facade-style detector: YOLO + optional RT-DETR fallback, output ``(N, 6)`` ``xyxy``+conf+cls.

    The public :meth:`detect` API returns NumPy arrays directly consumable by geometric stages
    that expect pixel-space boxes. Use :meth:`xyxy6_to_detections` to adapt to
    :class:`Detection` (normalized xywh) for the rest of the AeroSentry pipeline.

    Args:
        yolo_weights: Path or Ultralytics hub id for the primary YOLO checkpoint.
        rtdetr_weights: Path or hub id for RT-DETR (Transformer backbone via Ultralytics).
        device: Ultralytics device string (``\"0\"``, ``\"cpu\"``, etc.).
        imgsz: Inference square size (shared default; can widen RT-DETR separately later).
        yolo_conf: Score threshold fed to YOLO ``predict`` (boxes below this are dropped).
        rtdetr_conf: Score threshold for RT-DETR when fallback runs.
        uncertainty_thresh: Condition A — if YOLO yields ≥1 box and ``max(conf)`` is **below**
            this value, run RT-DETR on the same frame.
        cooldown_frames: After a fallback RT-DETR run, ignore fallback triggers for this many
            :meth:`detect` calls while still running YOLO each time.
    """

    def __init__(
        self,
        yolo_weights: str,
        rtdetr_weights: str = "rtdetr-l.pt",
        *,
        device: str = "0",
        imgsz: int = 640,
        yolo_conf: float = 0.25,
        rtdetr_conf: float = 0.25,
        uncertainty_thresh: float = 0.55,
        cooldown_frames: int = 10,
    ) -> None:
        from ultralytics import YOLO

        self._device = device
        self._imgsz = int(imgsz)
        self._yolo_conf = float(yolo_conf)
        self._rtdetr_conf = float(rtdetr_conf)
        self._uncertainty_thresh = float(uncertainty_thresh)
        self._cooldown_frames = int(cooldown_frames)

        # Single-load policy: both models resident for the detector lifetime.
        self._yolo = YOLO(str(yolo_weights))
        self._rtdetr = YOLO(str(rtdetr_weights))

        self._cooldown_remaining = 0
        self._last_public_state = HybridDetectorState.YOLO
        self._used_rtdetr_last = False

    @property
    def used_rtdetr_last(self) -> bool:
        """True if the immediately previous :meth:`detect` finished via the RT-DETR branch."""
        return self._used_rtdetr_last

    @property
    def cooldown_remaining(self) -> int:
        """Frames left in cooldown (0 = normal YOLO-only evaluation for triggers)."""
        return self._cooldown_remaining

    @property
    def last_state(self) -> HybridDetectorState:
        """State exposed after the most recent :meth:`detect` (YOLO or COOLDOWN)."""
        return self._last_public_state

    def detect(self, frame: np.ndarray, is_track_lost: bool = False) -> np.ndarray:
        """Run cascaded inference for one BGR frame.

        Args:
            frame: OpenCV BGR image, shape ``(H, W, 3)``.
            is_track_lost: Condition B — if True and YOLO returns zero boxes, run RT-DETR.

        Returns:
            Array of shape ``(N, 6)`` and dtype ``float32``: columns
            ``x1, y1, x2, y2, confidence, class_id`` (``class_id`` is integral, stored as float).
        """
        if frame is None or frame.size == 0:
            self._used_rtdetr_last = False
            return np.zeros((0, 6), dtype=np.float32)

        self._used_rtdetr_last = False
        # COOLDOWN: force YOLO path only; do not evaluate fallback triggers this frame.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._last_public_state = HybridDetectorState.COOLDOWN
            return self._predict_yolo_xyxy6(frame)

        self._last_public_state = HybridDetectorState.YOLO
        yolo_out = self._predict_yolo_xyxy6(frame)

        if self._need_rtdetr(yolo_out, is_track_lost):
            # FALLBACK: discard YOLO output for this frame; heavy model on same pixels.
            rtdetr_out = self._predict_rtdetr_xyxy6(frame)
            self._cooldown_remaining = self._cooldown_frames
            self._used_rtdetr_last = True
            return rtdetr_out

        return yolo_out

    def predict_yolo_only(self, frame: np.ndarray) -> np.ndarray:
        """Run **only** the primary YOLO branch (same weights/imgsz/conf as the facade).

        Ignores cooldown and RT-DETR. Use to compare micro TP/FP/FN against :meth:`detect`
        on the same video when ground-truth boxes are available.
        """
        if frame is None or frame.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return self._predict_yolo_xyxy6(frame)

    def _need_rtdetr(self, yolo_xyxy6: np.ndarray, is_track_lost: bool) -> bool:
        n = int(yolo_xyxy6.shape[0])
        # B: no YOLO boxes but tracker believes the target should still be visible.
        if n == 0:
            return bool(is_track_lost)
        # A: at least one box, but best score is below semantic certainty threshold.
        max_conf = float(np.max(yolo_xyxy6[:, 4]))
        return max_conf < self._uncertainty_thresh

    def _predict_yolo_xyxy6(self, frame: np.ndarray) -> np.ndarray:
        res = self._yolo.predict(
            frame,
            imgsz=self._imgsz,
            conf=self._yolo_conf,
            device=self._device,
            verbose=False,
        )[0]
        return _ultralytics_result_to_xyxy6(res)

    def _predict_rtdetr_xyxy6(self, frame: np.ndarray) -> np.ndarray:
        res = self._rtdetr.predict(
            frame,
            imgsz=self._imgsz,
            conf=self._rtdetr_conf,
            device=self._device,
            verbose=False,
        )[0]
        return _ultralytics_result_to_xyxy6(res)


def _ultralytics_result_to_xyxy6(res: object) -> np.ndarray:
    """Convert one Ultralytics ``Results`` to ``(N, 6)`` without redundant copies."""
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
    cls = boxes.cls.cpu().numpy().astype(np.float32).reshape(-1, 1)
    return np.concatenate([xyxy, conf, cls], axis=1)


def xyxy6_to_detections(
    xyxy6: np.ndarray,
    frame_shape: Tuple[int, int, int],
    class_names: Optional[Dict[int, str]] = None,
) -> List[Detection]:
    """Convert ``(N,6)`` pixel boxes to normalized :class:`Detection` list.

    Args:
        xyxy6: Output of :meth:`HybridDetector.detect`.
        frame_shape: ``(H, W, C)`` from the source frame.
        class_names: Optional id→label map; ids fall back to stringified integers.

    Returns:
        List suitable for :class:`FrameData` / :class:`FalsePositiveSuppressor`.
    """
    if xyxy6.size == 0:
        return []
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h <= 0 or w <= 0:
        return []

    out: List[Detection] = []
    names = class_names or {}
    n = xyxy6.shape[0]
    for i in range(n):
        x1, y1, x2, y2, conf, cid_f = (float(v) for v in xyxy6[i])
        cid = int(round(cid_f))
        bw = max((x2 - x1) / float(w), 1e-6)
        bh = max((y2 - y1) / float(h), 1e-6)
        cx = ((x1 + x2) * 0.5) / float(w)
        cy = ((y1 + y2) * 0.5) / float(h)
        out.append(
            Detection(
                class_id=cid,
                class_label=str(names.get(cid, str(cid))),
                confidence=float(conf),
                bbox=[cx, cy, bw, bh],
            )
        )
    return out


__all__ = [
    "HybridDetector",
    "HybridDetectorState",
    "xyxy6_to_detections",
]

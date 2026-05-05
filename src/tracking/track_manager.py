"""Multi-object association and M-of-N temporal voting for detection tracks (CPU-only)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.tracking.filters import BoundingBoxOneEuroFilter


def xywhn_to_xyxy(b: np.ndarray, clip: bool = True) -> np.ndarray:
    """Convert normalized ``cx,cy,w,h`` to ``x1,y1,x2,y2`` in the same [0,1] canvas."""
    cx, cy, w, h = b[0], b[1], b[2], b[3]
    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    box = np.array([x1, y1, x2, y2], dtype=np.float64)
    if clip:
        box = np.clip(box, 0.0, 1.0)
    return box


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for two boxes ``[x1,y1,x2,y2]``."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ar = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    br = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ar + br - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class _TrackState:
    track_id: int
    class_id: int
    class_label: str
    last_confidence: float
    bbox_filter: BoundingBoxOneEuroFilter
    vote_window: Deque[bool]
    miss_streak: int = 0
    last_smoothed_xywh: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )


class TrackManager:
    """Greedy IoU + class matching with **M-of-N** positive votes for confirmation.

    Tentative tracks accrue a sliding window of hits; ghosts that fail to reappear are purged
    after ``max_miss_streak`` consecutive misses.

    Args:
        vote_m: Minimum number of positive frames required within the last ``vote_n`` frames.
        vote_n: Sliding window length (frames).
        iou_threshold: Minimum IoU to associate a detection with a track.
        max_miss_streak: Delete track after this many unmatched frames.
        euro_min_cutoff: Passed to :class:`BoundingBoxOneEuroFilter`.
        euro_beta: Passed to :class:`BoundingBoxOneEuroFilter`.
        euro_d_cutoff: Passed to :class:`BoundingBoxOneEuroFilter`.
    """

    def __init__(
        self,
        vote_m: int = 5,
        vote_n: int = 7,
        iou_threshold: float = 0.3,
        max_miss_streak: int = 5,
        euro_min_cutoff: float = 1.0,
        euro_beta: float = 0.007,
        euro_d_cutoff: float = 1.0,
    ) -> None:
        if vote_m > vote_n or vote_m < 1 or vote_n < 1:
            raise ValueError("Require 1 <= vote_m <= vote_n")
        self._vote_m = int(vote_m)
        self._vote_n = int(vote_n)
        self._iou_threshold = float(iou_threshold)
        self._max_miss = int(max_miss_streak)
        self._euro_kw = dict(
            min_cutoff=euro_min_cutoff, beta=euro_beta, d_cutoff=euro_d_cutoff
        )
        self._tracks: Dict[int, _TrackState] = {}
        self._next_id = 0

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 0

    @staticmethod
    def is_track_confirmed(vote_window: Deque[bool], m: int, n: int) -> bool:
        """``True`` if the window is full and contains at least ``m`` hits."""
        if len(vote_window) < n:
            return False
        return sum(1 for x in vote_window if x) >= m

    def update(
        self,
        detections_xywh: List[np.ndarray],
        class_ids: List[int],
        class_labels: List[str],
        timestamp: float,
        confidences: Optional[List[float]] = None,
    ) -> Dict[int, Tuple[np.ndarray, bool, bool, str, int, float]]:
        n_det = len(detections_xywh)
        if confidences is None:
            confidences = [1.0] * n_det
        if len(class_ids) != n_det or len(class_labels) != n_det:
            raise ValueError("class_ids and class_labels length must match detections_xywh")

        order = sorted(range(n_det), key=lambda i: -confidences[i])
        det_boxes = [
            np.asarray(detections_xywh[i], dtype=np.float64).reshape(4) for i in range(n_det)
        ]
        det_xyxy = [xywhn_to_xyxy(d) for d in det_boxes]

        matched_track: Dict[int, int] = {}  # track_id -> det_idx

        for tr in self._tracks.values():
            tr.miss_streak += 1

        for di in order:
            cls = class_ids[di]
            best_tid: Optional[int] = None
            best_iou = self._iou_threshold
            for tid, tr in self._tracks.items():
                if tid in matched_track:
                    continue
                if tr.class_id != cls:
                    continue
                iou = iou_xyxy(det_xyxy[di], xywhn_to_xyxy(tr.last_smoothed_xywh))
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                matched_track[best_tid] = di
                tr = self._tracks[best_tid]
                tr.miss_streak = 0
                sm = np.array(
                    tr.bbox_filter.filter(tuple(det_boxes[di].tolist()), timestamp),
                    dtype=np.float64,
                )
                tr.last_smoothed_xywh = sm
                tr.class_label = class_labels[di]
                tr.class_id = cls
                tr.last_confidence = confidences[di]
                tr.vote_window.append(True)
                while len(tr.vote_window) > self._vote_n:
                    tr.vote_window.popleft()
            else:
                tid = self._next_id
                self._next_id += 1
                filt = BoundingBoxOneEuroFilter(**self._euro_kw)
                sm = np.array(
                    filt.filter(tuple(det_boxes[di].tolist()), timestamp),
                    dtype=np.float64,
                )
                vw: Deque[bool] = deque()
                vw.append(True)
                self._tracks[tid] = _TrackState(
                    track_id=tid,
                    class_id=cls,
                    class_label=class_labels[di],
                    last_confidence=confidences[di],
                    bbox_filter=filt,
                    vote_window=vw,
                    miss_streak=0,
                    last_smoothed_xywh=sm,
                )
                matched_track[tid] = di

        for tid, tr in list(self._tracks.items()):
            if tid not in matched_track:
                tr.vote_window.append(False)
                while len(tr.vote_window) > self._vote_n:
                    tr.vote_window.popleft()

        dead = [
            tid for tid, tr in self._tracks.items() if tr.miss_streak > self._max_miss
        ]
        for tid in dead:
            del self._tracks[tid]

        out: Dict[int, Tuple[np.ndarray, bool, bool, str, int, float]] = {}
        for tid, tr in self._tracks.items():
            hit = tid in matched_track
            conf = self.is_track_confirmed(tr.vote_window, self._vote_m, self._vote_n)
            out[tid] = (
                tr.last_smoothed_xywh.copy(),
                hit,
                conf,
                tr.class_label,
                tr.class_id,
                tr.last_confidence,
            )

        return out


__all__ = ["TrackManager", "iou_xyxy", "xywhn_to_xyxy"]

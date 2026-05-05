"""Facade: temporal voting, kinematic smoothing, and geometric FP gating (epipolar / homography)."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from src.core.data_contracts import Detection, FrameData
from src.tracking.geometric_ego_motion import GeometricEgoMotion
from src.tracking.track_manager import TrackManager


class FalsePositiveSuppressor:
    """Gatekeeper stage: suppress single-frame ghosts, billboards, and static clutter.

    Composes :class:`TrackManager` (M-of-N + IoU + One Euro per track). For **confirmed**
    tracks, runs :class:`GeometricEgoMotion` once per frame pair (CUDA ORB + RANSAC
    :math:`F` / :math:`H`) and culls ROI that move epipolar-consistently with the
    dominant background or behave as planar / 2D media.

    Tentative (unconfirmed) hits are not emitted; only confirmed tracks pass through.

    Args:
        track_manager: Optional custom :class:`TrackManager`.
        geo_estimator: Optional custom :class:`GeometricEgoMotion`.
    """

    def __init__(
        self,
        track_manager: Optional[TrackManager] = None,
        geo_estimator: Optional[GeometricEgoMotion] = None,
    ) -> None:
        self._tracks = track_manager or TrackManager()
        self.geo_estimator = geo_estimator or GeometricEgoMotion()
        self._prev_bgr: Optional[np.ndarray] = None
        self._prev_smoothed_xywh: Dict[int, np.ndarray] = {}

    def reset(self) -> None:
        """Reset tracks and frame history (e.g. new video source)."""
        self._tracks.reset()
        self._prev_bgr = None
        self._prev_smoothed_xywh.clear()

    def process(self, frame_data: FrameData) -> FrameData:
        """Return a new :class:`FrameData` with filtered ``detections`` (confirmed + airborne).

        Args:
            frame_data: Frame with raw detector outputs in ``detections``.

        Returns:
            Shallow-copied frame metadata with a fresh detection list.
        """
        dets = frame_data.detections
        if not dets:
            self._tracks.update([], [], [], frame_data.timestamp, [])
            self._advance_frame_cache(frame_data, {})
            return FrameData(
                frame=frame_data.frame,
                frame_id=frame_data.frame_id,
                timestamp=frame_data.timestamp,
                detections=[],
            )

        xywh = [np.asarray(list(d.bbox), dtype=np.float64) for d in dets]
        cids = [d.class_id for d in dets]
        clabels = [d.class_label for d in dets]
        confs = [float(d.confidence) for d in dets]

        states = self._tracks.update(
            xywh, cids, clabels, frame_data.timestamp, confs
        )

        confirmed_hits = {
            tid for tid, row in states.items() if row[1] and row[2]
        }

        geo_verbose = getattr(self.geo_estimator, "_verbose", False)
        if geo_verbose and (
            not confirmed_hits
            or self._prev_bgr is None
            or frame_data.frame is None
        ):
            parts: list[str] = []
            if not confirmed_hits:
                parts.append("no_confirmed_tracks")
            if self._prev_bgr is None:
                parts.append("no_prev_bgr")
            if frame_data.frame is None:
                parts.append("no_frame")
            print(
                f"[FalsePositiveSuppressor] geo skipped frame_id={frame_data.frame_id}: "
                f"{', '.join(parts)}",
                flush=True,
            )

        geo_pts_prev: Optional[np.ndarray] = None
        geo_pts_curr: Optional[np.ndarray] = None
        mask_F: Optional[np.ndarray] = None
        mask_H: Optional[np.ndarray] = None

        if (
            confirmed_hits
            and self._prev_bgr is not None
            and frame_data.frame is not None
        ):
            try:
                geo_pts_prev, geo_pts_curr = self.geo_estimator._extract_and_match_cuda(
                    self._prev_bgr,
                    frame_data.frame,
                    frame_id=frame_data.frame_id,
                )
                if (
                    geo_pts_prev is not None
                    and geo_pts_curr is not None
                    and len(geo_pts_prev) >= 8
                ):
                    mask_F, mask_H = self.geo_estimator._compute_global_models(
                        geo_pts_prev,
                        geo_pts_curr,
                        frame_id=frame_data.frame_id,
                    )
            except (cv2.error, ValueError, RuntimeError):
                mask_F, mask_H = None, None

        survivors: list[Detection] = []
        for tid, row in states.items():
            sm_xywh, hit, is_conf, label, cid, conf = row
            if not hit:
                continue
            if not is_conf:
                continue

            is_airborne = True
            bbox_tuple = (
                float(sm_xywh[0]),
                float(sm_xywh[1]),
                float(sm_xywh[2]),
                float(sm_xywh[3]),
            )

            if tid in confirmed_hits:
                if (
                    geo_pts_curr is not None
                    and mask_F is not None
                    and mask_H is not None
                ):
                    is_airborne = self.geo_estimator.analyze_bbox_motion(
                        bbox_tuple,
                        frame_data.frame.shape,
                        geo_pts_curr,
                        mask_F,
                        mask_H,
                        debug_tag=f"frame_id={frame_data.frame_id} track_id={tid}",
                    )

            if not is_airborne:
                continue

            survivors.append(
                Detection(
                    class_id=int(cid),
                    class_label=label,
                    confidence=float(conf),
                    bbox=bbox_tuple,
                    track_id=int(tid),
                )
            )

        self._advance_frame_cache(frame_data, states)
        return FrameData(
            frame=frame_data.frame,
            frame_id=frame_data.frame_id,
            timestamp=frame_data.timestamp,
            detections=survivors,
        )

    def _advance_frame_cache(
        self,
        frame_data: FrameData,
        states: Dict[int, tuple],
    ) -> None:
        """Store last BGR frame and per-track smoothed boxes for pairing with the next frame."""
        self._prev_bgr = np.ascontiguousarray(frame_data.frame, dtype=np.uint8).copy()
        self._prev_smoothed_xywh = {
            tid: row[0].copy()
            for tid, row in states.items()
        }


__all__ = ["FalsePositiveSuppressor"]

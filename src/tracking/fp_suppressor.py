"""Facade: temporal voting, kinematic smoothing, and geometric FP gating (epipolar / homography)."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from src.core.data_contracts import Detection, FrameData
from src.tracking.geometric_ego_motion import GeometricEgoMotion
from src.tracking.track_manager import TrackManager


class FalsePositiveSuppressor:
    """Gatekeeper stage: optional temporal tracks + geometric ego-motion gating.

    **Default (full):** :class:`TrackManager` (M-of-N + IoU + One Euro). For **confirmed**
    tracks, runs :class:`GeometricEgoMotion` once per frame pair (ORB + RANSAC
    :math:`F` / :math:`H`) and culls ROI consistent with dominant background / planar clutter.

    **Geo-only:** no TrackManager — each frame's raw detector boxes pass through the same
    global :math:`F` / :math:`H` gate using **unsmoothed** boxes. Use for ablation (is
    geometry alone helping FP?).

    Args:
        track_manager: Optional custom :class:`TrackManager` (ignored if ``geo_only=True``).
        geo_estimator: Optional custom :class:`GeometricEgoMotion`.
        geo_only: If ``True``, skip tracking; apply geometry per detection every frame
            (after the first; frame 0 has no previous image).
    """

    def __init__(
        self,
        track_manager: Optional[TrackManager] = None,
        geo_estimator: Optional[GeometricEgoMotion] = None,
        *,
        geo_only: bool = False,
    ) -> None:
        self._geo_only = bool(geo_only)
        self._tracks: Optional[TrackManager]
        if self._geo_only:
            self._tracks = None
        else:
            self._tracks = track_manager or TrackManager()
        self.geo_estimator = geo_estimator or GeometricEgoMotion()
        self._prev_bgr: Optional[np.ndarray] = None
        self._prev_smoothed_xywh: Dict[int, np.ndarray] = {}

    def reset(self) -> None:
        """Reset tracks and frame history (e.g. before a new video)."""
        if self._tracks is not None:
            self._tracks.reset()
        self._prev_bgr = None
        self._prev_smoothed_xywh.clear()

    def process(self, frame_data: FrameData) -> FrameData:
        if self._geo_only:
            return self._process_geo_only(frame_data)
        return self._process_with_tracks(frame_data)

    def _process_geo_only(self, frame_data: FrameData) -> FrameData:
        """YOLO boxes → global F/H → per-box keep/drop; no temporal confirmation."""
        dets = frame_data.detections
        frame = frame_data.frame
        geo_verbose = getattr(self.geo_estimator, "_verbose", False)

        if frame is None:
            return FrameData(
                frame=frame_data.frame,
                frame_id=frame_data.frame_id,
                timestamp=frame_data.timestamp,
                detections=[],
            )

        if not dets:
            if geo_verbose:
                print(
                    f"[FalsePositiveSuppressor] geo_only frame_id={frame_data.frame_id}: "
                    "no detections",
                    flush=True,
                )
            self._prev_bgr = np.ascontiguousarray(frame, dtype=np.uint8).copy()
            return FrameData(
                frame=frame_data.frame,
                frame_id=frame_data.frame_id,
                timestamp=frame_data.timestamp,
                detections=[],
            )

        geo_pts_prev: Optional[np.ndarray] = None
        geo_pts_curr: Optional[np.ndarray] = None
        mask_F: Optional[np.ndarray] = None
        mask_H: Optional[np.ndarray] = None

        if self._prev_bgr is not None:
            try:
                geo_pts_prev, geo_pts_curr = self.geo_estimator._extract_and_match_cuda(
                    self._prev_bgr,
                    frame,
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
        elif geo_verbose:
            print(
                f"[FalsePositiveSuppressor] geo_only skipped frame_id={frame_data.frame_id}: "
                "no_prev_bgr",
                flush=True,
            )

        survivors: list[Detection] = []
        for idx, d in enumerate(dets):
            bbox_tuple = (
                float(d.bbox[0]),
                float(d.bbox[1]),
                float(d.bbox[2]),
                float(d.bbox[3]),
            )
            is_airborne = True
            if (
                geo_pts_curr is not None
                and mask_F is not None
                and mask_H is not None
            ):
                is_airborne = self.geo_estimator.analyze_bbox_motion(
                    bbox_tuple,
                    frame.shape,
                    geo_pts_curr,
                    mask_F,
                    mask_H,
                    debug_tag=f"frame_id={frame_data.frame_id} det_idx={idx} geo_only",
                )
            if not is_airborne:
                continue
            survivors.append(
                Detection(
                    class_id=int(d.class_id),
                    class_label=d.class_label,
                    confidence=float(d.confidence),
                    bbox=bbox_tuple,
                    track_id=d.track_id,
                )
            )

        self._prev_bgr = np.ascontiguousarray(frame, dtype=np.uint8).copy()
        return FrameData(
            frame=frame_data.frame,
            frame_id=frame_data.frame_id,
            timestamp=frame_data.timestamp,
            detections=survivors,
        )

    def _process_with_tracks(self, frame_data: FrameData) -> FrameData:
        assert self._tracks is not None
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

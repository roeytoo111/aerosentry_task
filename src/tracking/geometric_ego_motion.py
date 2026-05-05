"""Global two-view geometry for ego-motion–consistent false positive rejection (ORB + RANSAC).

**CUDA is not required.** On a typical laptop (no ``cv2.cuda``), the code uses **CPU ORB** and
``BFMatcher`` — same logic as the GPU path, only slower. Logs show ``path=CPU (no cv2.cuda)``.
CUDA only accelerates descriptor extraction + matching when OpenCV is built with it (e.g. some Jetson
images); :math:`F` / :math:`H` fitting stays on the CPU either way.

We model inter-frame motion of the **entire scene** with:

* **Fundamental matrix** :math:`F` — epipolar geometry between two calibrated views. Inlier matches
  whose displacement is **consistent with a single rigid / general 3D scene + camera motion** lie on
  epipolar lines. Tracks whose ROI keypoints are **mostly** :math:`F`-inliers tend to move **with**
  the dominant background (static world as seen from a moving camera → "grounded" clutter / poster).
* **Homography** :math:`H` — exactly describes motion of a **planar** patch or **pure rotation**.
  Very high inlier ratios under :math:`H` suggest **flat 2D content** (billboard / screen) or dominant
  plane — another common false-positive pattern.

A **true airborne** target (parallax, off the dominant plane) should leave a subset of features inside
its box that are **not** explained as well by the global :math:`F` / :math:`H` models as the bulk
of the background.

Summary: **no CUDA → CPU ORB path** (lines ``_extract_and_match_cpu``); **CUDA available → try GPU ORB
first**, fall back to CPU on ``cv2.error``.

**Debugging during video infer:** set env ``AEROSENTRY_GEO_DEBUG=1`` or pass ``verbose=True`` to
:class:`GeometricEgoMotion`, or use ``python3 run.py infer --fp-suppressor --geo-debug …`` to print
per-step logs (ORB path, match count, RANSAC inliers, per-track keep/drop).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


def _env_geo_debug() -> bool:
    v = os.environ.get("AEROSENTRY_GEO_DEBUG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class GeometricEgoMotion:
    """Extract ORB matches once per frame pair (CUDA optional), fit :math:`F` and :math:`H`, gate tracks by ROI inliers."""

    def __init__(
        self,
        max_features: int = 2000,
        lowe_ratio: float = 0.75,
        ransac_threshold_F: float = 1.0,
        ransac_threshold_H: float = 3.0,
        fp_inlier_ratio_F: float = 0.65,
        fp_inlier_ratio_H: float = 0.85,
        min_pts_in_bbox: int = 5,
        fast_threshold: int = 15,              # Lower ORB threshold to find more features on smooth UAVs
        roi_margin_px: float = 3.0,            # Expand ROI slightly to capture edge features of targets
        skip_geo_area_threshold: float = 0.001, # Skip geometry check for very small targets (<0.1% of frame area)
        *,
        verbose: bool = False,
    ) -> None:
        self.max_features = int(max_features)
        self.lowe_ratio = float(lowe_ratio)
        self.ransac_threshold_F = float(ransac_threshold_F)
        self.ransac_threshold_H = float(ransac_threshold_H)
        self.fp_inlier_ratio_F = float(fp_inlier_ratio_F)
        self.fp_inlier_ratio_H = float(fp_inlier_ratio_H)
        self.min_pts_in_bbox = int(min_pts_in_bbox)
        
        self.fast_threshold = int(fast_threshold)
        self.roi_margin_px = float(roi_margin_px)
        self.skip_geo_area_threshold = float(skip_geo_area_threshold)
        
        self._verbose = bool(verbose) or _env_geo_debug()

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[GeometricEgoMotion] {msg}", flush=True)

    @staticmethod
    def _has_cuda() -> bool:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

    def _extract_and_match_cuda(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
        *,
        frame_id: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """ORB + ratio test on GPU when available; otherwise CPU ORB."""
        fid = "" if frame_id is None else f" frame_id={frame_id}"
        if prev_bgr is None or curr_bgr is None:
            self._log(f"extract_match:{fid} skip (None frame)")
            return None, None
        if prev_bgr.size == 0 or curr_bgr.size == 0:
            self._log(f"extract_match:{fid} skip (empty array)")
            return None, None
        h0, w0 = prev_bgr.shape[:2]
        h1, w1 = curr_bgr.shape[:2]
        if h0 != h1 or w0 != w1:
            self._log(
                f"extract_match:{fid} resize curr {w1}x{h1} -> {w0}x{h0} to match prev"
            )
            curr_bgr = cv2.resize(curr_bgr, (w0, h0), interpolation=cv2.INTER_LINEAR)

        if self._has_cuda():
            try:
                out = self._extract_and_match_cuda_impl(prev_bgr, curr_bgr)
                n = len(out[0]) if out[0] is not None else 0
                self._log(f"extract_match:{fid} path=CUDA good_matches={n}")
                return out
            except cv2.error as e:
                self._log(f"extract_match:{fid} CUDA failed ({e}); fallback CPU")
        else:
            self._log(f"extract_match:{fid} path=CPU (no cv2.cuda)")

        out = self._extract_and_match_cpu(prev_bgr, curr_bgr)
        n = len(out[0]) if out[0] is not None else 0
        self._log(f"extract_match:{fid} path=CPU good_matches={n}")
        return out

    def _extract_and_match_cuda_impl(
        self, prev_bgr: np.ndarray, curr_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """GPU path: upload BGR, grayscale on GPU, ORB + Hamming BF knn + Lowe ratio."""
        g0 = cv2.cuda_GpuMat()
        g1 = cv2.cuda_GpuMat()
        g0.upload(prev_bgr)
        g1.upload(curr_bgr)
        gray0 = cv2.cuda.cvtColor(g0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cuda.cvtColor(g1, cv2.COLOR_BGR2GRAY)

        orb = cv2.cuda_ORB.create(nfeatures=self.max_features, fastThreshold=self.fast_threshold)
        kp0, desc0 = orb.detectAndCompute(gray0, None)
        kp1, desc1 = orb.detectAndCompute(gray1, None)
        if desc0 is None or desc1 is None or len(kp0) < 2 or len(kp1) < 2:
            self._log("extract_match impl=CUDA no descriptors / too few keypoints -> None")
            return None, None

        matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        raw = matcher.knnMatch(desc0, desc1, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)
        if len(good) < 8:
            self._log(f"extract_match impl=CUDA good_matches={len(good)} (<8, abort)")
            return None, None

        pts_prev = np.float32([kp0[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp1[m.trainIdx].pt for m in good])
        return pts_prev, pts_curr

    def _extract_and_match_cpu(
        self, prev_bgr: np.ndarray, curr_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gray0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=self.max_features, fastThreshold=self.fast_threshold)
        kp0, d0 = orb.detectAndCompute(gray0, None)
        kp1, d1 = orb.detectAndCompute(gray1, None)
        if d0 is None or d1 is None or len(kp0) < 2 or len(kp1) < 2:
            self._log("extract_match impl=CPU no descriptors / too few keypoints -> None")
            return None, None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = bf.knnMatch(d0, d1, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)
        if len(good) < 8:
            self._log(f"extract_match impl=CPU good_matches={len(good)} (<8, abort)")
            return None, None
        pts_prev = np.float32([kp0[m.queryIdx].pt for m in good])
        pts_curr = np.float32([kp1[m.trainIdx].pt for m in good])
        return pts_prev, pts_curr

    def _compute_global_models(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        *,
        frame_id: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """RANSAC Fundamental + Homography; return boolean inlier masks aligned with ``pts_curr`` rows."""
        fid = "" if frame_id is None else f" frame_id={frame_id}"
        if pts_prev.shape[0] < 8 or pts_curr.shape[0] < 8:
            self._log(f"global_models:{fid} skip (need 8+ pts, got {pts_prev.shape[0]})")
            return None, None
        if pts_prev.shape != pts_curr.shape:
            self._log(f"global_models:{fid} skip (shape mismatch)")
            return None, None
        pts_p = np.ascontiguousarray(pts_prev, dtype=np.float32).reshape(-1, 2)
        pts_c = np.ascontiguousarray(pts_curr, dtype=np.float32).reshape(-1, 2)

        F, mask_F = cv2.findFundamentalMat(
            pts_p,
            pts_c,
            cv2.FM_RANSAC,
            self.ransac_threshold_F,
            0.99,
        )
        if mask_F is None:
            mask_F = np.zeros((len(pts_p), 1), dtype=np.uint8)
        mask_F_bool = mask_F.ravel().astype(bool)
        if F is None or mask_F_bool.size != len(pts_p):
            mask_F_bool = np.zeros(len(pts_p), dtype=bool)

        H, mask_H = cv2.findHomography(pts_p, pts_c, cv2.RANSAC, self.ransac_threshold_H)
        if H is None or mask_H is None:
            mask_H_bool = np.zeros(len(pts_p), dtype=bool)
        else:
            m = mask_H.ravel()
            mask_H_bool = m.astype(bool) if m.size == len(pts_p) else np.zeros(len(pts_p), dtype=bool)

        n_f = int(np.sum(mask_F_bool))
        n_h = int(np.sum(mask_H_bool))
        self._log(
            f"global_models:{fid} pts={len(pts_p)} F_inliers={n_f} H_inliers={n_h} "
            f"(RANSAC F thresh={self.ransac_threshold_F}px H thresh={self.ransac_threshold_H}px)"
        )
        return mask_F_bool, mask_H_bool

    def analyze_bbox_motion(
        self,
        bbox_normalized: Tuple[float, float, float, float],
        frame_shape: Tuple[int, ...],
        pts_curr: np.ndarray,
        inlier_mask_F: Optional[np.ndarray],
        inlier_mask_H: Optional[np.ndarray],
        *,
        debug_tag: str = "",
    ) -> bool:
        """Return ``True`` if the track is treated as **airborne** (keep); ``False`` to drop as FP.

        Args:
            bbox_normalized: ``(cx, cy, w, h)`` in normalized YOLO coordinates ``[0,1]``.
            frame_shape: ``(h, w, ...)`` from ``frame.shape``.
            pts_curr: Matched keypoint locations in the **current** frame, shape ``(N, 2)``.
            inlier_mask_F: RANSAC inliers for :math:`F`, length ``N`` (or ``None`` if skipped).
            inlier_mask_H: RANSAC inliers for :math:`H`, length ``N``.
            debug_tag: Optional short string for logs (e.g. ``frame_id=… track_id=…``).

        Returns:
            ``True`` → keep detection; ``False`` → grounded / planar FP.
        """
        tag = f" [{debug_tag}]" if debug_tag else ""

        if pts_curr is None or len(pts_curr) == 0:
            self._log(f"analyze_bbox:{tag} no keypoints -> keep (airborne)")
            return True
        if inlier_mask_F is None or inlier_mask_H is None:
            self._log(f"analyze_bbox:{tag} no F/H masks -> keep (airborne)")
            return True
        if len(inlier_mask_F) != len(pts_curr) or len(inlier_mask_H) != len(pts_curr):
            self._log(f"analyze_bbox:{tag} mask length mismatch -> keep (airborne)")
            return True

        h, w = int(frame_shape[0]), int(frame_shape[1])
        cx, cy, bw, bh = (float(bbox_normalized[i]) for i in range(4))
        
        area_norm = bw * bh
        if area_norm < self.skip_geo_area_threshold:
            self._log(
                f"analyze_bbox:{tag} KEEP airborne (box area {area_norm:.5f} < {self.skip_geo_area_threshold}) - too small for geometry"
            )
            return True

        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h

        x1 = x1 - self.roi_margin_px
        y1 = y1 - self.roi_margin_px
        x2 = x2 + self.roi_margin_px
        y2 = y2 + self.roi_margin_px

        px, py = pts_curr[:, 0], pts_curr[:, 1]
        inside = (px > x1) & (px < x2) & (py > y1) & (py < y2)
        idx = np.nonzero(inside)[0]
        total = int(idx.size)
        if total < self.min_pts_in_bbox:
            self._log(
                f"analyze_bbox:{tag} pts_inside_roi={total} < min_pts={self.min_pts_in_bbox} -> keep"
            )
            return True

        in_f = int(np.sum(inlier_mask_F[idx]))
        in_h = int(np.sum(inlier_mask_H[idx]))
        ratio_f = in_f / float(total)
        ratio_h = in_h / float(total)

        if ratio_f > self.fp_inlier_ratio_F:
            self._log(
                f"analyze_bbox:{tag} DROP grounded-FP ratio_F={ratio_f:.3f} "
                f"> {self.fp_inlier_ratio_F} (inliers {in_f}/{total})"
            )
            return False
        if ratio_h > self.fp_inlier_ratio_H:
            self._log(
                f"analyze_bbox:{tag} DROP planar-FP ratio_H={ratio_h:.3f} "
                f"> {self.fp_inlier_ratio_H} (inliers {in_h}/{total})"
            )
            return False
        self._log(
            f"analyze_bbox:{tag} KEEP airborne ratio_F={ratio_f:.3f} ratio_H={ratio_h:.3f} "
            f"(inliers F {in_f}/{total} H {in_h}/{total})"
        )
        return True


__all__ = ["GeometricEgoMotion"]
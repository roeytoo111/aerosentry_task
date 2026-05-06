"""Pipe-and-filter style orchestration for video ingest, detect, and post-process."""

from __future__ import annotations

import time
from typing import Optional

import cv2

from src.core.data_contracts import FrameData
from src.models.base_detector import BaseDetector
from src.models.detector_factory import DetectorFactory
from src.tracking.fp_suppressor import FalsePositiveSuppressor


class PipelineManager:
    """Coordinates frame ingestion, detection, and downstream filters."""

    def __init__(
        self,
        detector: Optional[BaseDetector] = None,
        *,
        model_name: str = "yolov11",
        suppressor: Optional[FalsePositiveSuppressor] = None,
    ) -> None:
        """Initialize the pipeline with an explicit detector or a factory key.

        Args:
            detector: Pre-built detector. If ``None``, ``model_name`` is passed to
                :class:`~src.models.detector_factory.DetectorFactory`.
            model_name: Keyword-only; used only when ``detector`` is ``None``.
            suppressor: Temporal + geometric FP gatekeeper; default :class:`FalsePositiveSuppressor`.
        """
        if detector is None:
            self._detector = DetectorFactory.create(model_name)
        else:
            self._detector = detector
        self._suppressor = suppressor if suppressor is not None else FalsePositiveSuppressor()

    def process_video(self, video_path: str) -> None:
        """Read a video file, run detection per frame, and apply post-processing stubs.

        This implements a minimal pipe: Ingest → Detect → ``_suppress_false_positives``.

        Args:
            video_path: Path readable by OpenCV (file or device index as string on some builds).

        Note:
            Frames are consumed sequentially; extend this method or add iterators
            for production telemetry and interceptor handoff.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                ts = time.perf_counter()
                frame_data = FrameData(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=ts,
                    detections=[],
                )
                predictions = self._detector.predict(frame)
                frame_data.detections.extend(predictions)
                filtered = self._suppressor.process(frame_data)
                frame_data.detections = filtered.detections
                frame_id += 1
        finally:
            cap.release()

    def reset_suppressor(self) -> None:
        """Clear track + ego history (e.g. before a new video)."""
        self._suppressor.reset()


__all__ = ["PipelineManager"]

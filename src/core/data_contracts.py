"""Immutable data contracts for frames and detections flowing through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

BBoxNormalized = Union[Tuple[float, float, float, float], List[float]]


@dataclass
class Detection:
    """A single object detection in normalized image coordinates.

    Bounding box uses YOLO-style normalized values:
    ``[x_center, y_center, width, height]`` in relative [0, 1] space.

    Attributes:
        class_id: Integer class identifier from the model label map.
        class_label: Human-readable class name (e.g. ``"fixed_wing_uav"``).
        confidence: Detector confidence score in ``[0.0, 1.0]``.
        bbox: Normalized box ``(x_center, y_center, width, height)``.
    """

    class_id: int
    class_label: str
    confidence: float
    bbox: BBoxNormalized
    track_id: Optional[int] = None

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:  # type: ignore[arg-type]
            raise ValueError("bbox must have exactly four elements: x, y, w, h")


@dataclass
class FrameData:
    """One video frame plus metadata and optional detections.

    Attributes:
        frame: BGR or RGB image as a NumPy array (layout defined by ingest).
        frame_id: Monotonic index within the current processing session.
        timestamp: Wall-clock or media time in seconds (source-defined).
        detections: Detections produced for this frame; empty until inference runs.
    """

    frame: np.ndarray
    frame_id: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)


__all__ = ["BBoxNormalized", "Detection", "FrameData"]

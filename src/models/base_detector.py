"""Abstract detector interface and concrete stub implementations for bring-up."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.core.data_contracts import Detection


class BaseDetector(ABC):
    """Strategy interface for swapping detection backends (PyTorch, TensorRT, etc.).

    Downstream pipeline code depends only on this contract, not on specific runtimes.
    """

    @abstractmethod
    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single frame.

        Args:
            frame: Image as ``H x W x C`` ``uint8`` or ``float`` array.

        Returns:
            Zero or more :class:`Detection` instances in normalized coordinates.
        """
        raise NotImplementedError


class YOLOv11Detector(BaseDetector):
    """Stub YOLOv11 path returning fixed mock detections for pipeline tests.

    Replace this with a real Ultralytics / TensorRT integration when weights are ready.
    """

    def predict(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return []
        # One synthetic box near image center (normalized).
        cx, cy = 0.5, 0.5
        bw, bh = 0.1, 0.1
        return [
            Detection(
                class_id=0,
                class_label="fixed_wing_uav",
                confidence=0.99,
                bbox=[cx, cy, bw, bh],
            )
        ]


class TensorRTDetector(BaseDetector):
    """Placeholder TensorRT engine wrapper; returns mock output until engines exist."""

    def predict(self, frame: np.ndarray) -> List[Detection]:
        if frame.size == 0:
            return []
        return [
            Detection(
                class_id=0,
                class_label="fixed_wing_uav_trt_stub",
                confidence=0.95,
                bbox=[0.52, 0.48, 0.08, 0.12],
            )
        ]


__all__ = ["BaseDetector", "TensorRTDetector", "YOLOv11Detector"]

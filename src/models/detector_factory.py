"""Factory for instantiating detector implementations by name."""

from __future__ import annotations

from typing import Dict, Type

from src.models.base_detector import BaseDetector, TensorRTDetector, YOLOv11Detector


class DetectorFactory:
    """Maps logical model names to concrete :class:`BaseDetector` classes.

    Adding a new backbone means registering it here; call sites stay unchanged.
    """

    _registry: Dict[str, Type[BaseDetector]] = {
        "yolov11": YOLOv11Detector,
        "tensorrt": TensorRTDetector,
    }

    @staticmethod
    def create(model_name: str) -> BaseDetector:
        """Build a detector instance for the given logical name.

        Args:
            model_name: Key such as ``"yolov11"`` or ``"tensorrt"`` (case-insensitive).

        Returns:
            A new, ready-to-use detector instance.

        Raises:
            ValueError: If ``model_name`` is not registered.
        """
        key = model_name.strip().lower()
        cls = DetectorFactory._registry.get(key)
        if cls is None:
            available = ", ".join(sorted(DetectorFactory._registry))
            raise ValueError(
                f"Unknown detector '{model_name}'. Registered backends: {available}"
            )
        return cls()

    @classmethod
    def register(cls, name: str, detector_cls: Type[BaseDetector]) -> None:
        """Register an additional detector type (e.g. from a plugin).

        Args:
            name: Case-insensitive key used with :meth:`create`.
            detector_cls: Subclass of :class:`BaseDetector`.
        """
        cls._registry[name.strip().lower()] = detector_cls


__all__ = ["DetectorFactory"]

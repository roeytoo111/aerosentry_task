"""One Euro Filter for jitter-stable bounding box smoothing (CPU-only)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _smoothing_factor(time_delta: float, cutoff: float) -> float:
    """Interpolation `alpha` for an exponential low-pass with time constant from cutoff."""
    r = 2.0 * math.pi * cutoff * time_delta
    return r / (r + 1.0)


@dataclass
class _LowPass:
    """Exponential moving average with per-step ``alpha``."""

    _x: Optional[float] = None

    def filter(self, x: float, alpha: float) -> float:
        if self._x is None:
            self._x = x
        else:
            self._x = alpha * x + (1.0 - alpha) * self._x
        return float(self._x)

    @property
    def value(self) -> Optional[float]:
        return self._x


class OneEuroFilter:
    """1€ filter for a scalar signal (Géry et al.) — low lag on transients, strong smoothing at rest.

    Suitable for normalizing detector jitter on box coordinates while following agile targets.

    Attributes:
        min_cutoff: Minimum cutoff frequency (Hz) when motion is negligible.
        beta: Speed coefficient; larger values reduce smoothing when velocity is high.
        d_cutoff: Cutoff for the derivative branch (reduces derivative noise).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_pass = _LowPass()
        self._dx_pass = _LowPass()
        self._t_prev: Optional[float] = None

    def reset(self) -> None:
        """Forget state (e.g. after track ID recycle)."""
        self._x_pass = _LowPass()
        self._dx_pass = _LowPass()
        self._t_prev = None

    def filter(self, x: float, timestamp: float) -> float:
        """Filter scalar ``x`` at monotonic ``timestamp`` (seconds).

        Args:
            x: Noisy measurement.
            timestamp: Time in seconds (must be non-decreasing across calls).

        Returns:
            Smoothed value.
        """
        if self._t_prev is None:
            self._t_prev = timestamp
            return self._x_pass.filter(x, 1.0)

        dt = max(timestamp - self._t_prev, 1e-6)
        self._t_prev = timestamp

        x_prev = self._x_pass.value
        if x_prev is None:
            x_prev = x
        dx = (x - x_prev) / dt
        alpha_d = _smoothing_factor(dt, self.d_cutoff)
        dx_hat = self._dx_pass.filter(dx, alpha_d)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = _smoothing_factor(dt, cutoff)
        return self._x_pass.filter(x, alpha)

    def __call__(self, x: float, timestamp: float) -> float:
        return self.filter(x, timestamp)


class BoundingBoxOneEuroFilter:
    """Independent 1€ filters for ``(x_center, y_center, width, height)`` in normalized space."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fw = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fh = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self._fw.reset()
        self._fh.reset()

    def filter(
        self,
        bbox: Tuple[float, float, float, float] | np.ndarray,
        timestamp: float,
    ) -> Tuple[float, float, float, float]:
        """Smooth a YOLO-style box ``(cx, cy, w, h)`` at ``timestamp``.

        Args:
            bbox: Normalized center box.
            timestamp: Seconds (monotonic per track).

        Returns:
            Smoothed ``(cx, cy, w, h)``.
        """
        arr = np.asarray(bbox, dtype=np.float64).reshape(4)
        cx = float(self._fx(arr[0], timestamp))
        cy = float(self._fy(arr[1], timestamp))
        w = float(max(self._fw(arr[2], timestamp), 1e-6))
        h = float(max(self._fh(arr[3], timestamp), 1e-6))
        return cx, cy, w, h


__all__ = ["BoundingBoxOneEuroFilter", "OneEuroFilter"]

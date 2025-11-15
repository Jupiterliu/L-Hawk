"""Real-time laser attack detection and mitigation utilities.

This module provides a lightweight, real-time friendly pipeline that can be
plugged into an existing vision system to detect and suppress laser raster
attacks.  The design follows two stages:

1. **Detection** – A sliding-window tracker monitors statistics of incoming
   frames (intensity, saturation, highlight coverage, etc.).  Robust
   thresholds are derived on-the-fly from recent history using fast median and
   median absolute deviation (MAD) estimators.  When the current frame exceeds
   any of the adaptive thresholds, it is flagged as a laser-corrupted frame.
2. **Defense / Recovery** – Saturated laser pixels are segmented using
   brightness, saturation, and color channel consistency cues.  The resulting
   mask is filled using OpenCV's fast inpainting routine, which preserves
   surrounding context so downstream detection and recognition performance is
   not degraded.

Example
-------
>>> detector = LaserDefenseSystem()
>>> while True:
...     frame = next_frame_from_camera()
...     clean, info = detector.process(frame)
...     if info.is_attack:
...         handle_security_alert()
...     consume_frame(clean)

The implementation purposefully avoids any heavy-weight dependencies so it can
be embedded on edge devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
from collections import deque


def _to_float_frame(frame: np.ndarray) -> np.ndarray:
    """Convert *frame* to ``float32`` in the ``[0, 1]`` range."""

    if frame.dtype == np.uint8:
        return frame.astype(np.float32) / 255.0
    if frame.dtype in (np.float32, np.float64):
        frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame /= 255.0
        return frame
    raise TypeError(f"Unsupported frame dtype: {frame.dtype}")


def _mad(values: np.ndarray) -> float:
    """Compute the Median Absolute Deviation (MAD)."""

    if values.size == 0:
        return 0.0
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


@dataclass
class DetectionConfig:
    """Configuration for the sliding-window detector."""

    window_size: int = 30
    warmup: int = 15
    intensity_k: float = 8.0
    saturation_k: float = 6.0
    highlight_ratio_k: float = 6.0
    channel_spread_k: float = 4.5
    highlight_value: float = 0.82


@dataclass
class DefenseConfig:
    """Configuration for the restoration module."""

    highlight_value: float = 0.82
    saturation_value: float = 0.35
    channel_agreement: float = 0.12
    dilation: int = 3
    inpaint_radius: int = 3


@dataclass
class DetectionResult:
    """Result returned by :class:`LaserDefenseSystem.process`."""

    is_attack: bool
    thresholds: Dict[str, Tuple[float, float]]
    features: Dict[str, float]
    mask: Optional[np.ndarray]


class LaserAttackDetector:
    """Sliding-window, threshold-based detector.

    Each call to :meth:`update` ingests the newest frame statistics and
    evaluates whether they breach the adaptive thresholds derived from the
    trailing window.  The thresholds are recalculated from scratch on every
    iteration; given the small window sizes (tens of frames) this remains
    computationally negligible.
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self._intensity: Deque[float] = deque(maxlen=config.window_size)
        self._saturation: Deque[float] = deque(maxlen=config.window_size)
        self._highlight_ratio: Deque[float] = deque(maxlen=config.window_size)
        self._channel_spread: Deque[float] = deque(maxlen=config.window_size)

    def _compute_threshold(self, series: Deque[float], k: float) -> Tuple[float, float]:
        arr = np.asarray(series, dtype=np.float32)
        median = float(np.median(arr))
        mad = _mad(arr)
        upper = median + k * (mad + 1e-6)
        lower = median - k * (mad + 1e-6)
        return lower, upper

    def update(self, frame: np.ndarray) -> Tuple[bool, Dict[str, Tuple[float, float]], Dict[str, float]]:
        float_frame = _to_float_frame(frame)
        hsv = cv2.cvtColor((float_frame * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32) / 255.0

        intensity = float(np.mean(float_frame))
        saturation = float(np.mean(hsv[..., 1]))
        highlight = float(np.mean(np.max(float_frame, axis=2) > self.config.highlight_value))
        channel_spread = float(np.mean(np.max(float_frame, axis=2) - np.min(float_frame, axis=2)))

        self._intensity.append(intensity)
        self._saturation.append(saturation)
        self._highlight_ratio.append(highlight)
        self._channel_spread.append(channel_spread)

        features = {
            "intensity": intensity,
            "saturation": saturation,
            "highlight_ratio": highlight,
            "channel_spread": channel_spread,
        }

        ready = len(self._intensity) >= self.config.warmup
        thresholds: Dict[str, Tuple[float, float]] = {}
        is_attack = False

        if ready:
            thresholds["intensity"] = self._compute_threshold(self._intensity, self.config.intensity_k)
            thresholds["saturation"] = self._compute_threshold(self._saturation, self.config.saturation_k)
            thresholds["highlight_ratio"] = self._compute_threshold(
                self._highlight_ratio, self.config.highlight_ratio_k
            )
            thresholds["channel_spread"] = self._compute_threshold(
                self._channel_spread, self.config.channel_spread_k
            )

            for key, (lower, upper) in thresholds.items():
                value = features[key]
                if value < lower or value > upper:
                    is_attack = True
                    break

        return is_attack, thresholds, features


class LaserArtifactRemover:
    """Fast artifact suppressor based on morphological inpainting."""

    def __init__(self, config: DefenseConfig):
        self.config = config

    def _build_mask(self, frame: np.ndarray) -> np.ndarray:
        float_frame = _to_float_frame(frame)
        hsv = cv2.cvtColor((float_frame * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32) / 255.0

        gray = cv2.cvtColor((float_frame * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0

        bright = gray > self.config.highlight_value
        saturated = hsv[..., 1] > self.config.saturation_value
        channel_diff = (np.max(float_frame, axis=2) - np.min(float_frame, axis=2)) < self.config.channel_agreement

        mask = bright & saturated & channel_diff

        if self.config.dilation > 0:
            kernel = np.ones((self.config.dilation, self.config.dilation), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        return mask.astype(np.uint8)

    def restore(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = self._build_mask(frame)
        if not np.any(mask):
            return frame, mask

        inpaint_mask = (mask * 255).astype(np.uint8)
        restored = cv2.inpaint(frame, inpaint_mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)
        return restored, mask


class LaserDefenseSystem:
    """High-level orchestrator that combines detection and restoration."""

    def __init__(
        self,
        detection_config: Optional[DetectionConfig] = None,
        defense_config: Optional[DefenseConfig] = None,
    ):
        self.detector = LaserAttackDetector(detection_config or DetectionConfig())
        self.remover = LaserArtifactRemover(defense_config or DefenseConfig())

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, DetectionResult]:
        """Detect and mitigate laser attacks on *frame*.

        Returns a tuple of the possibly restored frame and a
        :class:`DetectionResult` object containing debugging information.
        """

        is_attack, thresholds, features = self.detector.update(frame)
        mask = None
        output = frame
        if is_attack:
            output, mask = self.remover.restore(frame)

        result = DetectionResult(
            is_attack=is_attack,
            thresholds=thresholds,
            features=features,
            mask=mask,
        )
        return output, result


__all__ = [
    "LaserDefenseSystem",
    "LaserAttackDetector",
    "LaserArtifactRemover",
    "DetectionConfig",
    "DefenseConfig",
    "DetectionResult",
]


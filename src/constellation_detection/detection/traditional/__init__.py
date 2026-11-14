"""Traditional computer-vision star detection algorithms."""

from __future__ import annotations

from constellation_detection.detection.traditional.threshold import (
    DETECTOR_NAME,
    detect_stars_global_threshold,
)

__all__ = [
    "DETECTOR_NAME",
    "detect_stars_global_threshold",
]

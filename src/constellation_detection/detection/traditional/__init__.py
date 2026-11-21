"""Traditional computer-vision star detection algorithms."""

from __future__ import annotations

from constellation_detection.detection.traditional.dog import (
    detect_stars_multiscale_dog,
)
from constellation_detection.detection.traditional.threshold import (
    DETECTOR_NAME,
    detect_stars_global_threshold,
)
from constellation_detection.detection.traditional.tophat import (
    detect_stars_adaptive_tophat,
)

__all__ = [
    "DETECTOR_NAME",
    "detect_stars_adaptive_tophat",
    "detect_stars_global_threshold",
    "detect_stars_multiscale_dog",
]

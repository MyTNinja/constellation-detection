"""PyTorch deep learning star detection models, registry, and CPU/GPU inference."""

from __future__ import annotations

from constellation_detection.detection.dl.inference import (
    detect_stars_dl,
    extract_peaks_from_heatmap,
)
from constellation_detection.detection.dl.registry import (
    MODEL_REGISTRY,
    build_model,
)

__all__ = [
    "MODEL_REGISTRY",
    "build_model",
    "detect_stars_dl",
    "extract_peaks_from_heatmap",
]

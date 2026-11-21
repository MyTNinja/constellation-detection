"""Adaptive Top-Hat traditional star detector."""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from constellation_detection.detection.nms import fast_nms_stars
from constellation_detection.image.preprocessing import (
    apply_gaussian_blur,
    to_grayscale,
)

__all__ = [
    "DETECTOR_NAME",
    "detect_stars_adaptive_tophat",
]

DETECTOR_NAME = "CV2_AdaptiveTopHat"

_TOPHAT_KERNEL_SIZE: tuple[int, int] = (19, 19)
_BLUR_KERNEL_SIZE: tuple[int, int] = (3, 3)
_BLUR_SIGMA_X: float = 0.8
_ADAPTIVE_BLOCK_SIZE: int = 31
_ADAPTIVE_C: float = -18.0
_NOISE_STD_MULTIPLIER: float = 3.2
_GLOBAL_MIN_THRESHOLD: float = 28.0
_OPEN_KERNEL_SIZE: tuple[int, int] = (2, 2)
_MIN_COMPONENT_AREA: int = 2
_MAX_COMPONENT_AREA: int = 180
_MAX_SCORE_AREA_CAP: float = 20.0


def _validate_parameters(
    image: np.ndarray,
    min_dist_px: float,
    max_candidates: int,
) -> None:
    """Validate detector input array dtype and NMS parameters."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected image to be a numpy.ndarray, got {type(image).__name__}."
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image array for adaptive top-hat detection, got {image.dtype}."
        )

    if isinstance(min_dist_px, (bool, np.bool_)) or not isinstance(
        min_dist_px, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"min_dist_px must be a real number, got {type(min_dist_px).__name__}."
        )

    if not np.isfinite(float(min_dist_px)) or float(min_dist_px) < 0.0:
        raise ValueError(
            f"min_dist_px must be finite and non-negative, got {min_dist_px}."
        )

    if isinstance(max_candidates, (bool, np.bool_)) or not isinstance(
        max_candidates, (int, np.integer)
    ):
        raise TypeError(
            f"max_candidates must be an integer, got {type(max_candidates).__name__}."
        )

    if int(max_candidates) <= 0:
        raise ValueError(
            f"max_candidates must be a positive integer, got {max_candidates}."
        )


def detect_stars_adaptive_tophat(
    image: np.ndarray,
    min_dist_px: float = 10.0,
    max_candidates: int = 250,
) -> dict[str, Any]:
    """Detect star candidates using the Phase 1 Adaptive Top-Hat algorithm.

    Algorithm steps:
        1. Convert BGR input to grayscale (or accept existing 2D grayscale).
        2. Create an elliptical structuring element of size 19 x 19.
        3. Apply morphological white Top-Hat (``cv2.MORPH_TOPHAT``).
        4. Apply Gaussian blur with kernel ``(3, 3)`` and ``sigma_x=0.8``.
        5. Normalize response to ``[0, 255]`` using ``cv2.NORM_MINMAX``.
        6. Apply Gaussian adaptive thresholding with block size ``31`` and ``C=-18.0``.
        7. Compute noise floor as ``median(norm_th) + 3.2 * std(norm_th)``.
        8. Apply global binary threshold using ``max(28.0, noise_floor)``.
        9. Combine adaptive and global masks via bitwise AND.
        10. Apply morphological opening with a 2 x 2 ``uint8`` ones kernel.
        11. Extract connected components using 8-connectivity.
        12. Accept component areas from ``2`` through ``180`` pixels, inclusive.
        13. Compute flux-weighted centroid from ``norm_th`` within each component.
        14. Score each component as ``peak_val * sqrt(min(area, 20.0))``.
        15. Filter candidates using ``fast_nms_stars``.

    Args:
        image: Input 2D grayscale ``(H, W)`` or 3D BGR ``(H, W, C)`` ``uint8`` image.
        min_dist_px: Minimum suppression radius in pixels passed to ``fast_nms_stars``.
            Defaults to ``10.0``.
        max_candidates: Maximum number of star detections retained by NMS.
            Defaults to ``250``.

    Returns:
        Dictionary containing:
            - ``"detector"`` (``str``): ``"CV2_AdaptiveTopHat"``.
            - ``"stars_xyf"`` (``np.ndarray``): Array of shape ``(N, 3)`` and dtype
              ``float32`` with columns ``[x, y, score]`` sorted brightest-first.
            - ``"preprocessed"`` (``np.ndarray``): 2D ``uint8`` normalized Top-Hat image.
            - ``"mask"`` (``np.ndarray``): 2D ``uint8`` binary mask after opening.
            - ``"latency_ms"`` (``float``): Execution time in milliseconds.

    Raises:
        TypeError: If ``image``, ``min_dist_px``, or ``max_candidates`` have invalid types.
        ValueError: If ``image`` is empty, has invalid dimensions or dtype, or if
            ``min_dist_px`` / ``max_candidates`` are out of valid bounds.
    """
    _validate_parameters(image, min_dist_px=min_dist_px, max_candidates=max_candidates)

    start_time = time.perf_counter()

    # 1. Convert BGR to grayscale
    gray = to_grayscale(image)

    # 2. Elliptical structuring element (19 x 19) & 3. White Top-Hat
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _TOPHAT_KERNEL_SIZE)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)

    # 4. Gaussian blur (3, 3), sigma 0.8
    blurred = apply_gaussian_blur(
        tophat,
        kernel_size=_BLUR_KERNEL_SIZE,
        sigma_x=_BLUR_SIGMA_X,
    )

    # 5. Normalize to [0, 255] using cv2.NORM_MINMAX
    norm_th = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 6. Gaussian adaptive thresholding (block size 31, C = -18.0)
    adapt_mask = cv2.adaptiveThreshold(
        norm_th,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        _ADAPTIVE_BLOCK_SIZE,
        _ADAPTIVE_C,
    )

    # 7. Noise floor = median(norm_th) + 3.2 * std(norm_th)
    noise_floor = float(np.median(norm_th) + _NOISE_STD_MULTIPLIER * np.std(norm_th))

    # 8. Global binary threshold using max(28.0, noise_floor)
    _, global_mask = cv2.threshold(
        norm_th,
        max(_GLOBAL_MIN_THRESHOLD, noise_floor),
        255,
        cv2.THRESH_BINARY,
    )

    # 9. Combine adaptive and global masks using bitwise AND
    combined_mask = cv2.bitwise_and(adapt_mask, global_mask)

    # 10. Morphological opening with 2 x 2 uint8 ones kernel
    open_kernel = np.ones(_OPEN_KERNEL_SIZE, dtype=np.uint8)
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, open_kernel)

    # 11. Extract connected components using connectivity=8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    candidates: list[list[float]] = []
    for label_idx in range(1, num_labels):
        # 12. Accept component areas from 2 through 180 pixels, inclusive
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < _MIN_COMPONENT_AREA or area > _MAX_COMPONENT_AREA:
            continue

        x0 = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        comp_mask = labels[y0 : y0 + h, x0 : x0 + w] == label_idx
        ys, xs = np.nonzero(comp_mask)
        flux = norm_th[y0 : y0 + h, x0 : x0 + w][comp_mask].astype(np.float64)

        # 13. Flux-weighted centroid from normalized image within the component
        total_flux = float(np.sum(flux))
        if total_flux > 0.0:
            cx = float(x0 + np.sum(xs * flux) / total_flux)
            cy = float(y0 + np.sum(ys * flux) / total_flux)
        else:
            cx = float(x0 + np.mean(xs))
            cy = float(y0 + np.mean(ys))

        peak_val = float(np.max(flux)) if flux.size > 0 else 0.0

        # 14. Score = peak_val * sqrt(min(area, 20.0))
        score = peak_val * math.sqrt(min(float(area), _MAX_SCORE_AREA_CAP))
        candidates.append([cx, cy, score])

    # 15. Apply existing fast_nms_stars
    stars_xyf = fast_nms_stars(
        candidates,
        min_dist_px=float(min_dist_px),
        max_candidates=int(max_candidates),
    )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "detector": DETECTOR_NAME,
        "stars_xyf": stars_xyf,
        "preprocessed": norm_th,
        "mask": mask,
        "latency_ms": float(latency_ms),
    }

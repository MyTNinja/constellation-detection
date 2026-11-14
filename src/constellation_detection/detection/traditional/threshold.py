"""Global Threshold traditional star detector."""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

from constellation_detection.detection.nms import fast_nms_stars
from constellation_detection.image.preprocessing import preprocess_image

__all__ = [
    "DETECTOR_NAME",
    "detect_stars_global_threshold",
]

DETECTOR_NAME = "global_threshold"

_BLUR_KERNEL_SIZE: tuple[int, int] = (5, 5)
_OPEN_KERNEL_SIZE: tuple[int, int] = (2, 2)
_THRESH_STD_MULTIPLIER: float = 2.5
_THRESH_MIN: float = 55.0
_THRESH_MAX: float = 180.0
_MAX_CONTOUR_AREA: float = 350.0
_MAX_SCORE_AREA_CAP: float = 25.0


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
            f"Expected uint8 image array for global threshold detection, got {image.dtype}."
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


def detect_stars_global_threshold(
    image: np.ndarray,
    min_dist_px: float = 10.0,
    max_candidates: int = 250,
) -> dict[str, Any]:
    """Detect star candidates using the Phase 1 Global Threshold baseline algorithm.

    Algorithm steps:
        1. Convert BGR input to grayscale (or accept existing 2D grayscale).
        2. Apply Gaussian blur using a 5x5 kernel.
        3. Compute global threshold: ``clip(mean + 2.5 * std, 55, 180)``.
        4. Apply binary thresholding (``cv2.THRESH_BINARY``).
        5. Apply morphological opening with a 2x2 rectangular kernel.
        6. Extract external contours (``cv2.RETR_EXTERNAL``, ``cv2.CHAIN_APPROX_SIMPLE``).
        7. Reject contours with area strictly greater than ``350``.
        8. Compute contour centroids via spatial moments with zero-moment fallback.
        9. Compute candidate brightness score as
           ``blurred[iy, ix] * sqrt(max(1, min(area, 25)))``.
        10. Filter candidates using ``fast_nms_stars`` with ``min_dist_px=10.0``.

    Args:
        image: Input 2D grayscale ``(H, W)`` or 3D BGR ``(H, W, C)`` ``uint8`` image.
        min_dist_px: Minimum suppression radius in pixels passed to ``fast_nms_stars``.
            Defaults to ``10.0``.
        max_candidates: Maximum number of star detections retained by NMS.
            Defaults to ``250``.

    Returns:
        Dictionary containing:
            - ``"detector"`` (``str``): Identifier string (``"global_threshold"``).
            - ``"stars_xyf"`` (``np.ndarray``): Array of shape ``(N, 3)`` and dtype
              ``float32`` with columns ``[x, y, score]`` sorted brightest-first.
            - ``"preprocessed"`` (``np.ndarray``): 2D ``uint8`` 5x5 Gaussian-blurred
              grayscale image of shape ``(H, W)``.
            - ``"mask"`` (``np.ndarray``): 2D ``uint8`` binary mask after morphological
              opening of shape ``(H, W)``.
            - ``"latency_ms"`` (``float``): Execution time in milliseconds.

    Raises:
        TypeError: If ``image``, ``min_dist_px``, or ``max_candidates`` have invalid types.
        ValueError: If ``image`` is empty, has invalid dimensions or dtype, or if
            ``min_dist_px`` / ``max_candidates`` are out of valid bounds.
    """
    _validate_parameters(image, min_dist_px=min_dist_px, max_candidates=max_candidates)

    start_time = time.perf_counter()

    # 1. Convert BGR to grayscale & 2. Apply 5x5 Gaussian blur
    blurred = preprocess_image(image, kernel_size=_BLUR_KERNEL_SIZE, sigma_x=0.0)

    # 3. Compute adaptive-global clipped threshold: clip(mean + 2.5 * std, 55, 180)
    mean_val = float(np.mean(blurred))
    std_val = float(np.std(blurred))
    thresh_val = float(
        np.clip(
            mean_val + _THRESH_STD_MULTIPLIER * std_val,
            _THRESH_MIN,
            _THRESH_MAX,
        )
    )

    # 4. Apply binary thresholding
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    # 5. Apply morphological opening with a 2x2 kernel
    open_kernel = np.ones(_OPEN_KERNEL_SIZE, dtype=np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    # 6. Find external contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = blurred.shape
    candidates: list[list[float]] = []

    for cnt in contours:
        # 7. Reject contours with area greater than 350
        area = float(cv2.contourArea(cnt))
        if area > _MAX_CONTOUR_AREA:
            continue

        # 8. Calculate centroids using contour moments with zero-moment fallback
        moments = cv2.moments(cnt)
        if moments["m00"] != 0.0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(np.mean(cnt[:, 0, 0]))
            cy = float(np.mean(cnt[:, 0, 1]))

        ix = int(np.clip(round(cx), 0, width - 1))
        iy = int(np.clip(round(cy), 0, height - 1))

        # 9. Calculate score: blurred[iy, ix] * sqrt(max(1, min(area, 25)))
        score = float(blurred[iy, ix]) * math.sqrt(
            max(1.0, min(area, _MAX_SCORE_AREA_CAP))
        )
        candidates.append([cx, cy, score])

    # 10. Apply fast_nms_stars with min_dist_px=10 (by default)
    stars_xyf = fast_nms_stars(
        candidates,
        min_dist_px=float(min_dist_px),
        max_candidates=int(max_candidates),
    )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "detector": DETECTOR_NAME,
        "stars_xyf": stars_xyf,
        "preprocessed": blurred,
        "mask": mask,
        "latency_ms": float(latency_ms),
    }

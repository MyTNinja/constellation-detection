"""Multi-Scale Difference of Gaussians (DoG) traditional star detector."""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np
from skimage.feature import blob_dog

from constellation_detection.detection.nms import fast_nms_stars
from constellation_detection.image.preprocessing import (
    apply_gaussian_blur,
    to_grayscale,
)

__all__ = [
    "DETECTOR_NAME",
    "detect_stars_multiscale_dog",
]

DETECTOR_NAME = "CV3_MultiScaleDoG"

_TOPHAT_KERNEL_SIZE: tuple[int, int] = (19, 19)
_BLUR_KERNEL_SIZE: tuple[int, int] = (3, 3)
_BLUR_SIGMA_X: float = 0.8
_DOG_MIN_SIGMA: float = 1.1
_DOG_MAX_SIGMA: float = 4.0
_DOG_SIGMA_RATIO: float = 1.5
_DOG_THRESHOLD: float = 0.16
_MASK_CIRCLE_RADIUS: int = 4


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
            f"Expected uint8 image array for multi-scale DoG detection, got {image.dtype}."
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


def detect_stars_multiscale_dog(
    image: np.ndarray,
    min_dist_px: float = 10.0,
    max_candidates: int = 250,
) -> dict[str, Any]:
    """Detect star candidates using the Phase 1 Multi-Scale DoG algorithm.

    Algorithm steps:
        1. Convert BGR input to grayscale (or accept existing 2D grayscale).
        2. Apply white Top-Hat with an elliptical 19 x 19 structuring element.
        3. Apply Gaussian blur with kernel ``(3, 3)`` and ``sigma_x=0.8``.
        4. Normalize to ``[0, 255]`` using ``cv2.NORM_MINMAX``.
        5. Call ``skimage.feature.blob_dog`` on normalized ``float32`` image divided by ``255``.
        6. Use ``min_sigma=1.1``, ``max_sigma=4.0``, ``sigma_ratio=1.5``, ``threshold=0.16``.
        7. For each blob ``(by, bx, sigma)``, compute radius
           ``int(max(2, round(sigma * sqrt(2))))``.
        8. Clip the local image patch to image boundaries.
        9. Compute flux-weighted centroid from the local patch.
        10. If total patch flux is zero, fall back to blob coordinates ``(bx, by)`` and
            ``peak_val = 1.0``.
        11. Compute score as ``peak_val * sigma``.
        12. Apply ``fast_nms_stars`` with the supplied radius and candidate limit.
        13. Render visualization mask by drawing radius-4 filled circles at final detections.

    Args:
        image: Input 2D grayscale ``(H, W)`` or 3D BGR ``(H, W, C)`` ``uint8`` image.
        min_dist_px: Minimum suppression radius in pixels passed to ``fast_nms_stars``.
            Defaults to ``10.0``.
        max_candidates: Maximum number of star detections retained by NMS.
            Defaults to ``250``.

    Returns:
        Dictionary containing:
            - ``"detector"`` (``str``): ``"CV3_MultiScaleDoG"``.
            - ``"stars_xyf"`` (``np.ndarray``): Array of shape ``(N, 3)`` and dtype
              ``float32`` with columns ``[x, y, score]`` sorted brightest-first.
            - ``"preprocessed"`` (``np.ndarray``): 2D ``uint8`` normalized Top-Hat image.
            - ``"mask"`` (``np.ndarray``): 2D ``uint8`` binary visualization mask with
              radius-4 circles around final detections.
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

    # 2. White Top-Hat with elliptical 19 x 19 structuring element
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _TOPHAT_KERNEL_SIZE)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)

    # 3. Gaussian blur with kernel (3, 3), sigma 0.8
    blurred = apply_gaussian_blur(
        tophat,
        kernel_size=_BLUR_KERNEL_SIZE,
        sigma_x=_BLUR_SIGMA_X,
    )

    # 4. Normalize to [0, 255]
    norm_th = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5 & 6. Multi-scale DoG blob detection on float32 image / 255.0
    norm_f32 = norm_th.astype(np.float32) / 255.0
    blobs = blob_dog(
        norm_f32,
        min_sigma=_DOG_MIN_SIGMA,
        max_sigma=_DOG_MAX_SIGMA,
        sigma_ratio=_DOG_SIGMA_RATIO,
        threshold=_DOG_THRESHOLD,
    )

    height, width = norm_th.shape
    candidates: list[list[float]] = []

    for blob in blobs:
        by = float(blob[0])
        bx = float(blob[1])
        sigma = float(blob[2])

        # 7. Radius = int(max(2, round(sigma * sqrt(2))))
        rad = max(2, round(sigma * math.sqrt(2.0)))

        # 8. Clip local image patch to image boundaries
        iy = round(by)
        ix = round(bx)
        y1 = max(0, iy - rad)
        y2 = min(height, iy + rad + 1)
        x1 = max(0, ix - rad)
        x2 = min(width, ix + rad + 1)

        patch = norm_th[y1:y2, x1:x2].astype(np.float64)
        total_flux = float(np.sum(patch))

        # 9 & 10. Flux-weighted centroid from patch; zero-flux fallback
        if total_flux > 0.0:
            yy, xx = np.mgrid[y1:y2, x1:x2]
            cx = float(np.sum(xx * patch) / total_flux)
            cy = float(np.sum(yy * patch) / total_flux)
            peak_val = float(np.max(patch))
        else:
            cx = bx
            cy = by
            peak_val = 1.0

        # 11. Score = peak_val * sigma
        score = peak_val * sigma
        candidates.append([cx, cy, score])

    # 12. Apply existing fast_nms_stars
    stars_xyf = fast_nms_stars(
        candidates,
        min_dist_px=float(min_dist_px),
        max_candidates=int(max_candidates),
    )

    # 13. Visualization mask: radius-4 circles around final detections
    mask = np.zeros_like(norm_th, dtype=np.uint8)
    for star in stars_xyf:
        cv2.circle(
            mask,
            (round(float(star[0])), round(float(star[1]))),
            _MASK_CIRCLE_RADIUS,
            255,
            -1,
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "detector": DETECTOR_NAME,
        "stars_xyf": stars_xyf,
        "preprocessed": norm_th,
        "mask": mask,
        "latency_ms": float(latency_ms),
    }

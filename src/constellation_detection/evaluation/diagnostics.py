"""Star detection diagnostics, SNR computation, and latency summaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from constellation_detection.image.preprocessing import to_grayscale

__all__ = [
    "compute_star_annulus_snr",
    "summarize_detector_result",
]

_PATCH_HALF: int = 10  # 21 x 21 patch (-10..+10)
_EDGE_MARGIN_PX: int = 11
_CORE_R2_MAX: int = 9
_ANNULUS_R2_MIN: int = 36
_ANNULUS_R2_MAX: int = 100

_DY, _DX = np.mgrid[-_PATCH_HALF : _PATCH_HALF + 1, -_PATCH_HALF : _PATCH_HALF + 1]
_R2 = _DX * _DX + _DY * _DY
_CORE_MASK = _R2 <= _CORE_R2_MAX
_ANNULUS_MASK = (_R2 >= _ANNULUS_R2_MIN) & (_R2 <= _ANNULUS_R2_MAX)


def compute_star_annulus_snr(
    gray_img: np.ndarray,
    stars_xyf: np.ndarray,
    top_k: int = 30,
) -> float:
    """Compute mean local annulus Signal-to-Noise Ratio (SNR) for top detections.

    Preserves the Phase 1 diagnostic formula:
        - Uses a 21 x 21 patch centered on each star detection.
        - Core mask: squared radius ``<= 9`` (radius <= 3 px).
        - Background annulus mask: squared radius in ``[36, 100]`` (6 px <= r <= 10 px).
        - ``SNR = (max(core) - median(annulus)) / max(std(annulus), 1.0)``.
        - Excludes detections closer than 11 pixels to any image boundary.

    Args:
        gray_img: 2D grayscale image (or 3D BGR image, which is converted to 2D grayscale).
        stars_xyf: 2D array of shape ``(N, 3)`` with columns ``[x, y, flux_score]``.
        top_k: Maximum number of top-ranked detections to evaluate. Defaults to ``30``.

    Returns:
        Mean annulus SNR rounded to two decimal places, or ``0.0`` if ``stars_xyf``
        is empty or no detections satisfy the boundary margin.

    Raises:
        TypeError: If ``top_k`` is not an integer or ``gray_img`` is not a ``numpy.ndarray``.
        ValueError: If ``top_k <= 0`` or ``stars_xyf`` does not have shape ``(N, >=2)``.
    """
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)):
        raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")

    gray = to_grayscale(gray_img).astype(np.float64)

    if stars_xyf is None:
        return 0.0

    stars_arr = np.asarray(stars_xyf)
    if stars_arr.size == 0:
        return 0.0
    if stars_arr.ndim != 2 or stars_arr.shape[1] < 2:
        raise ValueError(
            f"Expected stars_xyf to have shape (N, 3) or (N, >=2), got {stars_arr.shape}."
        )

    height, width = gray.shape
    snrs: list[float] = []

    for star in stars_arr[: int(top_k)]:
        ix = round(float(star[0]))
        iy = round(float(star[1]))

        if (
            ix < _EDGE_MARGIN_PX
            or ix >= width - _EDGE_MARGIN_PX
            or iy < _EDGE_MARGIN_PX
            or iy >= height - _EDGE_MARGIN_PX
        ):
            continue

        patch = gray[
            iy - _PATCH_HALF : iy + _PATCH_HALF + 1,
            ix - _PATCH_HALF : ix + _PATCH_HALF + 1,
        ]
        core_max = float(np.max(patch[_CORE_MASK]))
        annulus_vals = patch[_ANNULUS_MASK]
        bg_median = float(np.median(annulus_vals))
        bg_std = max(float(np.std(annulus_vals)), 1.0)
        snrs.append((core_max - bg_median) / bg_std)

    if not snrs:
        return 0.0

    return round(float(np.mean(snrs)), 2)


def summarize_detector_result(
    result: dict[str, Any],
    gray_img: np.ndarray,
    top_k: int = 30,
) -> dict[str, Any]:
    """Summarize detection count, top-K annulus SNR, and latency for a detector output.

    Args:
        result: Standardized detector output dictionary containing ``detector``,
            ``stars_xyf``, and ``latency_ms``.
        gray_img: 2D grayscale (or 3D BGR) working-resolution input image.
        top_k: Number of top detections to include in annulus SNR. Defaults to ``30``.

    Returns:
        Dictionary with keys ``detector``, ``total_stars``, ``top30_snr``, and ``latency_ms``.
    """
    stars_xyf = result["stars_xyf"]
    total_stars = len(stars_xyf)
    top30_snr = compute_star_annulus_snr(gray_img, stars_xyf, top_k=top_k)
    latency_ms = round(float(result["latency_ms"]), 2)

    return {
        "detector": str(result["detector"]),
        "total_stars": total_stars,
        "top30_snr": top30_snr,
        "latency_ms": latency_ms,
    }

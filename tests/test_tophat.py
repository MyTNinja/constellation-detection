"""Unit tests for constellation_detection.detection.traditional.tophat."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from constellation_detection.detection.traditional import (
    detect_stars_adaptive_tophat as detect_pkg,
)
from constellation_detection.detection.traditional.tophat import (
    DETECTOR_NAME,
    detect_stars_adaptive_tophat,
)


def test_reexport_from_traditional_package() -> None:
    """Verify detect_stars_adaptive_tophat is re-exported by traditional subpackage."""
    assert detect_stars_adaptive_tophat is detect_pkg


def test_black_image_returns_zero_detections() -> None:
    """Black image produces empty (0, 3) float32 detections and standardized schema."""
    black_bgr = np.zeros((100, 120, 3), dtype=np.uint8)
    result = detect_stars_adaptive_tophat(black_bgr)

    assert set(result.keys()) == {
        "detector",
        "stars_xyf",
        "preprocessed",
        "mask",
        "latency_ms",
    }
    assert result["detector"] == "CV2_AdaptiveTopHat"
    assert result["detector"] == DETECTOR_NAME
    assert isinstance(result["stars_xyf"], np.ndarray)
    assert result["stars_xyf"].shape == (0, 3)
    assert result["stars_xyf"].dtype == np.float32

    assert result["preprocessed"].shape == (100, 120)
    assert result["preprocessed"].dtype == np.uint8
    assert np.all(result["preprocessed"] == 0)

    assert result["mask"].shape == (100, 120)
    assert result["mask"].dtype == np.uint8
    assert np.all(result["mask"] == 0)

    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0


def test_synthetic_bright_stars_detected_and_localized() -> None:
    """Well-separated synthetic stars are detected and localized accurately."""
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2.circle(image, (40, 45), radius=3, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (115, 50), radius=2, color=(220, 220, 220), thickness=-1)
    cv2.circle(image, (80, 120), radius=2, color=(190, 190, 190), thickness=-1)

    result = detect_stars_adaptive_tophat(image)
    stars = result["stars_xyf"]

    assert stars.shape == (3, 3)
    assert stars.dtype == np.float32
    assert np.all(stars[:, 2] > 0.0)
    assert stars[0, 2] >= stars[1, 2] >= stars[2, 2]

    # Verify each expected star coordinate is found within 1.5 px
    expected_centers = [(40.0, 45.0), (115.0, 50.0), (80.0, 120.0)]
    for ex, ey in expected_centers:
        dists = np.hypot(stars[:, 0] - ex, stars[:, 1] - ey)
        assert float(np.min(dists)) <= 1.5


def test_detects_stars_on_uneven_background_gradient() -> None:
    """White Top-Hat suppresses smooth light-pollution gradients and detects stars."""
    h, w = 160, 160
    _, xx = np.mgrid[0:h, 0:w]
    # Strong horizontal sky gradient from 10 to 120
    bg = (10.0 + 110.0 * (xx / float(w - 1))).astype(np.uint8)
    image = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    # Place one star in the dark half and one star in the bright gradient half
    cv2.circle(image, (35, 80), radius=2, color=(240, 240, 240), thickness=-1)
    cv2.circle(image, (130, 80), radius=2, color=(255, 255, 255), thickness=-1)

    result = detect_stars_adaptive_tophat(image)
    stars = result["stars_xyf"]

    assert len(stars) == 2
    for ex, ey in [(35.0, 80.0), (130.0, 80.0)]:
        dists = np.hypot(stars[:, 0] - ex, stars[:, 1] - ey)
        assert float(np.min(dists)) <= 1.5


def test_area_filtering_rejects_oversized_component() -> None:
    """Large blobs (> 180 px area) are rejected while valid stars (2..180 px) are retained."""
    image = np.zeros((200, 200), dtype=np.uint8)
    # Normal star
    cv2.circle(image, (150, 150), radius=3, color=255, thickness=-1)
    # Large bright ring/disk whose Top-Hat response exceeds 180 px area
    cv2.rectangle(image, (20, 20), (70, 70), color=255, thickness=4)

    result = detect_stars_adaptive_tophat(image)
    stars = result["stars_xyf"]

    assert len(stars) >= 1
    dists_star = np.hypot(stars[:, 0] - 150.0, stars[:, 1] - 150.0)
    assert float(np.min(dists_star)) <= 1.5
    # Center of the huge rectangle should not be detected as a single large component
    dists_rect = np.hypot(stars[:, 0] - 45.0, stars[:, 1] - 45.0)
    assert float(np.min(dists_rect)) > 10.0


def test_flux_weighted_centroid_shifts_toward_brighter_side() -> None:
    """Flux-weighted centroid is pulled toward the brighter side of an asymmetric star."""
    image = np.zeros((100, 100), dtype=np.uint8)
    # 4x2 horizontal bar with much brighter right half
    image[50:52, 48:50] = 150
    image[50:52, 50:52] = 255

    result = detect_stars_adaptive_tophat(image)
    stars = result["stars_xyf"]

    assert stars.shape == (1, 3)
    # Geometric center is x=49.5; flux-weighted centroid should be > 49.5
    assert stars[0, 0] > 49.5


def test_nms_suppresses_nearby_secondary_star() -> None:
    """Secondary star within min_dist_px of a brighter star is suppressed by NMS."""
    image = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(image, (50, 50), radius=2, color=255, thickness=-1)
    cv2.circle(image, (57, 50), radius=2, color=180, thickness=-1)

    result = detect_stars_adaptive_tophat(image, min_dist_px=10.0)
    stars = result["stars_xyf"]

    assert stars.shape == (1, 3)
    np.testing.assert_allclose(stars[0, :2], [50.0, 50.0], atol=1.5)


@pytest.mark.parametrize(
    "invalid_image,expected_exc",
    [
        (None, TypeError),
        ([[0, 0], [0, 0]], TypeError),
        (np.zeros((32, 32), dtype=np.float32), ValueError),
        (np.zeros((0, 32), dtype=np.uint8), ValueError),
        (np.zeros((32,), dtype=np.uint8), ValueError),
    ],
)
def test_invalid_image_inputs_raise_errors(
    invalid_image: object,
    expected_exc: type[Exception],
) -> None:
    """Invalid image types, dtypes, or dimensions raise TypeError or ValueError."""
    with pytest.raises(expected_exc):
        detect_stars_adaptive_tophat(invalid_image)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "min_dist_px,max_candidates,expected_exc",
    [
        (-1.0, 250, ValueError),
        (float("nan"), 250, ValueError),
        (float("inf"), 250, ValueError),
        (True, 250, TypeError),
        (10.0, 0, ValueError),
        (10.0, -5, ValueError),
        (10.0, 10.5, TypeError),
        (10.0, False, TypeError),
    ],
)
def test_invalid_nms_parameters_raise_errors(
    min_dist_px: object,
    max_candidates: object,
    expected_exc: type[Exception],
) -> None:
    """Invalid min_dist_px or max_candidates raise TypeError or ValueError."""
    image = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(expected_exc):
        detect_stars_adaptive_tophat(
            image,
            min_dist_px=min_dist_px,  # type: ignore[arg-type]
            max_candidates=max_candidates,  # type: ignore[arg-type]
        )

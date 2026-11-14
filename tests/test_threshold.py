"""Unit tests for constellation_detection.detection.traditional.threshold."""

from __future__ import annotations

import math
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from constellation_detection.detection.traditional import (
    DETECTOR_NAME,
)
from constellation_detection.detection.traditional import (
    detect_stars_global_threshold as detect_pkg,
)
from constellation_detection.detection.traditional.threshold import (
    detect_stars_global_threshold,
)


def test_reexport_from_traditional_package() -> None:
    """Verify detect_stars_global_threshold is re-exported by traditional subpackage."""
    assert detect_stars_global_threshold is detect_pkg


def test_black_image_returns_zero_detections() -> None:
    """Black image produces empty (0, 3) float32 detections and zero mask."""
    black_bgr = np.zeros((100, 120, 3), dtype=np.uint8)
    result = detect_stars_global_threshold(black_bgr)

    assert set(result.keys()) == {
        "detector",
        "stars_xyf",
        "preprocessed",
        "mask",
        "latency_ms",
    }
    assert result["detector"] == DETECTOR_NAME
    assert isinstance(result["stars_xyf"], np.ndarray)
    assert result["stars_xyf"].shape == (0, 3)
    assert result["stars_xyf"].dtype == np.float32

    assert isinstance(result["preprocessed"], np.ndarray)
    assert result["preprocessed"].shape == (100, 120)
    assert result["preprocessed"].dtype == np.uint8
    assert np.all(result["preprocessed"] == 0)

    assert isinstance(result["mask"], np.ndarray)
    assert result["mask"].shape == (100, 120)
    assert result["mask"].dtype == np.uint8
    assert np.all(result["mask"] == 0)

    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0


def test_synthetic_bright_stars_detected_and_sorted() -> None:
    """Well-separated synthetic bright stars are accurately localized and scored."""
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    # Draw three bright synthetic stars with different radii/intensities
    cv2.circle(image, (40, 45), radius=3, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (110, 50), radius=2, color=(220, 220, 220), thickness=-1)
    cv2.circle(image, (80, 120), radius=2, color=(180, 180, 180), thickness=-1)

    result = detect_stars_global_threshold(image)
    stars = result["stars_xyf"]

    assert stars.shape == (3, 3)
    assert stars.dtype == np.float32
    # Scores must be positive and sorted descending
    assert np.all(stars[:, 2] > 0.0)
    assert stars[0, 2] >= stars[1, 2] >= stars[2, 2]

    # Brightest star should be near (40, 45), second near (110, 50), third near (80, 120)
    np.testing.assert_allclose(stars[0, :2], [40.0, 45.0], atol=1.0)
    np.testing.assert_allclose(stars[1, :2], [110.0, 50.0], atol=1.0)
    np.testing.assert_allclose(stars[2, :2], [80.0, 120.0], atol=1.0)


def test_accepts_grayscale_2d_input() -> None:
    """Detector accepts already-grayscale 2D uint8 images directly."""
    gray = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(gray, (40, 40), radius=3, color=255, thickness=-1)

    result = detect_stars_global_threshold(gray)

    assert result["stars_xyf"].shape == (1, 3)
    np.testing.assert_allclose(result["stars_xyf"][0, :2], [40.0, 40.0], atol=1.0)


def test_large_contour_area_above_350_rejected() -> None:
    """Contours with area > 350 are rejected while valid stars are kept."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    # Large bright blob (radius 15 -> area ~ 706 > 350)
    cv2.circle(image, (60, 60), radius=15, color=(255, 255, 255), thickness=-1)
    # Normal star (radius 3 -> area ~ 28 <= 350)
    cv2.circle(image, (150, 150), radius=3, color=(255, 255, 255), thickness=-1)

    result = detect_stars_global_threshold(image)
    stars = result["stars_xyf"]

    assert stars.shape == (1, 3)
    np.testing.assert_allclose(stars[0, :2], [150.0, 150.0], atol=1.0)


def test_nms_integration_suppresses_nearby_secondary_detection() -> None:
    """Secondary star within 10 px of a brighter star is suppressed by NMS."""
    image = np.zeros((120, 120), dtype=np.uint8)
    # Brighter star at (50, 50)
    cv2.circle(image, (50, 50), radius=2, color=255, thickness=-1)
    # Dimmer star 7 px away at (57, 50) -> separated enough for distinct contour after opening,
    # but within min_dist_px=10.0 so NMS suppresses it
    cv2.circle(image, (57, 50), radius=2, color=180, thickness=-1)

    result = detect_stars_global_threshold(image, min_dist_px=10.0)
    stars = result["stars_xyf"]

    assert stars.shape == (1, 3)
    np.testing.assert_allclose(stars[0, :2], [50.0, 50.0], atol=1.0)


def test_zero_moment_fallback_and_score_formula() -> None:
    """Zero-moment contours fall back to cnt[0][0] and follow exact score formula."""
    image = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(image, (32, 28), radius=2, color=255, thickness=-1)

    zero_moments = {"m00": 0.0, "m10": 0.0, "m01": 0.0}
    with patch("cv2.moments", return_value=zero_moments):
        result = detect_stars_global_threshold(image)

    stars = result["stars_xyf"]
    assert stars.shape == (1, 3)

    # Verify the score matches blurred[iy, ix] * sqrt(max(1, min(area, 25)))
    contours, _ = cv2.findContours(
        result["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    assert len(contours) == 1
    cnt = contours[0]
    area = float(cv2.contourArea(cnt))
    expected_cx = float(cnt[0][0][0])
    expected_cy = float(cnt[0][0][1])
    ix = round(expected_cx)
    iy = round(expected_cy)
    expected_score = float(result["preprocessed"][iy, ix]) * math.sqrt(
        max(1.0, min(area, 25.0))
    )

    np.testing.assert_allclose(
        stars[0],
        [expected_cx, expected_cy, expected_score],
        rtol=1e-5,
    )


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
        detect_stars_global_threshold(invalid_image)  # type: ignore[arg-type]


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
        detect_stars_global_threshold(
            image,
            min_dist_px=min_dist_px,  # type: ignore[arg-type]
            max_candidates=max_candidates,  # type: ignore[arg-type]
        )

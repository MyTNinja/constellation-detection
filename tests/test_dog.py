"""Unit tests for constellation_detection.detection.traditional.dog."""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from constellation_detection.detection.traditional import (
    detect_stars_multiscale_dog as detect_pkg,
)
from constellation_detection.detection.traditional.dog import (
    DETECTOR_NAME,
    detect_stars_multiscale_dog,
)


def test_reexport_from_traditional_package() -> None:
    """Verify detect_stars_multiscale_dog is re-exported by traditional subpackage."""
    assert detect_stars_multiscale_dog is detect_pkg


def test_black_image_returns_zero_detections() -> None:
    """Black image produces empty (0, 3) float32 detections and empty mask."""
    black_bgr = np.zeros((100, 120, 3), dtype=np.uint8)
    result = detect_stars_multiscale_dog(black_bgr)

    assert set(result.keys()) == {
        "detector",
        "stars_xyf",
        "preprocessed",
        "mask",
        "latency_ms",
    }
    assert result["detector"] == "CV3_MultiScaleDoG"
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


def test_synthetic_multiscale_stars_detected_and_masked() -> None:
    """Multi-scale synthetic stars are detected, scored, and drawn on the mask."""
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    # Small, medium, and larger synthetic stars
    cv2.circle(image, (40, 45), radius=2, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (115, 50), radius=3, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (80, 120), radius=4, color=(255, 255, 255), thickness=-1)

    result = detect_stars_multiscale_dog(image)
    stars = result["stars_xyf"]

    assert stars.shape == (3, 3)
    assert stars.dtype == np.float32
    assert np.all(stars[:, 2] > 0.0)
    assert stars[0, 2] >= stars[1, 2] >= stars[2, 2]

    for ex, ey in [(40.0, 45.0), (115.0, 50.0), (80.0, 120.0)]:
        dists = np.hypot(stars[:, 0] - ex, stars[:, 1] - ey)
        assert float(np.min(dists)) <= 1.5

    # Visualization mask should have radius-4 filled circles (value 255) at detections
    for star in stars:
        ix, iy = round(float(star[0])), round(float(star[1]))
        assert result["mask"][iy, ix] == 255


def test_zero_flux_patch_fallback_uses_blob_coords_and_peak_one() -> None:
    """When local patch flux is zero, fallback uses (bx, by) and score = 1.0 * sigma."""
    black_img = np.zeros((64, 64), dtype=np.uint8)
    fake_blobs = np.array([[20.25, 30.75, 2.5]], dtype=np.float64)

    with patch(
        "constellation_detection.detection.traditional.dog.blob_dog",
        return_value=fake_blobs,
    ) as mock_dog:
        result = detect_stars_multiscale_dog(black_img)

    mock_dog.assert_called_once()
    stars = result["stars_xyf"]
    assert stars.shape == (1, 3)
    # Fallback: cx = bx = 30.75, cy = by = 20.25, score = 1.0 * 2.5 = 2.5
    np.testing.assert_allclose(stars[0], [30.75, 20.25, 2.5], rtol=1e-5)


def test_nms_suppresses_nearby_secondary_star() -> None:
    """Secondary detection within min_dist_px is suppressed by NMS."""
    image = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(image, (50, 50), radius=2, color=255, thickness=-1)
    cv2.circle(image, (57, 50), radius=2, color=200, thickness=-1)

    result = detect_stars_multiscale_dog(image, min_dist_px=10.0)
    stars = result["stars_xyf"]

    assert stars.shape == (1, 3)
    np.testing.assert_allclose(stars[0, :2], [50.0, 50.0], atol=2.0)


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
        detect_stars_multiscale_dog(invalid_image)  # type: ignore[arg-type]


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
        detect_stars_multiscale_dog(
            image,
            min_dist_px=min_dist_px,  # type: ignore[arg-type]
            max_candidates=max_candidates,  # type: ignore[arg-type]
        )

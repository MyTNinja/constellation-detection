"""Unit tests for constellation_detection.evaluation.diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from constellation_detection.evaluation.diagnostics import (
    compute_star_annulus_snr,
    summarize_detector_result,
)


def test_empty_detections_return_zero_snr() -> None:
    """Empty detections array returns 0.0."""
    gray = np.full((64, 64), 20, dtype=np.uint8)
    empty = np.empty((0, 3), dtype=np.float32)

    assert compute_star_annulus_snr(gray, empty) == 0.0


def test_known_synthetic_signal_and_flat_background_snr() -> None:
    """Flat background std < 1.0 clamps denominator to 1.0, yielding exact SNR."""
    gray = np.full((64, 64), 20, dtype=np.uint8)
    gray[32, 32] = 220  # Core max = 220, annulus median = 20, annulus std = 0 -> max(..., 1.0) = 1.0

    stars = np.array([[32.0, 32.0, 100.0]], dtype=np.float32)
    snr = compute_star_annulus_snr(gray, stars, top_k=30)

    assert snr == 200.0


def test_known_synthetic_signal_with_noisy_annulus_snr() -> None:
    """Non-zero annulus standard deviation (> 1.0) is used directly as denominator."""
    gray = np.full((64, 64), 30, dtype=np.uint8)
    # Create deterministic alternating background values (26 and 34 -> median=30, std=4.0)
    yy, xx = np.mgrid[0:64, 0:64]
    checker = np.where((xx + yy) % 2 == 0, 26, 34).astype(np.uint8)
    gray[:, :] = checker
    # Star core peak at center
    gray[30, 30] = 110

    stars = np.array([[30.0, 30.0, 50.0]], dtype=np.float32)
    snr = compute_star_annulus_snr(gray, stars, top_k=30)

    # Compute exact expected value from the annulus pixels
    patch = gray[20:41, 20:41].astype(np.float64)
    dy, dx = np.mgrid[-10:11, -10:11]
    r2 = dx * dx + dy * dy
    annulus = patch[(r2 >= 36) & (r2 <= 100)]
    expected = round((110.0 - float(np.median(annulus))) / max(float(np.std(annulus)), 1.0), 2)

    assert snr == expected


def test_edge_detections_within_11px_margin_are_excluded() -> None:
    """Detections closer than 11 pixels to any image edge are ignored."""
    gray = np.full((64, 64), 10, dtype=np.uint8)
    # Place stars strictly inside the 11-pixel exclusion margin on all four borders
    edge_stars = np.array(
        [
            [5.0, 32.0, 100.0],
            [10.0, 32.0, 90.0],
            [54.0, 32.0, 80.0],  # width - 11 = 53; 54 >= 53 is excluded
            [32.0, 10.0, 70.0],
            [32.0, 53.0, 60.0],  # height - 11 = 53; 53 >= 53 is excluded
        ],
        dtype=np.float32,
    )

    assert compute_star_annulus_snr(gray, edge_stars) == 0.0


def test_summarize_detector_result() -> None:
    """summarize_detector_result returns expected count, SNR, and rounded latency."""
    gray = np.full((64, 64), 15, dtype=np.uint8)
    gray[32, 32] = 115

    fake_result = {
        "detector": "CV2_AdaptiveTopHat",
        "stars_xyf": np.array([[32.0, 32.0, 500.0]], dtype=np.float32),
        "latency_ms": 12.3456,
    }
    summary = summarize_detector_result(fake_result, gray, top_k=30)

    assert summary == {
        "detector": "CV2_AdaptiveTopHat",
        "total_stars": 1,
        "top30_snr": 100.0,
        "latency_ms": 12.35,
    }


@pytest.mark.parametrize(
    "top_k,expected_exc",
    [
        (0, ValueError),
        (-5, ValueError),
        (True, TypeError),
        (3.5, TypeError),
    ],
)
def test_invalid_top_k_raises(top_k: object, expected_exc: type[Exception]) -> None:
    """Invalid top_k arguments raise TypeError or ValueError."""
    gray = np.zeros((32, 32), dtype=np.uint8)
    stars = np.empty((0, 3), dtype=np.float32)
    with pytest.raises(expected_exc):
        compute_star_annulus_snr(gray, stars, top_k=top_k)  # type: ignore[arg-type]

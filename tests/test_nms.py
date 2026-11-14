"""Unit tests for constellation_detection.detection.nms."""

from __future__ import annotations

import numpy as np

from constellation_detection.detection import fast_nms_stars as fast_nms_stars_pkg
from constellation_detection.detection.nms import fast_nms_stars


def test_import_reexport() -> None:
    """Verify fast_nms_stars is importable from both module and subpackage."""
    assert fast_nms_stars is fast_nms_stars_pkg


def test_empty_input_returns_empty_float32_array() -> None:
    """1. Empty input returns shape (0, 3) and dtype float32."""
    result = fast_nms_stars([])

    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 3)
    assert result.dtype == np.float32


def test_single_candidate_retained() -> None:
    """2. A single candidate is retained."""
    candidates = [[42.5, 108.25, 0.95]]
    result = fast_nms_stars(candidates)

    assert result.shape == (1, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0], [42.5, 108.25, 0.95], rtol=1e-6)


def test_candidates_sorted_by_descending_brightness() -> None:
    """3. Candidates are sorted by descending brightness."""
    candidates = [
        [10.0, 10.0, 15.0],
        [50.0, 50.0, 99.5],
        [90.0, 90.0, 42.0],
        [130.0, 130.0, 5.25],
    ]
    result = fast_nms_stars(candidates, min_dist_px=10.0)

    assert result.shape == (4, 3)
    expected_scores = np.array([99.5, 42.0, 15.0, 5.25], dtype=np.float32)
    np.testing.assert_allclose(result[:, 2], expected_scores, rtol=1e-6)


def test_weaker_candidate_inside_suppression_radius_removed() -> None:
    """4. A weaker candidate inside the suppression radius is removed."""
    candidates = [
        [100.0, 100.0, 50.0],  # Weaker star within 5 px of brighter star
        [100.0, 105.0, 120.0],  # Brightest star core
        [103.0, 101.0, 80.0],  # Also within 10 px of (100, 105) -> dist = 5.0 px
    ]
    result = fast_nms_stars(candidates, min_dist_px=10.0)

    assert result.shape == (1, 3)
    np.testing.assert_allclose(result[0], [100.0, 105.0, 120.0], rtol=1e-6)


def test_candidates_outside_suppression_radius_retained() -> None:
    """5. Candidates outside the suppression radius are retained."""
    candidates = [
        [20.0, 20.0, 100.0],
        [20.0, 30.5, 80.0],  # Distance 10.5 > 10.0
        [35.0, 20.0, 60.0],  # Distance 15.0 > 10.0 from first, > 10.0 from second
    ]
    result = fast_nms_stars(candidates, min_dist_px=10.0)

    assert result.shape == (3, 3)
    np.testing.assert_allclose(result[:, 2], [100.0, 80.0, 60.0], rtol=1e-6)


def test_max_candidates_limit_respected() -> None:
    """6. The maximum candidate limit is respected."""
    # 10 well-separated candidates (spaced 50 px apart)
    candidates = [[float(i * 50), 0.0, float(i + 1)] for i in range(10)]
    result = fast_nms_stars(candidates, min_dist_px=10.0, max_candidates=4)

    assert result.shape == (4, 3)
    assert result.dtype == np.float32
    # Should keep the 4 brightest candidates (scores 10.0, 9.0, 8.0, 7.0)
    np.testing.assert_allclose(result[:, 2], [10.0, 9.0, 8.0, 7.0], rtol=1e-6)


def test_output_three_column_format_and_dtype() -> None:
    """7. Output coordinates and scores are returned in the expected three-column format."""
    candidates = [
        (12.25, 34.5, 255.0),
        (100.75, 200.125, 128.5),
    ]
    result = fast_nms_stars(candidates, min_dist_px=5.0, max_candidates=10)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (2, 3)
    assert result.dtype == np.float32
    expected = np.array(
        [
            [12.25, 34.5, 255.0],
            [100.75, 200.125, 128.5],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)

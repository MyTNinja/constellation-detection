"""Unit tests for constellation_detection.visualization.overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from constellation_detection.detection.traditional import (
    detect_stars_adaptive_tophat,
    detect_stars_global_threshold,
    detect_stars_multiscale_dog,
)
from constellation_detection.visualization.overlays import (
    DETECTOR_COLORS,
    plot_detector_diagnostics,
)


def test_plot_detector_diagnostics_four_panels_and_png_export(tmp_path: Path) -> None:
    """Four-panel figure renders headlessly and exports a non-empty PNG file."""
    image = np.zeros((160, 180, 3), dtype=np.uint8)
    cv2.circle(image, (40, 40), radius=3, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (60, 55), radius=2, color=(230, 230, 230), thickness=-1)
    cv2.circle(image, (130, 110), radius=3, color=(240, 240, 240), thickness=-1)

    for detector_fn in (
        detect_stars_global_threshold,
        detect_stars_adaptive_tophat,
        detect_stars_multiscale_dog,
    ):
        result = detector_fn(image)
        fig, saved_path = plot_detector_diagnostics(
            image_bgr=image,
            result=result,
            output_dir=tmp_path / "viz",
            max_dim=800,
            top_k_show=35,
            connect_top_n=14,
            connect_rad_ratio=0.22,
        )

        assert len(fig.axes) == 4
        assert saved_path is not None
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0
        assert result["detector"] in DETECTOR_COLORS


def test_plot_detector_diagnostics_empty_detections_no_save() -> None:
    """Empty detections render cleanly without errors when output_dir is None."""
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detect_stars_adaptive_tophat(black)

    fig, saved_path = plot_detector_diagnostics(
        image_bgr=black,
        result=result,
        output_dir=None,
    )

    assert len(fig.axes) == 4
    assert saved_path is None

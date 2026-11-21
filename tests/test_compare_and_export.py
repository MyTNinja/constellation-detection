"""Unit tests for detector comparison and prediction CSV export."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from constellation_detection.evaluation.compare import (
    LEADERBOARD_COLUMNS,
    METRICS_COLUMNS,
    build_detector_leaderboard,
    evaluate_image_detectors,
    resize_to_max_dim,
)
from constellation_detection.evaluation.export import (
    PREDICTION_CSV_COLUMNS,
    build_image_output_stem,
    export_predictions_csv,
    load_predictions_csv,
)


def _make_synthetic_starfield(height: int = 180, width: int = 200) -> np.ndarray:
    """Create a deterministic BGR synthetic starfield with 4 bright stars."""
    img = np.full((height, width, 3), 10, dtype=np.uint8)
    for cx, cy, radius, val in [
        (45, 50, 3, 255),
        (140, 45, 2, 235),
        (95, 110, 3, 245),
        (155, 140, 2, 220),
    ]:
        cv2.circle(img, (cx, cy), radius=radius, color=(val, val, val), thickness=-1)
    return img


def test_export_and_load_predictions_csv_roundtrip(tmp_path: Path) -> None:
    """Prediction CSV export writes exact headers and round-trips cleanly."""
    stars = np.array(
        [
            [45.25, 50.5, 980.0],
            [140.0, 45.75, 720.5],
        ],
        dtype=np.float32,
    )
    csv_path = export_predictions_csv(stars, tmp_path / "preds" / "sample.csv")

    assert csv_path.exists()
    loaded = load_predictions_csv(csv_path)

    assert tuple(loaded.columns) == PREDICTION_CSV_COLUMNS
    assert list(loaded["rank"]) == [1, 2]
    np.testing.assert_allclose(loaded["x_px"].to_numpy(), [45.25, 140.0], rtol=1e-5)
    np.testing.assert_allclose(loaded["y_px"].to_numpy(), [50.5, 45.75], rtol=1e-5)
    np.testing.assert_allclose(loaded["flux_score"].to_numpy(), [980.0, 720.5], rtol=1e-5)


def test_export_empty_predictions_preserves_headers(tmp_path: Path) -> None:
    """Empty (0, 3) detections export a valid header-only CSV."""
    empty = np.empty((0, 3), dtype=np.float32)
    csv_path = export_predictions_csv(empty, tmp_path / "empty_preds.csv")

    loaded = load_predictions_csv(csv_path)
    assert loaded.empty
    assert tuple(loaded.columns) == PREDICTION_CSV_COLUMNS


def test_evaluate_image_detectors_runs_all_three_and_builds_leaderboard(
    tmp_path: Path,
) -> None:
    """evaluate_image_detectors runs all 3 detectors, exports CSVs, and aggregates leaderboard."""
    starfield = _make_synthetic_starfield()
    preds_dir = tmp_path / "predictions"

    metrics_df, results, working_img = evaluate_image_detectors(
        image=starfield,
        image_name="synth_field.png",
        predictions_dir=preds_dir,
        max_dim=800,
    )

    assert working_img.shape == starfield.shape
    assert len(results) == 3
    assert tuple(metrics_df.columns) == METRICS_COLUMNS
    assert len(metrics_df) == 3
    assert list(metrics_df["detector"]) == [
        "global_threshold",
        "CV2_AdaptiveTopHat",
        "CV3_MultiScaleDoG",
    ]
    assert (metrics_df["resolution"] == "200x180").all()
    assert (metrics_df["total_stars"] > 0).all()
    assert (metrics_df["top30_snr"] > 0.0).all()
    assert (metrics_df["latency_ms"] >= 0.0).all()

    # Verify each saved prediction CSV exists and has exact schema
    for csv_str in metrics_df["saved_csv"]:
        csv_file = Path(csv_str)
        assert csv_file.exists()
        df_pred = load_predictions_csv(csv_file)
        assert tuple(df_pred.columns) == PREDICTION_CSV_COLUMNS
        assert len(df_pred) > 0

    leaderboard = build_detector_leaderboard(metrics_df)
    assert tuple(leaderboard.columns) == LEADERBOARD_COLUMNS
    assert len(leaderboard) == 3


def test_evaluate_black_image_handles_empty_detections(tmp_path: Path) -> None:
    """Black image produces zero stars, 0.0 SNR, and valid header-only CSVs."""
    black = np.zeros((100, 120, 3), dtype=np.uint8)
    metrics_df, results, _ = evaluate_image_detectors(
        image=black,
        image_name="black.png",
        predictions_dir=tmp_path / "preds",
    )

    assert len(results) == 3
    assert (metrics_df["total_stars"] == 0).all()
    assert (metrics_df["top30_snr"] == 0.0).all()
    for csv_str in metrics_df["saved_csv"]:
        df_pred = load_predictions_csv(csv_str)
        assert df_pred.empty
        assert tuple(df_pred.columns) == PREDICTION_CSV_COLUMNS


def test_resize_to_max_dim_downscales_proportionally() -> None:
    """Images larger than max_dim are proportionally downscaled."""
    large = np.zeros((1000, 1600, 3), dtype=np.uint8)
    resized = resize_to_max_dim(large, max_dim=800)

    assert resized.shape == (500, 800, 3)


def test_build_detector_leaderboard_empty_dataframe() -> None:
    """Empty metrics DataFrame returns an empty leaderboard with expected columns."""
    empty_metrics = pd.DataFrame(columns=list(METRICS_COLUMNS))
    leaderboard = build_detector_leaderboard(empty_metrics)

    assert leaderboard.empty
    assert tuple(leaderboard.columns) == LEADERBOARD_COLUMNS


def test_build_detector_leaderboard_missing_columns_raises() -> None:
    """Missing required columns raise ValueError."""
    bad_df = pd.DataFrame({"detector": ["global_threshold"]})
    with pytest.raises(ValueError, match="missing required columns"):
        build_detector_leaderboard(bad_df)


def test_same_stem_different_extension_produces_distinct_csv_paths(
    tmp_path: Path,
) -> None:
    """Images with identical stems (field.jpg vs field.png) do not overwrite each other's CSVs."""
    assert build_image_output_stem("field.jpg") == "field_jpg"
    assert build_image_output_stem("field.png") == "field_png"
    assert build_image_output_stem("field.jpg") != build_image_output_stem("field.png")

    starfield_jpg = _make_synthetic_starfield()
    starfield_png = _make_synthetic_starfield()
    # Add an extra bright star to field.png so CSV outputs are measurably distinct
    cv2.circle(starfield_png, (30, 140), radius=3, color=(255, 255, 255), thickness=-1)

    preds_dir = tmp_path / "predictions"
    metrics_jpg, _, _ = evaluate_image_detectors(
        image=starfield_jpg,
        image_name="field.jpg",
        predictions_dir=preds_dir,
    )
    metrics_png, _, _ = evaluate_image_detectors(
        image=starfield_png,
        image_name="field.png",
        predictions_dir=preds_dir,
    )

    jpg_paths = set(metrics_jpg["saved_csv"])
    png_paths = set(metrics_png["saved_csv"])
    assert len(jpg_paths) == 3
    assert len(png_paths) == 3
    assert jpg_paths.isdisjoint(png_paths)

    for csv_str in jpg_paths | png_paths:
        csv_file = Path(csv_str)
        assert csv_file.exists()
        df = load_predictions_csv(csv_file)
        assert tuple(df.columns) == PREDICTION_CSV_COLUMNS

    assert all("field_jpg_" in Path(p).name for p in jpg_paths)
    assert all("field_png_" in Path(p).name for p in png_paths)

"""Evaluation diagnostics, detector comparison, and prediction CSV export."""

from __future__ import annotations

from constellation_detection.evaluation.compare import (
    LEADERBOARD_COLUMNS,
    METRICS_COLUMNS,
    build_detector_leaderboard,
    evaluate_image_detectors,
    resize_to_max_dim,
    run_all_detectors,
)
from constellation_detection.evaluation.diagnostics import (
    compute_star_annulus_snr,
    summarize_detector_result,
)
from constellation_detection.evaluation.export import (
    PREDICTION_CSV_COLUMNS,
    export_predictions_csv,
    load_predictions_csv,
)

__all__ = [
    "LEADERBOARD_COLUMNS",
    "METRICS_COLUMNS",
    "PREDICTION_CSV_COLUMNS",
    "build_detector_leaderboard",
    "compute_star_annulus_snr",
    "evaluate_image_detectors",
    "export_predictions_csv",
    "load_predictions_csv",
    "resize_to_max_dim",
    "run_all_detectors",
    "summarize_detector_result",
]

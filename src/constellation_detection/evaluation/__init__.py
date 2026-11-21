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
    build_image_output_stem,
    export_predictions_csv,
    load_predictions_csv,
)
from constellation_detection.evaluation.matching import (
    compute_average_precision,
    evaluate_phase1_baselines_on_manifest,
    match_detections_to_ground_truth,
    scale_working_stars_to_native,
    select_best_threshold,
)

__all__ = [
    "LEADERBOARD_COLUMNS",
    "METRICS_COLUMNS",
    "PREDICTION_CSV_COLUMNS",
    "build_detector_leaderboard",
    "build_image_output_stem",
    "compute_average_precision",
    "compute_star_annulus_snr",
    "evaluate_image_detectors",
    "evaluate_phase1_baselines_on_manifest",
    "export_predictions_csv",
    "load_predictions_csv",
    "match_detections_to_ground_truth",
    "resize_to_max_dim",
    "run_all_detectors",
    "scale_working_stars_to_native",
    "select_best_threshold",
    "summarize_detector_result",
]

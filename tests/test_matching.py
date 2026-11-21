"""Unit tests for Hungarian GT matching, ignore-masking, PR-AUC, and threshold selection."""

from __future__ import annotations

import math

import numpy as np

from constellation_detection.evaluation.matching import (
    compute_average_precision,
    match_detections_to_ground_truth,
    scale_working_stars_to_native,
    select_best_threshold,
)


def test_scale_working_stars_to_native_and_empty() -> None:
    empty = scale_working_stars_to_native(
        np.empty((0, 3), dtype=np.float32),
        working_shape=(400, 800),
        native_shape=(800, 1600),
    )
    assert empty.shape == (0, 3)

    pts = np.array([[100.0, 50.0, 200.0]], dtype=np.float32)
    scaled = scale_working_stars_to_native(
        pts, working_shape=(400, 800), native_shape=(800, 1600)
    )
    assert np.allclose(scaled[0], [200.0, 100.0, 200.0])


def test_hungarian_matching_ignore_and_stratified_saturation() -> None:
    gt = [
        {"x_native_px": 10.0, "y_native_px": 10.0, "is_saturated": True, "ignore": False},
        {"x_native_px": 50.0, "y_native_px": 50.0, "is_saturated": False, "ignore": False},
        {"x_native_px": 80.0, "y_native_px": 80.0, "is_saturated": False, "ignore": True},
    ]
    preds = np.array(
        [
            [11.0, 10.0, 0.95],  # TP on saturated star (err = 1.0)
            [80.5, 80.5, 0.50],  # Near ignore star -> neither TP nor FP
            [120.0, 120.0, 0.20],  # Genuine FP
        ],
        dtype=np.float32,
    )
    res = match_detections_to_ground_truth(preds, gt, tolerance_px=3.0)
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["fn"] == 1
    assert math.isclose(res["precision"], 0.5)
    assert math.isclose(res["recall"], 0.5)
    assert math.isclose(res["rmse_px"], 1.0)
    assert math.isclose(res["saturated_recall"], 1.0)


def test_pr_auc_and_validation_threshold_selection() -> None:
    gt_imgs = [
        [
            {"x_native_px": 20.0, "y_native_px": 20.0, "is_saturated": False, "ignore": False},
            {"x_native_px": 40.0, "y_native_px": 40.0, "is_saturated": False, "ignore": False},
        ]
    ]
    preds_imgs = [
        np.array(
            [
                [20.2, 20.1, 0.9],
                [40.1, 39.8, 0.8],
                [5.0, 5.0, 0.15],  # Low-score FP
            ],
            dtype=np.float32,
        )
    ]
    ap = compute_average_precision(preds_imgs, gt_imgs, tolerance_px=3.0)
    assert math.isclose(ap, 1.0, abs_tol=1e-4)

    best_t, summary = select_best_threshold(
        preds_imgs, gt_imgs, tolerance_px=3.0, thresholds=[0.1, 0.5, 0.85]
    )
    assert math.isclose(best_t, 0.5)
    assert math.isclose(summary["f1"], 1.0)

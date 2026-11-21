"""Headless unit tests for Phase 2 training curves and GT match overlay plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from constellation_detection.visualization import (
    plot_star_detection_matches,
    plot_training_curves,
)


def test_plot_training_curves_saves_png_and_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_training_curves([])

    history = [
        {
            "epoch": 1,
            "train_loss": 1.2,
            "val_loss": 1.1,
            "train_hm_loss": 1.0,
            "val_hm_loss": 0.9,
            "train_off_loss": 0.2,
            "val_off_loss": 0.2,
        },
        {
            "epoch": 2,
            "train_loss": 0.7,
            "val_loss": 0.8,
            "train_hm_loss": 0.6,
            "val_hm_loss": 0.65,
            "train_off_loss": 0.1,
            "val_off_loss": 0.15,
        },
    ]
    out_png = tmp_path / "vis" / "loss_curves.png"
    fig, saved = plot_training_curves(history, output_path=out_png)
    assert fig is not None
    assert saved == out_png.resolve()
    assert out_png.is_file() and out_png.stat().st_size > 1024


def test_plot_star_detection_matches_with_and_without_heatmap(tmp_path: Path) -> None:
    image = np.full((48, 64, 3), 20, dtype=np.uint8)
    gt = [
        {"x_native_px": 15.0, "y_native_px": 20.0, "is_saturated": False, "ignore": False},
        {"x_native_px": 45.0, "y_native_px": 30.0, "is_saturated": False, "ignore": False},
        {"x_native_px": 55.0, "y_native_px": 10.0, "is_saturated": False, "ignore": True},
    ]
    preds = np.array(
        [
            [15.4, 20.1, 0.9],  # TP
            [5.0, 5.0, 0.4],    # FP
        ],
        dtype=np.float32,
    )
    heatmap = np.zeros((48, 64), dtype=np.float32)
    heatmap[20, 15] = 0.9

    out_with_hm = tmp_path / "overlay_hm.png"
    fig1, p1 = plot_star_detection_matches(
        image, preds, gt, tolerance_px=3.0, heatmap=heatmap, output_path=out_with_hm
    )
    assert len(fig1.axes) == 3
    assert p1 == out_with_hm.resolve() and out_with_hm.stat().st_size > 1024

    # Empty predictions without heatmap
    out_empty = tmp_path / "overlay_empty.png"
    fig2, p2 = plot_star_detection_matches(
        image[:, :, 0],
        np.empty((0, 3), dtype=np.float32),
        gt,
        tolerance_px=3.0,
        heatmap=None,
        output_path=out_empty,
    )
    assert len(fig2.axes) == 2
    assert p2 == out_empty.resolve() and out_empty.stat().st_size > 1024

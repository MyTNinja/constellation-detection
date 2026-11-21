"""Multi-detector comparison and leaderboard aggregation for Phase 1."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from constellation_detection.detection.traditional import (
    detect_stars_adaptive_tophat,
    detect_stars_global_threshold,
    detect_stars_multiscale_dog,
)
from constellation_detection.evaluation.diagnostics import compute_star_annulus_snr
from constellation_detection.evaluation.export import (
    build_image_output_stem,
    export_predictions_csv,
)
from constellation_detection.image.preprocessing import to_grayscale

__all__ = [
    "LEADERBOARD_COLUMNS",
    "METRICS_COLUMNS",
    "build_detector_leaderboard",
    "evaluate_image_detectors",
    "resize_to_max_dim",
    "run_all_detectors",
]

METRICS_COLUMNS: tuple[str, ...] = (
    "image_file",
    "resolution",
    "detector",
    "total_stars",
    "top30_snr",
    "latency_ms",
    "saved_csv",
)

LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "detector",
    "total_stars",
    "top30_snr",
    "latency_ms",
)

DetectorFn = Callable[..., dict[str, Any]]

_DEFAULT_DETECTORS: tuple[DetectorFn, ...] = (
    detect_stars_global_threshold,
    detect_stars_adaptive_tophat,
    detect_stars_multiscale_dog,
)


def resize_to_max_dim(image: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """Proportionally downscale an image so its longest side does not exceed ``max_dim``.

    Args:
        image: Input 2D or 3D image array.
        max_dim: Maximum allowed width or height in pixels. Defaults to ``800``.

    Returns:
        Original image copy if ``max(H, W) <= max_dim``, otherwise proportionally
        resized image using ``cv2.INTER_AREA``.

    Raises:
        TypeError: If ``max_dim`` is not an integer.
        ValueError: If ``max_dim <= 0`` or ``image`` is invalid.
    """
    if isinstance(max_dim, (bool, np.bool_)) or not isinstance(
        max_dim, (int, np.integer)
    ):
        raise TypeError(f"max_dim must be an integer, got {type(max_dim).__name__}.")
    if int(max_dim) <= 0:
        raise ValueError(f"max_dim must be positive, got {max_dim}.")

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty 2D or 3D numpy.ndarray.")

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= int(max_dim):
        return image.copy()

    scale = float(max_dim) / float(longest)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def run_all_detectors(
    image: np.ndarray,
    min_dist_px: float = 10.0,
    max_candidates: int = 250,
    detectors: tuple[DetectorFn, ...] = _DEFAULT_DETECTORS,
) -> list[dict[str, Any]]:
    """Run all Phase 1 traditional star detectors against the same image.

    Args:
        image: Input 2D grayscale or 3D BGR ``uint8`` image at working resolution.
        min_dist_px: Minimum NMS separation radius in pixels. Defaults to ``10.0``.
        max_candidates: Maximum retained candidates per detector. Defaults to ``250``.
        detectors: Tuple of detector callables to execute. Defaults to Global Threshold,
            Adaptive Top-Hat, and Multi-Scale DoG.

    Returns:
        List of standardized detector result dictionaries.
    """
    return [
        detector_fn(image, min_dist_px=min_dist_px, max_candidates=max_candidates)
        for detector_fn in detectors
    ]


def evaluate_image_detectors(
    image: np.ndarray,
    image_name: str,
    predictions_dir: str | os.PathLike[str] | None = None,
    max_dim: int = 800,
    min_dist_px: float = 10.0,
    max_candidates: int = 250,
    top_k_snr: int = 30,
) -> tuple[pd.DataFrame, list[dict[str, Any]], np.ndarray]:
    """Run all three Phase 1 detectors on one image and return metrics, results, and working image.

    Computes only ground-truth-free diagnostics (``total_stars``, ``top30_snr``,
    and ``latency_ms``). Does not fabricate precision, recall, F1, or localization
    accuracy without verified ground-truth annotations.

    Args:
        image: Input 2D grayscale or 3D BGR ``uint8`` image.
        image_name: Display filename or identifier for the image.
        predictions_dir: Optional directory where per-detector prediction CSVs are saved.
            Prediction files are written separately from any ground-truth directories.
        max_dim: Working resolution maximum dimension. Defaults to ``800``.
        min_dist_px: NMS suppression radius in pixels. Defaults to ``10.0``.
        max_candidates: Maximum retained star candidates. Defaults to ``250``.
        top_k_snr: Number of top detections used for annulus SNR. Defaults to ``30``.

    Returns:
        Tuple ``(metrics_df, results, working_image)`` where:
            - ``metrics_df`` has columns
              ``image_file, resolution, detector, total_stars, top30_snr, latency_ms, saved_csv``.
            - ``results`` is the list of detector output dictionaries.
            - ``working_image`` is the working-resolution image passed to all detectors.
    """
    working_image = resize_to_max_dim(image, max_dim=max_dim)
    gray_working = to_grayscale(working_image)
    height, width = gray_working.shape
    resolution = f"{width}x{height}"

    results = run_all_detectors(
        working_image,
        min_dist_px=min_dist_px,
        max_candidates=max_candidates,
    )

    image_id = build_image_output_stem(image_name)
    rows: list[dict[str, Any]] = []

    for res in results:
        detector_name = str(res["detector"])
        stars_xyf = res["stars_xyf"]
        top30_snr = compute_star_annulus_snr(gray_working, stars_xyf, top_k=top_k_snr)

        saved_csv = ""
        if predictions_dir is not None:
            csv_name = f"{image_id}_{detector_name}_predictions.csv"
            csv_path = export_predictions_csv(
                stars_xyf,
                Path(predictions_dir) / csv_name,
            )
            saved_csv = str(csv_path)

        rows.append(
            {
                "image_file": image_name,
                "resolution": resolution,
                "detector": detector_name,
                "total_stars": len(stars_xyf),
                "top30_snr": top30_snr,
                "latency_ms": round(float(res["latency_ms"]), 2),
                "saved_csv": saved_csv,
            }
        )

    metrics_df = pd.DataFrame(rows, columns=list(METRICS_COLUMNS))
    return metrics_df, results, working_image


def build_detector_leaderboard(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image detector metrics into a mean leaderboard grouped by detector.

    Args:
        metrics_df: DataFrame containing at least ``detector``, ``total_stars``,
            ``top30_snr``, and ``latency_ms``.

    Returns:
        Aggregated DataFrame with columns ``detector, total_stars, top30_snr, latency_ms``
        containing mean values per detector, rounded to two decimal places.

    Raises:
        ValueError: If required columns are missing from ``metrics_df``.
    """
    required = list(LEADERBOARD_COLUMNS)
    missing = [col for col in required if col not in metrics_df.columns]
    if missing:
        raise ValueError(f"metrics_df is missing required columns: {missing}.")

    if metrics_df.empty:
        return pd.DataFrame(columns=required)

    leaderboard = (
        metrics_df.groupby("detector", sort=False, as_index=False)[
            ["total_stars", "top30_snr", "latency_ms"]
        ]
        .mean()
        .round(2)
    )
    return leaderboard[required]

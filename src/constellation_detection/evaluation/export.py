"""Prediction CSV export utilities (strictly separate from ground-truth labels)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "PREDICTION_CSV_COLUMNS",
    "export_predictions_csv",
    "load_predictions_csv",
]

PREDICTION_CSV_COLUMNS: tuple[str, str, str, str] = (
    "rank",
    "x_px",
    "y_px",
    "flux_score",
)


def export_predictions_csv(
    stars_xyf: np.ndarray,
    output_path: str | os.PathLike[str],
) -> Path:
    """Export detector star predictions to a CSV file with exact Phase 1 headers.

    Prediction files are strictly unverified detector outputs and must be stored
    separately from verified ground-truth annotations.

    Args:
        stars_xyf: Array of shape ``(N, 3)`` containing ``[x_px, y_px, flux_score]``
            ordered brightest-first. Empty ``(0, 3)`` arrays are supported and write
            a header-only CSV file.
        output_path: Destination CSV path. Parent directories are created automatically.

    Returns:
        Resolved ``Path`` to the written CSV file.

    Raises:
        ValueError: If ``stars_xyf`` is non-empty and does not have shape ``(N, 3)``.
    """
    stars_arr = np.asarray(stars_xyf, dtype=np.float32)

    if stars_arr.size == 0:
        df = pd.DataFrame(columns=list(PREDICTION_CSV_COLUMNS))
    else:
        if stars_arr.ndim != 2 or stars_arr.shape[1] != 3:
            raise ValueError(
                f"Expected stars_xyf to have shape (N, 3), got {stars_arr.shape}."
            )
        df = pd.DataFrame(
            {
                "rank": np.arange(1, stars_arr.shape[0] + 1, dtype=np.int64),
                "x_px": stars_arr[:, 0].astype(np.float64),
                "y_px": stars_arr[:, 1].astype(np.float64),
                "flux_score": stars_arr[:, 2].astype(np.float64),
            },
            columns=list(PREDICTION_CSV_COLUMNS),
        )

    dest = Path(output_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def load_predictions_csv(csv_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load an exported detector prediction CSV and validate its schema.

    Args:
        csv_path: Path to a CSV written by :func:`export_predictions_csv`.

    Returns:
        Pandas ``DataFrame`` with columns ``rank, x_px, y_px, flux_score``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If expected prediction columns are missing.
    """
    src = Path(csv_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {src}")

    df = pd.read_csv(src)
    expected = list(PREDICTION_CSV_COLUMNS)
    if list(df.columns) != expected:
        raise ValueError(
            f"Invalid prediction CSV columns: expected {expected}, got {list(df.columns)}."
        )
    return df

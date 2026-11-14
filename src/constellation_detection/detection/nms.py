"""Non-Maximum Suppression (NMS) utilities for star candidate filtering."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["fast_nms_stars"]


def fast_nms_stars(
    stars_list: Sequence[Sequence[float]],
    min_dist_px: float = 10.0,
    max_candidates: int = 250,
) -> np.ndarray:
    """Fast vectorized Non-Maximum Suppression using cKDTree.

    1. Sorts candidates brightest-first and caps pre-filtering pool at ``max_candidates * 2``.
    2. Suppresses any secondary detection within ``min_dist_px`` of a brighter star core,
       retaining at most ``max_candidates`` stars.

    Args:
        stars_list: Sequence of star candidates, where each candidate is a 3-element
            sequence ``(x, y, brightness_score)`` in pixel coordinates.
        min_dist_px: Minimum Euclidean distance in pixels between retained star centers.
            Any dimmer candidate within ``min_dist_px`` (inclusive) of a brighter retained
            star is suppressed. Defaults to ``10.0``.
        max_candidates: Maximum number of retained star candidates to return.
            Defaults to ``250``.

    Returns:
        A 2D NumPy array of shape ``(N, 3)`` and dtype ``np.float32`` containing
        ``[x, y, brightness_score]`` rows ordered from brightest to dimmest, where
        ``0 <= N <= max_candidates``. Returns an empty array of shape ``(0, 3)``
        and dtype ``np.float32`` if ``stars_list`` is empty.
    """
    if not stars_list:
        return np.empty((0, 3), dtype=np.float32)

    stars_sorted = sorted(
        stars_list,
        key=lambda s: s[2],
        reverse=True,
    )[: max_candidates * 2]

    coords = np.array(
        [[s[0], s[1]] for s in stars_sorted],
        dtype=np.float32,
    )
    tree = cKDTree(coords)

    n = len(stars_sorted)
    suppressed = np.zeros(n, dtype=bool)
    kept = []

    for i in range(n):
        if suppressed[i]:
            continue

        kept.append(stars_sorted[i])

        if len(kept) >= max_candidates:
            break

        neighbors = tree.query_ball_point(
            coords[i],
            r=min_dist_px,
        )

        for nb in neighbors:
            if nb > i:
                suppressed[nb] = True

    return np.array(kept, dtype=np.float32)

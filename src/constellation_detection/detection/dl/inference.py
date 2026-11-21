"""Local-CPU-safe tiled inference and sub-pixel peak extraction for DL models."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from constellation_detection.datasets.star_dataset import normalize_starfield_image
from constellation_detection.detection.nms import fast_nms_stars

__all__ = ["detect_stars_dl", "extract_peaks_from_heatmap"]


def extract_peaks_from_heatmap(
    heatmap_prob: np.ndarray,
    offset_map: np.ndarray | None = None,
    threshold: float = 0.35,
    min_dist_px: float = 0.0,
    max_candidates: int = 500,
) -> np.ndarray:
    """Extract sorted ``float32 (N, 3)`` ``[x_px, y_px, score]`` peaks from heatmap."""
    import torch
    import torch.nn.functional as F

    hm = np.asarray(heatmap_prob, dtype=np.float32)
    if hm.ndim != 2 or hm.size == 0:
        raise ValueError(f"Expected non-empty 2D heatmap, got shape {hm.shape}.")

    hm_t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(hm_t, kernel_size=3, stride=1, padding=1)
    is_peak = (hm_t == pooled) & (hm_t >= float(threshold))
    ys, xs = np.where(is_peak.squeeze(0).squeeze(0).numpy())

    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    candidates: list[list[float]] = []
    h, w = hm.shape
    for iy, ix in zip(ys, xs):
        score = float(hm[iy, ix])
        if offset_map is not None:
            dx = float(offset_map[0, iy, ix])
            dy = float(offset_map[1, iy, ix])
        else:
            dx, dy = 0.0, 0.0
        px = min(float(w - 1), max(0.0, float(ix) + dx))
        py = min(float(h - 1), max(0.0, float(iy) + dy))
        candidates.append([px, py, score])

    if min_dist_px > 0.0:
        return fast_nms_stars(
            candidates,
            min_dist_px=float(min_dist_px),
            max_candidates=int(max_candidates),
        )

    candidates.sort(key=lambda item: item[2], reverse=True)
    return np.asarray(candidates[: int(max_candidates)], dtype=np.float32)


def _tile_starts(total_len: int, tile_size: int, step: int) -> list[int]:
    if total_len <= tile_size:
        return [0]
    starts = list(range(0, total_len - tile_size + 1, step))
    last = total_len - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def detect_stars_dl(
    image: np.ndarray,
    model: Any,
    device: Any = "cpu",
    tile_size: int = 512,
    tile_overlap: int = 64,
    threshold: float = 0.35,
    min_dist_px: float = 0.0,
    max_candidates: int = 500,
    in_channels: int = 1,
    detector_name: str = "StarCenterUNetMini",
) -> dict[str, Any]:
    """Run tiled CPU/GPU star detection and return standardized detector dict."""
    import torch

    t0 = time.perf_counter()
    norm_chw = normalize_starfield_image(image, in_channels=in_channels)
    _, height, width = norm_chw.shape

    tile = max(32, int(tile_size))
    overlap = max(0, min(tile // 2, int(tile_overlap)))
    step = max(16, tile - overlap)

    hm_accum = np.zeros((height, width), dtype=np.float32)
    off_accum = np.zeros((2, height, width), dtype=np.float32)
    w_accum = np.zeros((height, width), dtype=np.float32)

    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    with torch.inference_mode():
        for y0 in _tile_starts(height, tile, step):
            y1 = min(height, y0 + tile)
            for x0 in _tile_starts(width, tile, step):
                x1 = min(width, x0 + tile)
                patch = torch.from_numpy(norm_chw[:, y0:y1, x0:x1]).unsqueeze(0).to(dev)
                out = model(patch)
                probs = (
                    torch.sigmoid(out["heatmap_logits"])
                    .squeeze(0)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                offs = out["offset"].squeeze(0).cpu().numpy()

                th, tw = probs.shape
                wy0 = min(overlap // 2, th // 4) if y0 > 0 else 0
                wy1 = th - min(overlap // 2, th // 4) if y1 < height else th
                wx0 = min(overlap // 2, tw // 4) if x0 > 0 else 0
                wx1 = tw - min(overlap // 2, tw // 4) if x1 < width else tw

                hm_accum[y0 + wy0 : y0 + wy1, x0 + wx0 : x0 + wx1] += probs[
                    wy0:wy1, wx0:wx1
                ]
                off_accum[:, y0 + wy0 : y0 + wy1, x0 + wx0 : x0 + wx1] += offs[
                    :, wy0:wy1, wx0:wx1
                ]
                w_accum[y0 + wy0 : y0 + wy1, x0 + wx0 : x0 + wx1] += 1.0

    safe_w = np.maximum(w_accum, 1.0)
    hm_prob = hm_accum / safe_w
    off_mean = off_accum / safe_w[np.newaxis, :, :]

    stars_xyf = extract_peaks_from_heatmap(
        hm_prob,
        offset_map=off_mean,
        threshold=threshold,
        min_dist_px=min_dist_px,
        max_candidates=max_candidates,
    )
    preprocessed = np.clip(np.round(norm_chw[0] * 255.0), 0.0, 255.0).astype(np.uint8)
    mask = (hm_prob >= float(threshold)).astype(np.uint8) * 255
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "detector": detector_name,
        "stars_xyf": stars_xyf,
        "preprocessed": preprocessed,
        "mask": mask,
        "latency_ms": float(latency_ms),
    }

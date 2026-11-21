"""Pure-NumPy normalization, target heatmap rendering, and PyTorch dataset loader."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from constellation_detection.datasets.annotations import load_annotation_record
from constellation_detection.datasets.manifest import load_split_manifest

__all__ = [
    "StarHeatmapDataset",
    "normalize_starfield_image",
    "render_star_targets",
    "rotate_points_90_cw_ccw",
]


def normalize_starfield_image(
    image: np.ndarray,
    in_channels: int = 1,
) -> np.ndarray:
    """Convert raw 2D/3D starfield image to normalized ``float32 (C, H, W)`` in ``[0, 1]``.

    Shared identically across training datasets and inference wrappers to prevent
    train/inference preprocessing skew.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Expected non-empty 2D or 3D image ndarray.")
    if in_channels not in (1, 3):
        raise ValueError(f"in_channels must be 1 or 3, got {in_channels}.")

    if in_channels == 1:
        if image.ndim == 2:
            arr = image.astype(np.float32)
        elif image.shape[2] == 1:
            arr = image[:, :, 0].astype(np.float32)
        elif image.shape[2] == 4:
            arr = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
        else:
            arr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        chw = arr[np.newaxis, :, :]
    else:
        if image.ndim == 2:
            bgr = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        elif image.shape[2] == 1:
            bgr = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            bgr = image[:, :, :3]
        else:
            bgr = image
        rgb = bgr[:, :, ::-1].astype(np.float32)
        chw = np.transpose(rgb, (2, 0, 1))

    lo = float(np.percentile(chw, 1.0))
    hi = float(np.percentile(chw, 99.8))
    if hi <= lo:
        lo = float(np.min(chw))
        hi = float(np.max(chw))
    if hi <= lo:
        return np.zeros_like(chw, dtype=np.float32)
    return np.clip((chw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def rotate_points_90_cw_ccw(
    points_xy: np.ndarray,
    height: int,
    width: int,
    k_ccw: int,
) -> tuple[np.ndarray, int, int]:
    """Transform OpenCV ``(x, y)`` coordinates identically to ``np.rot90(..., k=k_ccw)``."""
    k = int(k_ccw) % 4
    pts = np.asarray(points_xy, dtype=np.float64).copy()
    if pts.size == 0:
        if k in (1, 3):
            return pts, width, height
        return pts, height, width

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    if k == 0:
        return pts, height, width
    if k == 1:
        pts[:, 0] = y
        pts[:, 1] = float(width - 1) - x
        return pts, width, height
    if k == 2:
        pts[:, 0] = float(width - 1) - x
        pts[:, 1] = float(height - 1) - y
        return pts, height, width
    pts[:, 0] = float(height - 1) - y
    pts[:, 1] = x
    return pts, width, height


def render_star_targets(
    stars: list[dict[str, Any]],
    height: int,
    width: int,
    sigma_px: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render Gaussian star-center heatmaps, offsets, positive mask, and ignore mask.

    Returns:
        Tuple ``(heatmap, offset_map, pos_mask, ignore_mask)`` with float32 arrays of
        shapes ``(1, H, W)``, ``(2, H, W)``, ``(1, H, W)``, and ``(1, H, W)``.
    """
    heatmap = np.zeros((1, height, width), dtype=np.float32)
    offset_map = np.zeros((2, height, width), dtype=np.float32)
    pos_mask = np.zeros((1, height, width), dtype=np.float32)
    ignore_mask = np.zeros((1, height, width), dtype=np.float32)

    half = max(2, math.ceil(3.0 * float(sigma_px)))
    ignore_half = max(1, math.ceil(float(sigma_px)))
    inv_2s2 = 1.0 / (2.0 * float(sigma_px) * float(sigma_px))

    for st in stars:
        x = float(st["x_native_px"])
        y = float(st["y_native_px"])
        if x < 0.0 or x > float(width - 1) or y < 0.0 or y > float(height - 1):
            continue
        ix = min(width - 1, max(0, round(x)))
        iy = min(height - 1, max(0, round(y)))

        if bool(st.get("ignore", False)):
            x0, x1 = max(0, ix - ignore_half), min(width, ix + ignore_half + 1)
            y0, y1 = max(0, iy - ignore_half), min(height, iy + ignore_half + 1)
            ignore_mask[0, y0:y1, x0:x1] = 1.0
            continue

        x0, x1 = max(0, ix - half), min(width, ix + half + 1)
        y0, y1 = max(0, iy - half), min(height, iy + half + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        gauss = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) * inv_2s2).astype(np.float32)
        heatmap[0, y0:y1, x0:x1] = np.maximum(heatmap[0, y0:y1, x0:x1], gauss)
        pos_mask[0, iy, ix] = 1.0
        offset_map[0, iy, ix] = float(np.clip(x - float(ix), -0.5, 0.5))
        offset_map[1, iy, ix] = float(np.clip(y - float(iy), -0.5, 0.5))

    # Do not ignore pixels that already hold an explicit verified usable star peak
    ignore_mask[pos_mask > 0.5] = 0.0
    return heatmap, offset_map, pos_mask, ignore_mask


class StarHeatmapDataset:
    """PyTorch-compatible dataset reading verified manifest + star annotations.

    Applies chirality-safe 90-degree rotations and mild photometric jitter in
    training mode (no mirror flips) while returning full normalized images on
    validation and test splits.
    """

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        data_dir: str | os.PathLike[str],
        split: str = "train",
        crop_size: int | None = 256,
        augment: bool = True,
        sigma_px: float = 1.5,
        in_channels: int = 1,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_dir).expanduser().resolve()
        manifest = load_split_manifest(manifest_path)
        self.split = str(split)
        self.samples = [
            dict(s) for s in manifest["samples"] if str(s["split"]) == self.split
        ]
        if not self.samples:
            raise ValueError(f"No samples found for split={self.split!r}.")
        self.crop_size = crop_size
        self.augment = bool(augment and self.split == "train")
        self.sigma_px = float(sigma_px)
        self.in_channels = int(in_channels)
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        sample = self.samples[idx]
        img_path = self.data_root / sample["image_relpath"]
        ann_path = self.data_root / sample["annotation_relpath"]

        raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.size == 0:
            raise ValueError(f"Failed to read image: {img_path}")
        ann = load_annotation_record(ann_path)
        img_norm = normalize_starfield_image(raw, in_channels=self.in_channels)
        _, h, w = img_norm.shape
        stars = list(ann["stars"])

        if self.crop_size is not None:
            cs = min(int(self.crop_size), h, w)
            usable = [st for st in stars if not bool(st.get("ignore", False))]
            if self.split == "train" and usable and self._rng.random() < 0.7:
                anchor = self._rng.choice(usable)
                jitter_x = self._rng.randint(-cs // 4, cs // 4)
                jitter_y = self._rng.randint(-cs // 4, cs // 4)
                cx0 = round(float(anchor["x_native_px"])) - cs // 2 + jitter_x
                cy0 = round(float(anchor["y_native_px"])) - cs // 2 + jitter_y
                x0 = min(max(0, cx0), w - cs)
                y0 = min(max(0, cy0), h - cs)
            elif self.split == "train":
                x0 = self._rng.randint(0, w - cs)
                y0 = self._rng.randint(0, h - cs)
            else:
                x0 = (w - cs) // 2
                y0 = (h - cs) // 2

            img_norm = img_norm[:, y0 : y0 + cs, x0 : x0 + cs]
            cropped_stars: list[dict[str, Any]] = []
            for st in stars:
                sx = float(st["x_native_px"]) - float(x0)
                sy = float(st["y_native_px"]) - float(y0)
                if 0.0 <= sx <= float(cs - 1) and 0.0 <= sy <= float(cs - 1):
                    item = dict(st)
                    item["x_native_px"] = sx
                    item["y_native_px"] = sy
                    cropped_stars.append(item)
            stars = cropped_stars
            h, w = cs, cs

        if self.augment:
            k_rot = self._rng.randint(0, 3)
            if k_rot > 0:
                img_norm = np.ascontiguousarray(np.rot90(img_norm, k=k_rot, axes=(1, 2)))
                if stars:
                    pts = np.array(
                        [[s["x_native_px"], s["y_native_px"]] for s in stars],
                        dtype=np.float64,
                    )
                    rot_pts, h, w = rotate_points_90_cw_ccw(pts, h, w, k_rot)
                    rotated_stars = []
                    for s, (rx, ry) in zip(stars, rot_pts):
                        item = dict(s)
                        item["x_native_px"] = float(rx)
                        item["y_native_px"] = float(ry)
                        rotated_stars.append(item)
                    stars = rotated_stars

            gain = float(self._np_rng.uniform(0.85, 1.15))
            bias = float(self._np_rng.uniform(-0.04, 0.04))
            noise_std = float(self._np_rng.uniform(0.0, 0.015))
            noise = self._np_rng.normal(0.0, noise_std, size=img_norm.shape)
            img_norm = np.clip(img_norm * gain + bias + noise, 0.0, 1.0).astype(
                np.float32
            )

        heatmap, offset_map, pos_mask, ignore_mask = render_star_targets(
            stars, height=h, width=w, sigma_px=self.sigma_px
        )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img_norm)),
            "heatmap": torch.from_numpy(heatmap),
            "offset": torch.from_numpy(offset_map),
            "pos_mask": torch.from_numpy(pos_mask),
            "ignore_mask": torch.from_numpy(ignore_mask),
            "image_id": str(sample["image_id"]),
        }

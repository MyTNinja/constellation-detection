"""Headless Phase 2 visualization helpers for training curves and GT match overlays."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from constellation_detection.evaluation.matching import match_detections_to_ground_truth

__all__ = [
    "plot_star_detection_matches",
    "plot_training_curves",
]

_UNMATCHABLE_COST = 1e6


def _to_rgb_display(image: np.ndarray) -> np.ndarray:
    """Convert 2D grayscale or 3D BGR/BGRA array into a Matplotlib-ready RGB image."""
    arr = np.asarray(image)
    if arr.ndim not in (2, 3) or arr.size == 0:
        raise ValueError(f"Expected non-empty 2D or 3D image array, got {arr.shape}.")

    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
    elif arr.shape[2] == 1:
        rgb = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 3:
        rgb = arr[:, :, ::-1]
    elif arr.shape[2] == 4:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported channel count in image shape {arr.shape}.")

    if rgb.dtype == np.uint8:
        return rgb

    rgb_f = rgb.astype(np.float32)
    lo = float(np.percentile(rgb_f, 1.0))
    hi = float(np.percentile(rgb_f, 99.8))
    if hi <= lo:
        lo = float(np.min(rgb_f))
        hi = float(np.max(rgb_f))
    if hi <= lo:
        return np.zeros_like(rgb_f, dtype=np.float32)
    return np.clip((rgb_f - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _classify_matches_for_plot(
    pred_xyf: np.ndarray,
    gt_stars: Sequence[dict[str, Any]],
    tolerance_px: float,
) -> dict[str, np.ndarray]:
    """Partition predictions and ground-truth stars into TP, FP, FN, and ignored sets."""
    preds = np.asarray(pred_xyf, dtype=np.float64)
    if preds.size == 0:
        preds = np.empty((0, 3), dtype=np.float64)

    usable = [s for s in gt_stars if not bool(s.get("ignore", False))]
    ignored = [s for s in gt_stars if bool(s.get("ignore", False))]

    gt_xy = (
        np.array(
            [[float(s["x_native_px"]), float(s["y_native_px"])] for s in usable],
            dtype=np.float64,
        )
        if usable
        else np.empty((0, 2), dtype=np.float64)
    )
    ign_xy = (
        np.array(
            [[float(s["x_native_px"]), float(s["y_native_px"])] for s in ignored],
            dtype=np.float64,
        )
        if ignored
        else np.empty((0, 2), dtype=np.float64)
    )

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    if preds.shape[0] > 0 and gt_xy.shape[0] > 0:
        dist_mat = cdist(preds[:, :2], gt_xy)
        cost_mat = np.where(dist_mat <= tolerance_px, dist_mat, _UNMATCHABLE_COST)
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        for r_idx, c_idx in zip(row_ind, col_ind):
            if float(dist_mat[r_idx, c_idx]) <= tolerance_px:
                matched_preds.add(int(r_idx))
                matched_gts.add(int(c_idx))

    unmatched_pred_idx = [i for i in range(preds.shape[0]) if i not in matched_preds]
    if unmatched_pred_idx and ign_xy.shape[0] > 0:
        d_ign = np.min(cdist(preds[unmatched_pred_idx, :2], ign_xy), axis=1)
        fp_idx = [idx for idx, dist in zip(unmatched_pred_idx, d_ign) if dist > tolerance_px]
    else:
        fp_idx = unmatched_pred_idx

    fn_idx = [j for j in range(gt_xy.shape[0]) if j not in matched_gts]
    tp_idx = sorted(matched_preds)

    return {
        "tp_xy": preds[tp_idx, :2] if tp_idx else np.empty((0, 2), dtype=np.float64),
        "fp_xy": preds[fp_idx, :2] if fp_idx else np.empty((0, 2), dtype=np.float64),
        "fn_xy": gt_xy[fn_idx] if fn_idx else np.empty((0, 2), dtype=np.float64),
        "ign_xy": ign_xy,
    }


def plot_training_curves(
    history: Sequence[dict[str, Any]],
    output_path: str | os.PathLike[str] | None = None,
    title: str = "Phase 2 Training Curves",
) -> tuple[Figure, Path | None]:
    """Render headless 1x2 loss curves (total and component losses) across epochs."""
    if not history:
        raise ValueError("Cannot plot training curves from an empty history sequence.")

    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(history)]

    fig = Figure(figsize=(12, 4.5), dpi=120)
    FigureCanvasAgg(fig)
    ax_total, ax_comp = fig.subplots(1, 2)

    has_total = False
    if all("train_loss" in row for row in history):
        ax_total.plot(
            epochs,
            [float(row["train_loss"]) for row in history],
            marker="o",
            markersize=3,
            label="train_loss",
        )
        has_total = True
    if all("val_loss" in row for row in history):
        ax_total.plot(
            epochs,
            [float(row["val_loss"]) for row in history],
            marker="s",
            markersize=3,
            label="val_loss",
        )
        has_total = True

    ax_total.set_title("Total Objective Loss")
    ax_total.set_xlabel("Epoch")
    ax_total.set_ylabel("Loss")
    ax_total.grid(True, alpha=0.3)
    if has_total:
        ax_total.legend()

    component_keys = (
        ("train_hm_loss", "train_hm_loss"),
        ("val_hm_loss", "val_hm_loss"),
        ("train_off_loss", "train_off_loss"),
        ("val_off_loss", "val_off_loss"),
    )
    has_comp = False
    for key, label in component_keys:
        if all(key in row for row in history):
            ax_comp.plot(
                epochs,
                [float(row[key]) for row in history],
                marker="o",
                markersize=2.5,
                label=label,
            )
            has_comp = True

    ax_comp.set_title("Heatmap & Sub-Pixel Offset Losses")
    ax_comp.set_xlabel("Epoch")
    ax_comp.set_ylabel("Loss")
    ax_comp.grid(True, alpha=0.3)
    if has_comp:
        ax_comp.legend()

    fig.suptitle(title)
    fig.tight_layout()

    saved_path: Path | None = None
    if output_path is not None:
        dest = Path(output_path).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=120, bbox_inches="tight")
        saved_path = dest

    return fig, saved_path


def plot_star_detection_matches(
    image: np.ndarray,
    pred_xyf: np.ndarray,
    gt_stars: Sequence[dict[str, Any]],
    tolerance_px: float = 3.0,
    heatmap: np.ndarray | None = None,
    title: str = "Star Detection Matches",
    output_path: str | os.PathLike[str] | None = None,
) -> tuple[Figure, Path | None]:
    """Render TP/FP/FN star match overlay against verified GT annotations."""
    metrics = match_detections_to_ground_truth(
        pred_xyf, gt_stars, tolerance_px=tolerance_px
    )
    groups = _classify_matches_for_plot(pred_xyf, gt_stars, tolerance_px=tolerance_px)
    rgb = _to_rgb_display(image)

    n_cols = 3 if heatmap is not None else 2
    fig = Figure(figsize=(6 * n_cols, 5.5), dpi=120)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, n_cols)

    axes[0].imshow(rgb)
    axes[0].set_title(f"Input ({rgb.shape[1]}x{rgb.shape[0]})")
    axes[0].axis("off")

    overlay_idx = 1
    if heatmap is not None:
        hm = np.asarray(heatmap, dtype=np.float32)
        if hm.ndim == 3 and hm.shape[0] == 1:
            hm = hm[0]
        if hm.ndim != 2:
            raise ValueError(f"Expected 2D heatmap or (1, H, W), got {hm.shape}.")
        axes[1].imshow(hm, cmap="magma", vmin=0.0, vmax=1.0)
        axes[1].set_title("Predicted Center Heatmap")
        axes[1].axis("off")
        overlay_idx = 2

    ax_ov = axes[overlay_idx]
    ax_ov.imshow(rgb)

    if groups["ign_xy"].shape[0] > 0:
        ax_ov.scatter(
            groups["ign_xy"][:, 0],
            groups["ign_xy"][:, 1],
            s=10,
            c="lightgray",
            alpha=0.45,
            label=f"Ignore GT ({groups['ign_xy'].shape[0]})",
        )
    if groups["fn_xy"].shape[0] > 0:
        ax_ov.scatter(
            groups["fn_xy"][:, 0],
            groups["fn_xy"][:, 1],
            s=36,
            facecolors="none",
            edgecolors="gold",
            marker="s",
            linewidths=1.0,
            label=f"FN ({metrics['fn']})",
        )
    if groups["fp_xy"].shape[0] > 0:
        ax_ov.scatter(
            groups["fp_xy"][:, 0],
            groups["fp_xy"][:, 1],
            s=36,
            c="red",
            marker="x",
            linewidths=1.2,
            label=f"FP ({metrics['fp']})",
        )
    if groups["tp_xy"].shape[0] > 0:
        ax_ov.scatter(
            groups["tp_xy"][:, 0],
            groups["tp_xy"][:, 1],
            s=42,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.2,
            label=f"TP ({metrics['tp']})",
        )

    rmse_str = f"{metrics['rmse_px']:.2f} px" if metrics["rmse_px"] is not None else "n/a"
    ax_ov.set_title(
        f"Matches (tau={tolerance_px:.1f} px)\n"
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} RMSE={rmse_str}"
    )
    ax_ov.axis("off")
    handles, _ = ax_ov.get_legend_handles_labels()
    if handles:
        ax_ov.legend(loc="upper right", fontsize=7, framealpha=0.75)

    fig.suptitle(title)
    fig.tight_layout()

    saved_path: Path | None = None
    if output_path is not None:
        dest = Path(output_path).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=120, bbox_inches="tight")
        saved_path = dest

    return fig, saved_path

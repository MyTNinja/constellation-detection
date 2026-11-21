"""Hungarian one-to-one star matching, PR-AUC, and validation threshold selection."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from constellation_detection.datasets.annotations import load_annotation_record
from constellation_detection.datasets.manifest import load_split_manifest
from constellation_detection.evaluation.compare import (
    resize_to_max_dim,
    run_all_detectors,
)
from constellation_detection.image.io import load_image

__all__ = [
    "compute_average_precision",
    "evaluate_phase1_baselines_on_manifest",
    "match_detections_to_ground_truth",
    "scale_working_stars_to_native",
    "select_best_threshold",
]

_UNMATCHABLE_COST = 1e6


def scale_working_stars_to_native(
    stars_xyf: np.ndarray,
    working_shape: tuple[int, int],
    native_shape: tuple[int, int],
) -> np.ndarray:
    """Scale detector coordinates from working resolution back to native pixels.

    Follows ``docs/context/ARCHITECTURE.md`` §21.3:
    ``x_native = x_work * (W_native / W_work)``, ``y_native = y_work * (H_native / H_work)``.
    """
    arr = np.asarray(stars_xyf, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    work_h, work_w = int(working_shape[0]), int(working_shape[1])
    nat_h, nat_w = int(native_shape[0]), int(native_shape[1])
    if work_h <= 0 or work_w <= 0 or nat_h <= 0 or nat_w <= 0:
        raise ValueError("Image shapes must be strictly positive.")
    out = arr.copy()
    out[:, 0] = out[:, 0] * float(nat_w / float(work_w))
    out[:, 1] = out[:, 1] * float(nat_h / float(work_h))
    return out


def match_detections_to_ground_truth(
    pred_xyf: np.ndarray,
    gt_stars: Sequence[dict[str, Any]],
    tolerance_px: float = 3.0,
) -> dict[str, Any]:
    """Perform optimal one-to-one bipartite matching within ``tolerance_px``.

    Predictions matched only to ``ignore=True`` ground-truth stars are masked
    from false-positive counts so uncalibrated borderline faint stars neither
    reward recall nor unfairly penalize precision.
    """
    if tolerance_px <= 0.0:
        raise ValueError(f"tolerance_px must be positive, got {tolerance_px}.")

    preds = np.asarray(pred_xyf, dtype=np.float64)
    if preds.size == 0:
        preds = np.empty((0, 3), dtype=np.float64)

    usable = [s for s in gt_stars if not bool(s.get("ignore", False))]
    ignored = [s for s in gt_stars if bool(s.get("ignore", False))]

    sat_total = sum(1 for s in usable if bool(s.get("is_saturated", False)))
    n_pred = int(preds.shape[0])
    n_gt = len(usable)

    if n_pred == 0 or n_gt == 0:
        fp = n_pred
        if n_pred > 0 and ignored:
            ign_xy = np.array(
                [[float(s["x_native_px"]), float(s["y_native_px"])] for s in ignored],
                dtype=np.float64,
            )
            d_ign = np.min(cdist(preds[:, :2], ign_xy), axis=1)
            fp = int(np.sum(d_ign > tolerance_px))
        return {
            "tp": 0,
            "fp": fp,
            "fn": n_gt,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "rmse_px": None,
            "median_ae_px": None,
            "saturated_total": sat_total,
            "saturated_tp": 0,
            "saturated_recall": 0.0 if sat_total > 0 else None,
            "saturated_rmse_px": None,
            "unsaturated_rmse_px": None,
            "matched_errors_px": [],
            "saturated_errors_px": [],
            "unsaturated_errors_px": [],
        }

    gt_xy = np.array(
        [[float(s["x_native_px"]), float(s["y_native_px"])] for s in usable],
        dtype=np.float64,
    )
    dist_mat = cdist(preds[:, :2], gt_xy)
    cost_mat = np.where(dist_mat <= tolerance_px, dist_mat, _UNMATCHABLE_COST)
    row_ind, col_ind = linear_sum_assignment(cost_mat)

    matched_preds: set[int] = set()
    matched_errors: list[float] = []
    sat_errors: list[float] = []
    unsat_errors: list[float] = []

    for r_idx, c_idx in zip(row_ind, col_ind):
        err = float(dist_mat[r_idx, c_idx])
        if err <= tolerance_px:
            matched_preds.add(int(r_idx))
            matched_errors.append(err)
            if bool(usable[int(c_idx)].get("is_saturated", False)):
                sat_errors.append(err)
            else:
                unsat_errors.append(err)

    unmatched_indices = [i for i in range(n_pred) if i not in matched_preds]
    fp = len(unmatched_indices)
    if unmatched_indices and ignored:
        ign_xy = np.array(
            [[float(s["x_native_px"]), float(s["y_native_px"])] for s in ignored],
            dtype=np.float64,
        )
        d_ign = np.min(cdist(preds[unmatched_indices, :2], ign_xy), axis=1)
        fp = int(np.sum(d_ign > tolerance_px))

    tp = len(matched_errors)
    fn = n_gt - tp
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )

    def _rmse(errs: list[float]) -> float | None:
        return math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else None

    med_ae = float(np.median(matched_errors)) if matched_errors else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rmse_px": _rmse(matched_errors),
        "median_ae_px": med_ae,
        "saturated_total": sat_total,
        "saturated_tp": len(sat_errors),
        "saturated_recall": (
            float(len(sat_errors) / sat_total) if sat_total > 0 else None
        ),
        "saturated_rmse_px": _rmse(sat_errors),
        "unsaturated_rmse_px": _rmse(unsat_errors),
        "matched_errors_px": matched_errors,
        "saturated_errors_px": sat_errors,
        "unsaturated_errors_px": unsat_errors,
    }


def compute_average_precision(
    all_pred_xyf: Sequence[np.ndarray],
    all_gt_stars: Sequence[Sequence[dict[str, Any]]],
    tolerance_px: float = 3.0,
) -> float:
    """Compute detection PR-AUC (Average Precision) across multiple images."""
    if len(all_pred_xyf) != len(all_gt_stars):
        raise ValueError("Prediction and ground-truth image lists must align.")

    total_gt = 0
    usable_gt_by_img: list[np.ndarray] = []
    ignored_gt_by_img: list[np.ndarray] = []
    matched_gt_flags: list[np.ndarray] = []
    detections: list[tuple[float, int, float, float]] = []

    for img_idx, (preds, gt_list) in enumerate(zip(all_pred_xyf, all_gt_stars)):
        usable = [s for s in gt_list if not bool(s.get("ignore", False))]
        ignored = [s for s in gt_list if bool(s.get("ignore", False))]
        total_gt += len(usable)
        usable_arr = (
            np.array(
                [[float(s["x_native_px"]), float(s["y_native_px"])] for s in usable],
                dtype=np.float64,
            )
            if usable
            else np.empty((0, 2), dtype=np.float64)
        )
        ign_arr = (
            np.array(
                [[float(s["x_native_px"]), float(s["y_native_px"])] for s in ignored],
                dtype=np.float64,
            )
            if ignored
            else np.empty((0, 2), dtype=np.float64)
        )
        usable_gt_by_img.append(usable_arr)
        ignored_gt_by_img.append(ign_arr)
        matched_gt_flags.append(np.zeros(len(usable), dtype=bool))

        p_arr = np.asarray(preds, dtype=np.float64)
        if p_arr.size > 0:
            for row in p_arr:
                detections.append((float(row[2]), img_idx, float(row[0]), float(row[1])))

    if total_gt == 0 or not detections:
        return 0.0

    detections.sort(key=lambda item: item[0], reverse=True)
    tp_list: list[float] = []
    fp_list: list[float] = []

    for _score, img_idx, px, py in detections:
        gt_coords = usable_gt_by_img[img_idx]
        flags = matched_gt_flags[img_idx]
        matched = False
        if gt_coords.shape[0] > 0:
            dists = np.hypot(gt_coords[:, 0] - px, gt_coords[:, 1] - py)
            avail = np.where(~flags)[0]
            if avail.size > 0:
                best_sub = int(np.argmin(dists[avail]))
                best_idx = int(avail[best_sub])
                if float(dists[best_idx]) <= tolerance_px:
                    flags[best_idx] = True
                    matched = True

        if matched:
            tp_list.append(1.0)
            fp_list.append(0.0)
        else:
            ign_coords = ignored_gt_by_img[img_idx]
            is_ign = False
            if ign_coords.shape[0] > 0:
                d_ign = float(np.min(np.hypot(ign_coords[:, 0] - px, ign_coords[:, 1] - py)))
                is_ign = d_ign <= tolerance_px
            if not is_ign:
                tp_list.append(0.0)
                fp_list.append(1.0)

    if not tp_list:
        return 0.0

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return round(min(1.0, max(0.0, ap)), 6)


def select_best_threshold(
    val_preds: Sequence[np.ndarray],
    val_gt_stars: Sequence[Sequence[dict[str, Any]]],
    tolerance_px: float = 3.0,
    thresholds: Sequence[float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Select confidence threshold strictly on the validation split by max aggregate F1."""
    if thresholds is None:
        thresh_grid = [round(0.05 + 0.05 * i, 2) for i in range(18)]
    else:
        thresh_grid = [float(t) for t in thresholds]

    best_thresh = thresh_grid[0]
    best_summary: dict[str, Any] = {"f1": -1.0}

    for thresh in thresh_grid:
        tp_sum = 0
        fp_sum = 0
        fn_sum = 0
        for pred, gt in zip(val_preds, val_gt_stars):
            arr = np.asarray(pred, dtype=np.float32)
            filtered = arr[arr[:, 2] >= thresh] if arr.size > 0 else arr
            res = match_detections_to_ground_truth(
                filtered, gt, tolerance_px=tolerance_px
            )
            tp_sum += res["tp"]
            fp_sum += res["fp"]
            fn_sum += res["fn"]

        prec = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
        rec = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = (
            float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        )
        if f1 > float(best_summary["f1"]):
            best_thresh = float(thresh)
            best_summary = {
                "threshold": best_thresh,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "tp": tp_sum,
                "fp": fp_sum,
                "fn": fn_sum,
            }
    return best_thresh, best_summary


def evaluate_phase1_baselines_on_manifest(
    manifest_path: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    split: str = "test",
    tolerance_px: float = 3.0,
    max_dim: int = 800,
) -> list[dict[str, Any]]:
    """Evaluate frozen Phase 1 detectors on verified GT split in native pixel space."""
    data_root = Path(data_dir).expanduser().resolve()
    manifest = load_split_manifest(manifest_path)
    split_samples = [s for s in manifest["samples"] if str(s["split"]) == split]
    if not split_samples:
        raise ValueError(f"No samples found in manifest for split={split!r}.")

    preds_by_det: dict[str, list[np.ndarray]] = {}
    latencies_by_det: dict[str, list[float]] = {}
    gt_list: list[list[dict[str, Any]]] = []

    for sample in split_samples:
        img = load_image(data_root / sample["image_relpath"])
        ann = load_annotation_record(data_root / sample["annotation_relpath"])
        gt_list.append(ann["stars"])

        native_h, native_w = img.shape[:2]
        working = resize_to_max_dim(img, max_dim=max_dim)
        work_h, work_w = working.shape[:2]

        for res in run_all_detectors(working):
            det_name = str(res["detector"])
            scaled = scale_working_stars_to_native(
                res["stars_xyf"],
                working_shape=(work_h, work_w),
                native_shape=(native_h, native_w),
            )
            preds_by_det.setdefault(det_name, []).append(scaled)
            latencies_by_det.setdefault(det_name, []).append(float(res["latency_ms"]))

    rows: list[dict[str, Any]] = []
    n_imgs = len(split_samples)
    for det_name, pred_list in preds_by_det.items():
        tp_tot = fp_tot = fn_tot = sat_tot = sat_tp = 0
        all_errs: list[float] = []
        for pred, gt in zip(pred_list, gt_list):
            m = match_detections_to_ground_truth(pred, gt, tolerance_px=tolerance_px)
            tp_tot += m["tp"]
            fp_tot += m["fp"]
            fn_tot += m["fn"]
            sat_tot += m["saturated_total"]
            sat_tp += m["saturated_tp"]
            all_errs.extend(m["matched_errors_px"])

        prec = float(tp_tot / (tp_tot + fp_tot)) if (tp_tot + fp_tot) > 0 else 0.0
        rec = float(tp_tot / (tp_tot + fn_tot)) if (tp_tot + fn_tot) > 0 else 0.0
        f1 = (
            float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        )
        ap = compute_average_precision(pred_list, gt_list, tolerance_px=tolerance_px)
        rmse = (
            math.sqrt(sum(e * e for e in all_errs) / len(all_errs))
            if all_errs
            else None
        )
        rows.append(
            {
                "detector": det_name,
                "split": split,
                "num_images": n_imgs,
                "tolerance_px": tolerance_px,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "pr_auc": round(ap, 4),
                "fppi": round(fp_tot / float(n_imgs), 2),
                "rmse_px": round(rmse, 4) if rmse is not None else None,
                "saturated_recall": (
                    round(sat_tp / float(sat_tot), 4) if sat_tot > 0 else None
                ),
                "mean_latency_ms": round(
                    sum(latencies_by_det[det_name]) / float(n_imgs), 2
                ),
            }
        )
    return rows

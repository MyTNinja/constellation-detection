"""Programmatic dataset integrity and astrometric alignment auditor."""

from __future__ import annotations

import hashlib
import math
import os
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from constellation_detection.datasets.annotations import load_annotation_record
from constellation_detection.datasets.manifest import load_split_manifest

__all__ = ["audit_pilot_dataset"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash64(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for val in diff.flatten():
        bits = (bits << 1) | int(bool(val))
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def audit_pilot_dataset(
    manifest_path: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    max_alignment_median_px: float = 2.0,
) -> dict[str, Any]:
    """Run read-only programmatic audit implementing ``dataset-audit`` SKILL.md.

    Verifies file readability, image dimensions, SHA-256 checksums, annotation schema
    & bounds, train/val/test disjointness (group + SHA-256 + near-duplicate dHash),
    and empirical sub-pixel alignment between projected bright catalogue stars and
    local peak maxima in the rendered images.
    """
    data_root = Path(data_dir).expanduser().resolve()
    manifest = load_split_manifest(manifest_path)

    issues: list[str] = []
    warnings_list: list[str] = []
    dhashes_by_split: dict[str, list[tuple[str, int]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    alignment_offsets_px: list[float] = []
    stats: dict[str, int] = {
        "total_images": 0,
        "total_stars": 0,
        "usable_stars": 0,
        "ignored_stars": 0,
        "saturated_stars": 0,
        "blended_stars": 0,
    }

    for sample in manifest["samples"]:
        img_rel = Path(sample["image_relpath"])
        ann_rel = Path(sample["annotation_relpath"])
        if img_rel.is_absolute() or ann_rel.is_absolute():
            issues.append(f"Absolute path in manifest sample {sample['image_id']}")
            continue

        img_path = (data_root / img_rel).resolve()
        ann_path = (data_root / ann_rel).resolve()
        try:
            img_path.relative_to(data_root)
            ann_path.relative_to(data_root)
        except ValueError:
            issues.append(f"Path escapes data_dir for {sample['image_id']}")
            continue

        if not img_path.is_file() or not ann_path.is_file():
            issues.append(f"Missing image or annotation for {sample['image_id']}")
            continue

        if sample.get("image_sha256") and _sha256(img_path) != sample["image_sha256"]:
            issues.append(f"Image SHA-256 mismatch for {sample['image_id']}")
        if (
            sample.get("annotation_sha256")
            and _sha256(ann_path) != sample["annotation_sha256"]
        ):
            issues.append(f"Annotation SHA-256 mismatch for {sample['image_id']}")

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            issues.append(f"OpenCV failed to decode image {sample['image_id']}")
            continue

        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        try:
            ann = load_annotation_record(ann_path)
        except (TypeError, ValueError, OSError) as exc:
            issues.append(f"Invalid annotation for {sample['image_id']}: {exc}")
            continue

        if ann["native_width"] != w or ann["native_height"] != h:
            issues.append(
                f"Dimension mismatch for {sample['image_id']}: "
                f"image={w}x{h} vs ann={ann['native_width']}x{ann['native_height']}"
            )
            continue

        stats["total_images"] += 1
        dhashes_by_split[sample["split"]].append((sample["image_id"], _dhash64(gray)))

        stars = ann["stars"]
        stats["total_stars"] += len(stars)
        for st in stars:
            if st["ignore"]:
                stats["ignored_stars"] += 1
            else:
                stats["usable_stars"] += 1
            if st["is_saturated"]:
                stats["saturated_stars"] += 1
            if st["is_blended"]:
                stats["blended_stars"] += 1

        # Empirical star-to-peak WCS alignment test on up to 12 bright unsaturated stars
        bright_candidates = [
            st
            for st in stars
            if not st["ignore"]
            and not st["is_saturated"]
            and 6.0 <= float(st["x_native_px"]) <= float(w - 7)
            and 6.0 <= float(st["y_native_px"]) <= float(h - 7)
        ]
        bright_candidates.sort(key=lambda s: float(s.get("mag", 99.0)))
        for st in bright_candidates[:12]:
            gx = float(st["x_native_px"])
            gy = float(st["y_native_px"])
            cx, cy = round(gx), round(gy)
            patch = gray[cy - 5 : cy + 6, cx - 5 : cx + 6].astype(np.float64)
            weights = np.clip(patch - float(np.median(patch)), 0.0, None)
            total_w = float(np.sum(weights))
            if total_w > 0.0:
                yy, xx = np.mgrid[cy - 5 : cy + 6, cx - 5 : cx + 6]
                peak_x = float(np.sum(xx * weights) / total_w)
                peak_y = float(np.sum(yy * weights) / total_w)
            else:
                py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
                peak_x = float(cx - 5 + px)
                peak_y = float(cy - 5 + py)
            alignment_offsets_px.append(math.hypot(peak_x - gx, peak_y - gy))

    # Near-duplicate perceptual hash leakage scan across splits
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for s1, s2 in pairs:
        for id1, h1 in dhashes_by_split[s1]:
            for id2, h2 in dhashes_by_split[s2]:
                if _hamming(h1, h2) <= 2:
                    warnings_list.append(
                        f"Near-identical dHash between {s1}:{id1} and {s2}:{id2}"
                    )

    if stats["usable_stars"] == 0:
        issues.append("Zero non-ignored ground-truth stars across dataset.")

    median_align_px = (
        round(float(statistics.median(alignment_offsets_px)), 4)
        if alignment_offsets_px
        else None
    )
    mean_align_px = (
        round(float(statistics.mean(alignment_offsets_px)), 4)
        if alignment_offsets_px
        else None
    )
    if median_align_px is None or median_align_px > max_alignment_median_px:
        issues.append(
            f"Empirical bright-star WCS alignment failed: median_px={median_align_px} "
            f"(limit={max_alignment_median_px})."
        )

    verdict = "FAIL" if issues else ("CONDITIONAL" if warnings_list else "PASS")
    return {
        "verdict": verdict,
        "dataset_version": manifest["dataset_version"],
        "split_counts": manifest["split_counts"],
        "stats": stats,
        "alignment": {
            "num_bright_probes": len(alignment_offsets_px),
            "median_residual_px": median_align_px,
            "mean_residual_px": mean_align_px,
            "threshold_px": max_alignment_median_px,
        },
        "issues": issues,
        "warnings": warnings_list,
    }

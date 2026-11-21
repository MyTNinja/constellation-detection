"""Leakage-resistant group-level dataset split manifest builder and validator."""

from __future__ import annotations

import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0.0"

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "build_group_split_manifest",
    "load_split_manifest",
    "verify_split_disjointness",
]


def _allocate_group_counts(
    n_groups: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int, int]:
    if n_groups < 3:
        raise ValueError(
            f"Need at least 3 distinct groups for train/val/test, got {n_groups}."
        )
    n_val = max(1, round(n_groups * val_ratio))
    n_test = max(1, round(n_groups * (1.0 - train_ratio - val_ratio)))
    while n_val + n_test >= n_groups:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    n_train = n_groups - n_val - n_test
    return n_train, n_val, n_test


def build_group_split_manifest(
    entries: list[dict[str, Any]],
    dataset_version: str,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, Any]:
    """Partition dataset entries cleanly at the ``group_id`` level.

    Args:
        entries: List of sample metadata dicts, each containing ``image_id``,
            ``group_id``, ``image_relpath``, ``annotation_relpath``,
            ``image_sha256``, and ``annotation_sha256``.
        dataset_version: Semantic dataset release identifier (e.g., ``"ds01-v1.0"``).
        seed: Explicit RNG seed controlling group assignment order.
        train_ratio: Target fraction of groups assigned to ``train``.
        val_ratio: Target fraction of groups assigned to ``val``.

    Returns:
        Manifest dictionary ready to save to ``data/splits/``.
    """
    if not entries:
        raise ValueError("Cannot build a split manifest from zero entries.")
    if train_ratio <= 0.0 or val_ratio <= 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError("train_ratio and val_ratio must sum strictly below 1.0.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        gid = str(entry["group_id"])
        grouped.setdefault(gid, []).append(dict(entry))

    group_ids = sorted(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_train, n_val, _n_test = _allocate_group_counts(
        len(group_ids), train_ratio=train_ratio, val_ratio=val_ratio
    )
    train_groups = set(group_ids[:n_train])
    val_groups = set(group_ids[n_train : n_train + n_val])

    manifest_samples: list[dict[str, Any]] = []
    counts = {"train": 0, "val": 0, "test": 0}
    for gid in sorted(grouped.keys()):
        if gid in train_groups:
            split_name = "train"
        elif gid in val_groups:
            split_name = "val"
        else:
            split_name = "test"
        for item in grouped[gid]:
            rec = dict(item)
            rec["split"] = split_name
            manifest_samples.append(rec)
            counts[split_name] += 1

    manifest_samples.sort(key=lambda r: (str(r["split"]), str(r["image_id"])))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": str(dataset_version),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(seed),
        "split_ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": round(1.0 - float(train_ratio) - float(val_ratio), 4),
        },
        "split_counts": counts,
        "total_images": len(manifest_samples),
        "total_groups": len(group_ids),
        "samples": manifest_samples,
    }
    verify_split_disjointness(manifest)
    return manifest


def save_split_manifest(
    manifest: dict[str, Any],
    dest_path: str | os.PathLike[str],
) -> Path:
    """Validate split disjointness and write manifest JSON atomically."""
    verify_split_disjointness(manifest)
    dest = Path(dest_path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(encoded)
        os.replace(tmp_path, dest)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return dest


def load_split_manifest(manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a dataset split manifest JSON."""
    src = Path(manifest_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Split manifest not found: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Split manifest must be a JSON object: {src}")
    verify_split_disjointness(data)
    return data


def verify_split_disjointness(manifest: dict[str, Any]) -> None:
    """Assert zero leakage across ``train``, ``val``, and ``test`` splits.

    Checks uniqueness of ``image_id``, disjointness of ``group_id`` across splits,
    and disjointness of ``image_sha256`` across splits.
    """
    samples = manifest.get("samples", [])
    if not samples:
        raise ValueError("Split manifest contains no samples.")

    seen_ids: set[str] = set()
    groups_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    hashes_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for sample in samples:
        split_name = str(sample.get("split", ""))
        if split_name not in groups_by_split:
            raise ValueError(f"Invalid split name: {split_name!r}")
        img_id = str(sample["image_id"])
        if img_id in seen_ids:
            raise ValueError(f"Duplicate image_id across manifest: {img_id!r}")
        seen_ids.add(img_id)

        groups_by_split[split_name].add(str(sample["group_id"]))
        if sample.get("image_sha256"):
            hashes_by_split[split_name].add(str(sample["image_sha256"]))

    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for s1, s2 in split_pairs:
        shared_groups = groups_by_split[s1] & groups_by_split[s2]
        if shared_groups:
            raise ValueError(
                f"Group leakage between {s1} and {s2}: {sorted(shared_groups)}"
            )
        shared_hashes = hashes_by_split[s1] & hashes_by_split[s2]
        if shared_hashes:
            raise ValueError(
                f"Exact duplicate image hash leakage between {s1} and {s2}: "
                f"{sorted(shared_hashes)}"
            )

"""Verified ground-truth annotation schema validation and atomic I/O.

Enforces ``ADR-008`` (strict separation from ``outputs/predictions/``) and
``ADR-011`` (native OpenCV pixel coordinate contract).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

ANNOTATION_SCHEMA_VERSION = "1.0.0"

__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "load_annotation_record",
    "save_annotation_record",
    "validate_annotation_record",
]

_FORBIDDEN_PROVENANCES = {
    "phase1_predictions",
    "outputs_predictions",
    "global_threshold",
    "CV2_AdaptiveTopHat",
    "CV3_MultiScaleDoG",
}


def validate_annotation_record(record: dict[str, Any]) -> None:
    """Validate a single image star annotation dictionary against schema ``1.0.0``.

    Raises:
        ValueError: If required keys, coordinate bounds, or GT provenance fail.
    """
    required_top = {
        "schema_version",
        "image_id",
        "image_relpath",
        "coordinate_frame",
        "native_width",
        "native_height",
        "ground_truth_provenance",
        "wcs_metadata",
        "limiting_magnitude",
        "ignore_magnitude_upper",
        "domain",
        "stars",
    }
    missing = required_top - set(record.keys())
    if missing:
        raise ValueError(f"Annotation record missing keys: {sorted(missing)}")

    if str(record["schema_version"]) != ANNOTATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported annotation schema_version: {record['schema_version']!r}"
        )

    if str(record["coordinate_frame"]) != "native_opencv_top_left_px":
        raise ValueError(
            f"Invalid coordinate_frame {record['coordinate_frame']!r}; "
            "expected 'native_opencv_top_left_px'."
        )

    provenance = str(record["ground_truth_provenance"]).strip()
    if not provenance or provenance in _FORBIDDEN_PROVENANCES:
        raise ValueError(
            f"Forbidden or empty ground_truth_provenance: {provenance!r}"
        )

    width = int(record["native_width"])
    height = int(record["native_height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid native image size: {width}x{height}.")

    stars = record["stars"]
    if not isinstance(stars, list):
        raise TypeError("Annotation 'stars' field must be a list.")

    for idx, star in enumerate(stars):
        if not isinstance(star, dict):
            raise TypeError(f"Star entry {idx} must be an object.")
        x = float(star["x_native_px"])
        y = float(star["y_native_px"])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Star {idx} has non-finite coordinates: ({x}, {y}).")
        if x < 0.0 or x > float(width - 1) or y < 0.0 or y > float(height - 1):
            raise ValueError(
                f"Star {idx} out of native OpenCV bounds [0, {width - 1}]x[0, {height - 1}]: "
                f"({x}, {y})."
            )
        for flag in ("is_saturated", "is_blended", "ignore"):
            if not isinstance(star.get(flag), bool):
                raise TypeError(f"Star {idx} missing boolean flag {flag!r}.")


def save_annotation_record(
    record: dict[str, Any],
    dest_path: str | os.PathLike[str],
) -> Path:
    """Validate and atomically write a verified ground-truth annotation JSON."""
    validate_annotation_record(record)
    dest = Path(dest_path).expanduser().resolve()
    if "predictions" in dest.parts:
        raise ValueError(
            f"Refusing to write ground-truth annotation under predictions directory: {dest}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")

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


def load_annotation_record(src_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate an annotation JSON file from disk."""
    src = Path(src_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Annotation file not found: {src}")
    record = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise TypeError(f"Annotation file must contain a JSON object: {src}")
    validate_annotation_record(record)
    return record

"""Dataset preparation, astrometric WCS projection, manifests, and audit utilities."""

from __future__ import annotations

from constellation_detection.datasets.annotations import (
    ANNOTATION_SCHEMA_VERSION,
    load_annotation_record,
    save_annotation_record,
    validate_annotation_record,
)
from constellation_detection.datasets.audit import audit_pilot_dataset
from constellation_detection.datasets.fits_wcs import (
    FitsImage,
    TanSipWcs,
    read_fits_image,
    render_fits_to_uint8,
)
from constellation_detection.datasets.manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_group_split_manifest,
    load_split_manifest,
    save_split_manifest,
    verify_split_disjointness,
)
from constellation_detection.datasets.star_dataset import (
    StarHeatmapDataset,
    normalize_starfield_image,
    render_star_targets,
    rotate_points_90_cw_ccw,
)

__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "FitsImage",
    "StarHeatmapDataset",
    "TanSipWcs",
    "audit_pilot_dataset",
    "build_group_split_manifest",
    "load_annotation_record",
    "load_split_manifest",
    "normalize_starfield_image",
    "read_fits_image",
    "render_fits_to_uint8",
    "render_star_targets",
    "rotate_points_90_cw_ccw",
    "save_annotation_record",
    "save_split_manifest",
    "validate_annotation_record",
    "verify_split_disjointness",
]

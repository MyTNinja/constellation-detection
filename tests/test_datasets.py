"""Offline unit tests for FITS WCS projection, annotation validation, splits, and audit."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from constellation_detection.datasets import (
    ANNOTATION_SCHEMA_VERSION,
    TanSipWcs,
    audit_pilot_dataset,
    build_group_split_manifest,
    read_fits_image,
    render_fits_to_uint8,
    save_annotation_record,
    save_split_manifest,
    validate_annotation_record,
    verify_split_disjointness,
)


def _write_synthetic_fits(
    path: Path,
    image: np.ndarray,
    extra_cards: list[str] | None = None,
) -> None:
    h, w = image.shape
    cards = [
        f"{'SIMPLE':<8}= {'T':>20}",
        f"{'BITPIX':<8}= {-32:>20d}",
        f"{'NAXIS':<8}= {2:>20d}",
        f"{'NAXIS1':<8}= {w:>20d}",
        f"{'NAXIS2':<8}= {h:>20d}",
    ]
    if extra_cards:
        cards.extend(extra_cards)
    cards.append("END")

    header_bytes = b"".join(f"{c:<80}".encode("ascii")[:80] for c in cards)
    pad = (-len(header_bytes)) % 2880
    header_block = header_bytes + (b" " * pad)

    payload = image.astype(">f4").tobytes()
    payload_pad = (-len(payload)) % 2880
    path.write_bytes(header_block + payload + (b"\x00" * payload_pad))


def test_fits_reader_and_tan_sip_roundtrip(tmp_path: Path) -> None:
    fits_path = tmp_path / "synthetic.fits"
    arr = np.linspace(0.0, 100.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    cards = [
        f"{'CTYPE1':<8}= 'RA---TAN-SIP'",
        f"{'CTYPE2':<8}= 'DEC--TAN-SIP'",
        f"{'CRPIX1':<8}= {32.5:>20.6f}",
        f"{'CRPIX2':<8}= {32.5:>20.6f}",
        f"{'CRVAL1':<8}= {180.0:>20.6f}",
        f"{'CRVAL2':<8}= {45.0:>20.6f}",
        f"{'CD1_1':<8}= {-0.00028:>20.10f}",
        f"{'CD1_2':<8}= {0.0:>20.10f}",
        f"{'CD2_1':<8}= {0.0:>20.10f}",
        f"{'CD2_2':<8}= {0.00028:>20.10f}",
        f"{'A_ORDER':<8}= {2:>20d}",
        f"{'B_ORDER':<8}= {2:>20d}",
        f"{'A_2_0':<8}= {1.5e-5:>20.10e}",
        f"{'B_0_2':<8}= {-1.2e-5:>20.10e}",
    ]
    _write_synthetic_fits(fits_path, arr, cards)

    fits_img = read_fits_image(fits_path)
    assert fits_img.width == 64
    assert fits_img.height == 64
    assert np.allclose(fits_img.data, arr, atol=1e-4)

    wcs = TanSipWcs.from_header(fits_img.header)
    assert math.isclose(wcs.pixel_scale_arcsec, 1.008, abs_tol=1e-3)

    x_in = np.array([5.25, 31.5, 58.75])
    y_in = np.array([10.5, 31.5, 49.25])
    ra, dec = wcs.pixel_to_world(x_in, y_in)
    x_out, y_out = wcs.world_to_pixel(ra, dec)
    assert np.allclose(x_out, x_in, atol=1e-8)
    assert np.allclose(y_out, y_in, atol=1e-8)


def test_tpv_projection_roundtrip() -> None:
    tpv_header = {
        "CTYPE1": "RA---TPV",
        "CTYPE2": "DEC--TPV",
        "CRPIX1": 1500.0,
        "CRPIX2": -900.0,
        "CRVAL1": 192.38,
        "CRVAL2": 27.52,
        "CD1_1": -0.000281,
        "CD1_2": 8.45e-06,
        "CD2_1": -8.45e-06,
        "CD2_2": -0.000281,
        "PV1_0": -1.5e-05,
        "PV1_1": 1.00005,
        "PV1_2": 4.5e-05,
        "PV1_4": 5.4e-04,
        "PV1_12": -1.2e-03,
        "PV2_0": 4.9e-05,
        "PV2_1": 0.99996,
        "PV2_2": -1.6e-04,
        "PV2_5": 4.4e-04,
        "PV2_14": 8.8e-04,
    }
    wcs = TanSipWcs.from_header(tpv_header)
    assert wcs.projection == "TPV"
    x_in = np.array([10.0, 250.25, 600.5])
    y_in = np.array([15.0, 310.75, 580.25])
    ra, dec = wcs.pixel_to_world(x_in, y_in)
    x_out, y_out = wcs.world_to_pixel(ra, dec)
    assert np.allclose(x_out, x_in, atol=1e-8)
    assert np.allclose(y_out, y_in, atol=1e-8)


def test_annotation_validation_rejects_bad_bounds_and_predictions(tmp_path: Path) -> None:
    valid = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "image_id": "test_01",
        "image_relpath": "raw/test_01.png",
        "coordinate_frame": "native_opencv_top_left_px",
        "native_width": 64,
        "native_height": 64,
        "ground_truth_provenance": "ztf_fits_wcs_gaia_dr3_projected",
        "wcs_metadata": {"solver": "tan"},
        "limiting_magnitude": 16.0,
        "ignore_magnitude_upper": 17.0,
        "domain": "synthetic_fixture",
        "stars": [
            {
                "x_native_px": 12.5,
                "y_native_px": 24.5,
                "is_saturated": False,
                "is_blended": False,
                "ignore": False,
            }
        ],
    }
    validate_annotation_record(valid)

    bad_prov = dict(valid, ground_truth_provenance="phase1_predictions")
    with pytest.raises(ValueError):
        validate_annotation_record(bad_prov)

    bad_coord = dict(
        valid,
        stars=[
            {
                "x_native_px": 64.0,
                "y_native_px": 10.0,
                "is_saturated": False,
                "is_blended": False,
                "ignore": False,
            }
        ],
    )
    with pytest.raises(ValueError):
        validate_annotation_record(bad_coord)

    with pytest.raises(ValueError):
        save_annotation_record(valid, tmp_path / "predictions" / "bad.json")


def test_split_disjointness_and_dataset_audit(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    ann_dir = data_dir / "annotations"
    raw_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    wcs = TanSipWcs(
        crpix1=32.5,
        crpix2=32.5,
        crval1=120.0,
        crval2=30.0,
        cd=np.array([[-0.0003, 0.0], [0.0, 0.0003]]),
        cd_inv=np.linalg.inv(np.array([[-0.0003, 0.0], [0.0, 0.0003]])),
    )

    entries = []
    rng = np.random.default_rng(7)
    for idx in range(6):
        img = np.clip(10.0 + rng.normal(0.0, 1.5, (64, 64)), 0.0, 50.0)
        star_x, star_y = 20.0 + idx * 3.0, 18.0 + idx * 2.0
        yy, xx = np.mgrid[0:64, 0:64]
        img = img + 400.0 * np.exp(-((xx - star_x) ** 2 + (yy - star_y) ** 2) / 3.0)
        img_u8 = render_fits_to_uint8(img)

        img_path = raw_dir / f"f{idx}.png"
        cv2.imwrite(str(img_path), img_u8)

        ra, dec = wcs.pixel_to_world(star_x, star_y)
        px_x, px_y = wcs.world_to_pixel(float(ra), float(dec))

        ann_path = ann_dir / f"f{idx}.json"
        save_annotation_record(
            {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "image_id": f"f{idx}",
                "image_relpath": f"raw/f{idx}.png",
                "coordinate_frame": "native_opencv_top_left_px",
                "native_width": 64,
                "native_height": 64,
                "ground_truth_provenance": "synthetic_wcs_fixture",
                "wcs_metadata": {"solver": "synthetic_tan"},
                "limiting_magnitude": 15.0,
                "ignore_magnitude_upper": 16.0,
                "domain": "synthetic_fixture",
                "stars": [
                    {
                        "x_native_px": float(px_x),
                        "y_native_px": float(px_y),
                        "mag": 10.0,
                        "is_saturated": False,
                        "is_blended": False,
                        "ignore": False,
                    }
                ],
            },
            ann_path,
        )
        entries.append(
            {
                "image_id": f"f{idx}",
                "group_id": f"grp_{idx}",
                "image_relpath": f"raw/f{idx}.png",
                "annotation_relpath": f"annotations/f{idx}.json",
                "image_sha256": hashlib.sha256(img_path.read_bytes()).hexdigest(),
                "annotation_sha256": hashlib.sha256(ann_path.read_bytes()).hexdigest(),
            }
        )

    manifest = build_group_split_manifest(entries, dataset_version="test-v1", seed=42)
    manifest_path = data_dir / "splits" / "manifest.json"
    save_split_manifest(manifest, manifest_path)

    report = audit_pilot_dataset(manifest_path, data_dir)
    assert report["verdict"] in {"PASS", "CONDITIONAL"}
    assert report["issues"] == []
    assert report["alignment"]["median_residual_px"] <= 1.0

    # Force group leakage and confirm rejection
    leaky = dict(manifest)
    leaky_samples = [dict(s) for s in manifest["samples"]]
    leaky_samples[0]["split"] = "train"
    leaky_samples[1]["split"] = "test"
    leaky_samples[1]["group_id"] = leaky_samples[0]["group_id"]
    leaky["samples"] = leaky_samples
    with pytest.raises(ValueError):
        verify_split_disjointness(leaky)

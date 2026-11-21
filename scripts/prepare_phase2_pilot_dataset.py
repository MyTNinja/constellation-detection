"""Prepare the audited UD-003 Phase 2 pilot dataset (ZTF FITS WCS + Gaia DR3).

Downloads well-separated public ZTF science cutouts from NASA/IPAC IRSA,
renders lossless 8-bit PNG images under ``data/raw/ztf_pilot/``, queries
ESA Gaia DR3 ICRS sources per field, projects stars into native OpenCV pixel
coordinates via each cutout's FITS WCS, saves verified annotations under
``data/annotations/ztf_pilot/``, writes the group-partitioned split manifest,
and executes the programmatic dataset audit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from constellation_detection.config import ProjectPaths, to_relative_posix
from constellation_detection.datasets import (
    ANNOTATION_SCHEMA_VERSION,
    TanSipWcs,
    audit_pilot_dataset,
    build_group_split_manifest,
    read_fits_image,
    render_fits_to_uint8,
    save_annotation_record,
    save_split_manifest,
)

ZTF_IBE_SEARCH_URL = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
ZTF_IBE_DATA_URL = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"
GAIA_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

# 40 northern/equatorial sky targets (Dec >= -20 deg), mutual separation >= 6 deg,
# spanning low, medium, and high Galactic latitudes for diverse star densities.
PILOT_SKY_TARGETS: tuple[tuple[str, float, float], ...] = (
    ("f01_ngp", 192.85, 27.13),
    ("f02_uma", 165.93, 61.75),
    ("f03_boo", 213.92, 19.18),
    ("f04_vir", 187.71, 12.39),
    ("f05_leo", 152.09, 11.97),
    ("f06_cnc", 130.10, 19.67),
    ("f07_her", 250.42, 36.46),
    ("f08_lyr", 283.39, 38.78),
    ("f09_cyg", 305.56, 40.26),
    ("f10_aql", 297.70, 8.87),
    ("f11_peg", 345.00, 15.21),
    ("f12_and", 10.68, 41.27),
    ("f13_cas", 14.18, 60.24),
    ("f14_per", 51.08, 49.86),
    ("f15_tau", 56.75, 24.12),
    ("f16_ori", 83.82, -5.39),
    ("f17_gem", 106.03, 20.57),
    ("f18_aur", 79.17, 45.99),
    ("f19_mon", 100.24, 9.89),
    ("f20_hya", 141.90, -8.66),
    ("f21_crb", 233.67, 26.71),
    ("f22_dra", 268.38, 56.87),
    ("f23_cep", 336.00, 58.00),
    ("f24_lac", 337.00, 46.00),
    ("f25_tri", 23.46, 30.66),
    ("f26_ari", 32.00, 20.00),
    ("f27_com", 190.00, 20.00),
    ("f28_cvn", 202.47, 47.19),
    ("f29_lyn", 125.00, 52.00),
    ("f30_cam", 75.00, 65.00),
    ("f31_vul", 300.00, 24.00),
    ("f32_sge", 295.00, 18.00),
    ("f33_del", 310.00, 14.00),
    ("f34_oph", 262.00, 10.00),
    ("f35_ser", 236.00, 6.00),
    ("f36_lib", 229.00, -14.00),
    ("f37_aqr", 331.00, -10.00),
    ("f38_cet", 25.00, -12.00),
    ("f39_psc", 12.00, 10.00),
    ("f40_sex", 155.00, -2.00),
)


def _build_ssl_context() -> ssl.SSLContext:
    """Create a strict TLS context populated with certifi and OS trust stores."""
    ctx = ssl.create_default_context()
    try:
        from pip._vendor import certifi

        ca_file = certifi.where()
    except (ImportError, AttributeError):
        ca_file = None
    if ca_file:
        ctx.load_verify_locations(cafile=ca_file)
    enum_certs = getattr(ssl, "enum_certificates", None)
    if enum_certs is not None:
        for store in ("ROOT", "CA"):
            try:
                certs = enum_certs(store)
            except OSError:
                continue
            for cert_bytes, enc, _ in certs:
                if enc != "x509_asn":
                    continue
                pem = (
                    cert_bytes
                    if isinstance(cert_bytes, str)
                    else ssl.DER_cert_to_PEM_cert(cert_bytes)
                )
                try:
                    ctx.load_verify_locations(cadata=pem)
                except ssl.SSLError:
                    continue
    return ctx


_SSL_CTX = _build_ssl_context()


def _http_get_bytes(url: str, timeout: float = 45.0, max_retries: int = 3) -> bytes:
    last_err: BaseException | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "constellation-detection/0.1 (Python urllib)"},
            )
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt + 1 >= max_retries:
                break
    raise RuntimeError(f"GET failed for {url}: {last_err}")


def _http_post_text(
    url: str,
    params: dict[str, str],
    timeout: float = 45.0,
    max_retries: int = 3,
) -> str:
    body = urllib.parse.urlencode(params).encode("utf-8")
    last_err: BaseException | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={"User-Agent": "constellation-detection/0.1 (Python urllib)"},
            )
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt + 1 >= max_retries:
                break
    raise RuntimeError(f"POST failed for {url}: {last_err}")


def _search_ztf_exposure(ra_deg: float, dec_deg: float) -> dict[str, str] | None:
    params = urllib.parse.urlencode(
        {"POS": f"{ra_deg},{dec_deg}", "SIZE": "0.01", "ct": "csv"}
    )
    csv_text = _http_get_bytes(f"{ZTF_IBE_SEARCH_URL}?{params}").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    good_rows = []
    for row in rows:
        if row.get("filtercode") not in {"zg", "zr"}:
            continue
        infobits = row.get("infobits", "0")
        if infobits and int(float(infobits)) != 0:
            continue
        try:
            maglim = float(row.get("maglimit", "0") or 0.0)
        except ValueError:
            continue
        if maglim >= 18.5:
            good_rows.append((maglim, row))
    if not good_rows:
        return None
    good_rows.sort(key=lambda item: item[0], reverse=True)
    return good_rows[0][1]


def _build_ztf_cutout_url(
    meta: dict[str, str],
    ra_deg: float,
    dec_deg: float,
    cutout_px: int,
) -> str:
    filefracday = meta["filefracday"].strip()
    year = filefracday[:4]
    mmdd = filefracday[4:8]
    fracday = filefracday[8:]
    padded_field = f"{int(meta['field']):06d}"
    padded_ccd = f"{int(meta['ccdid']):02d}"
    filtercode = meta["filtercode"].strip()
    qid = int(meta["qid"])
    fname = (
        f"ztf_{filefracday}_{padded_field}_{filtercode}"
        f"_c{padded_ccd}_o_q{qid}_sciimg.fits"
    )
    return (
        f"{ZTF_IBE_DATA_URL}/{year}/{mmdd}/{fracday}/{fname}"
        f"?center={ra_deg},{dec_deg}&size={cutout_px}pix"
    )


def _atomic_write_bytes(dest: Path, data: bytes) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
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
            fh.write(data)
        os.replace(tmp_path, dest)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return dest


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _query_gaia_dr3_cone(
    ra_center: float,
    dec_center: float,
    radius_deg: float,
    max_mag: float,
) -> list[dict[str, Any]]:
    query = (
        "SELECT TOP 5000 source_id, ra, dec, phot_g_mean_mag "
        "FROM gaiadr3.gaia_source "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra_center:.6f}, {dec_center:.6f}, {radius_deg:.6f})) = 1 "
        f"AND phot_g_mean_mag <= {max_mag:.2f} "
        "ORDER BY phot_g_mean_mag ASC"
    )
    csv_text = _http_post_text(
        GAIA_TAP_SYNC_URL,
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
    )
    stars: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(csv_text)):
        try:
            stars.append(
                {
                    "catalog_id": f"GAIA_DR3_{row['source_id']}",
                    "ra_deg": float(row["ra"]),
                    "dec_deg": float(row["dec"]),
                    "mag": float(row["phot_g_mean_mag"]),
                }
            )
        except (KeyError, ValueError):
            continue
    return stars


def _compute_cone(wcs: TanSipWcs, width: int, height: int) -> tuple[float, float, float]:
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    ra_c, dec_c = wcs.pixel_to_world(cx, cy)
    corners_x = np.array([0.0, float(width - 1), 0.0, float(width - 1)])
    corners_y = np.array([0.0, 0.0, float(height - 1), float(height - 1)])
    ra_k, dec_k = wcs.pixel_to_world(corners_x, corners_y)

    r0 = math.radians(float(ra_c))
    d0 = math.radians(float(dec_c))
    rk = np.deg2rad(ra_k)
    dk = np.deg2rad(dec_k)
    cos_ang = np.sin(d0) * np.sin(dk) + np.cos(d0) * np.cos(dk) * np.cos(rk - r0)
    max_ang_deg = float(np.rad2deg(np.arccos(np.clip(cos_ang, -1.0, 1.0))).max())
    return float(ra_c), float(dec_c), max_ang_deg * 1.15


def prepare_pilot_dataset(
    data_dir: Path,
    num_fields: int = 24,
    cutout_px: int = 1024,
    dataset_version: str = "ds01-v1.0",
    seed: int = 42,
) -> dict[str, Any]:
    """Download ZTF FITS cutouts + Gaia DR3 annotations and run dataset audit."""
    fits_dir = data_dir / "raw" / "ztf_fits_cache"
    png_dir = data_dir / "raw" / "ztf_pilot"
    ann_dir = data_dir / "annotations" / "ztf_pilot"
    manifest_path = data_dir / "splits" / "dataset_manifest_v1.0.json"

    for d in (fits_dir, png_dir, ann_dir, manifest_path.parent):
        d.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []

    for label, ra_deg, dec_deg in PILOT_SKY_TARGETS:
        if len(entries) >= num_fields:
            break

        meta = _search_ztf_exposure(ra_deg, dec_deg)
        if meta is None:
            print(f"[skip] No clean zg/zr exposure found for {label}", flush=True)
            continue

        image_id = (
            f"ztf_{label}_f{int(meta['field']):06d}"
            f"_{meta['filtercode']}_c{int(meta['ccdid']):02d}_q{meta['qid']}"
        )
        fits_path = fits_dir / f"{image_id}.fits"
        png_path = png_dir / f"{image_id}.png"
        ann_path = ann_dir / f"{image_id}.json"

        if not fits_path.is_file():
            cutout_url = _build_ztf_cutout_url(meta, ra_deg, dec_deg, cutout_px)
            print(f"[fetch] {image_id} from IRSA...", flush=True)
            try:
                fits_bytes = _http_get_bytes(cutout_url, timeout=60.0)
            except RuntimeError as exc:
                print(f"[warn] Download failed for {image_id}: {exc}", flush=True)
                continue
            if fits_bytes.startswith(b"\x1f\x8b"):
                fits_bytes = gzip.decompress(fits_bytes)
            if not fits_bytes.startswith(b"SIMPLE"):
                print(f"[warn] Non-FITS response for {image_id}", flush=True)
                continue
            _atomic_write_bytes(fits_path, fits_bytes)

        fits_img = read_fits_image(fits_path)
        if fits_img.width < 512 or fits_img.height < 512:
            print(
                f"[skip] Edge-clipped cutout {image_id}: "
                f"{fits_img.width}x{fits_img.height}",
                flush=True,
            )
            continue

        if not png_path.is_file():
            rendered = render_fits_to_uint8(fits_img.data)
            ok, buf = cv2.imencode(".png", rendered)
            if not ok:
                raise RuntimeError(f"Failed PNG encoding for {image_id}")
            _atomic_write_bytes(png_path, buf.tobytes())

        wcs = TanSipWcs.from_header(fits_img.header)
        ra_c, dec_c, cone_deg = _compute_cone(wcs, fits_img.width, fits_img.height)
        raw_maglim = float(
            fits_img.header.get("MAGLIM")
            or meta.get("maglimit")
            or 19.5
        )
        usable_mag_limit = round(min(18.0, raw_maglim - 1.5), 2)
        ignore_upper = round(usable_mag_limit + 1.0, 2)

        gaia_sources = _query_gaia_dr3_cone(
            ra_c, dec_c, cone_deg, max_mag=ignore_upper
        )
        if not gaia_sources:
            print(f"[skip] Zero Gaia sources returned for {image_id}", flush=True)
            continue
        if len(gaia_sources) >= 5000:
            max_returned_mag = max(float(s["mag"]) for s in gaia_sources)
            ignore_upper = min(ignore_upper, round(max_returned_mag, 2))
            usable_mag_limit = min(usable_mag_limit, round(ignore_upper - 1.0, 2))

        ra_arr = np.array([s["ra_deg"] for s in gaia_sources], dtype=np.float64)
        dec_arr = np.array([s["dec_deg"] for s in gaia_sources], dtype=np.float64)
        px_x, px_y = wcs.world_to_pixel(ra_arr, dec_arr)

        saturate_level = float(fits_img.header.get("SATURATE") or 55000.0)
        in_bounds_stars: list[dict[str, Any]] = []
        for idx, s in enumerate(gaia_sources):
            x = float(px_x[idx])
            y = float(px_y[idx])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if not (
                2.0 <= x <= float(fits_img.width - 3)
                and 2.0 <= y <= float(fits_img.height - 3)
            ):
                continue
            ix, iy = round(x), round(y)
            win = fits_img.data[
                max(0, iy - 2) : min(fits_img.height, iy + 3),
                max(0, ix - 2) : min(fits_img.width, ix + 3),
            ]
            finite_win = win[np.isfinite(win)]
            if finite_win.size == 0:
                continue
            peak_adu = float(np.max(finite_win))
            is_sat = bool(peak_adu >= 0.9 * saturate_level or s["mag"] <= 11.0)
            in_bounds_stars.append(
                {
                    "x_native_px": round(x, 4),
                    "y_native_px": round(y, 4),
                    "sigma_pos_px": 0.25,
                    "catalog_id": s["catalog_id"],
                    "ra_deg": round(s["ra_deg"], 6),
                    "dec_deg": round(s["dec_deg"], 6),
                    "mag": round(s["mag"], 3),
                    "mag_band": "G",
                    "is_saturated": is_sat,
                    "is_blended": False,
                    "ignore": bool(s["mag"] > usable_mag_limit),
                }
            )

        if len(in_bounds_stars) >= 2:
            coords = np.array(
                [[s["x_native_px"], s["y_native_px"]] for s in in_bounds_stars],
                dtype=np.float64,
            )
            tree = cKDTree(coords)
            for idx, pairs in enumerate(tree.query_ball_tree(tree, r=3.0)):
                if len(pairs) > 1:
                    in_bounds_stars[idx]["is_blended"] = True

        if not any(not s["ignore"] for s in in_bounds_stars):
            print(f"[skip] No usable bright Gaia stars in {image_id}", flush=True)
            continue

        ann_record = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "image_id": image_id,
            "image_relpath": to_relative_posix(png_path, data_dir),
            "coordinate_frame": "native_opencv_top_left_px",
            "native_width": fits_img.width,
            "native_height": fits_img.height,
            "ground_truth_provenance": "ztf_fits_wcs_gaia_dr3_projected",
            "wcs_metadata": {
                "solver": "ztf_irsa_fits_tan_sip",
                "pixel_scale_arcsec": round(wcs.pixel_scale_arcsec, 4),
                "crval1_deg": wcs.crval1,
                "crval2_deg": wcs.crval2,
            },
            "limiting_magnitude": usable_mag_limit,
            "ignore_magnitude_upper": ignore_upper,
            "domain": "survey_fits_ztf",
            "stars": in_bounds_stars,
        }
        save_annotation_record(ann_record, ann_path)

        entries.append(
            {
                "image_id": image_id,
                "group_id": f"ztf_field_{int(meta['field']):06d}_{label}",
                "image_relpath": to_relative_posix(png_path, data_dir),
                "annotation_relpath": to_relative_posix(ann_path, data_dir),
                "image_sha256": _sha256_file(png_path),
                "annotation_sha256": _sha256_file(ann_path),
                "domain": "survey_fits_ztf",
            }
        )
        print(
            f"[ok {len(entries)}/{num_fields}] {image_id}: "
            f"{len(in_bounds_stars)} Gaia stars projected",
            flush=True,
        )

    if len(entries) < 3:
        raise RuntimeError(
            f"Only {len(entries)} ZTF fields succeeded; need >= 3 for train/val/test."
        )

    manifest = build_group_split_manifest(
        entries,
        dataset_version=dataset_version,
        seed=seed,
    )
    save_split_manifest(manifest, manifest_path)
    return audit_pilot_dataset(manifest_path, data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and audit the UD-003 ZTF + Gaia DR3 Phase 2 pilot dataset."
    )
    parser.add_argument("--num-fields", type=int, default=24)
    parser.add_argument("--cutout-px", type=int, default=1024)
    parser.add_argument("--dataset-version", type=str, default="ds01-v1.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = ProjectPaths.from_paths()
    report = prepare_pilot_dataset(
        data_dir=paths.data_dir,
        num_fields=args.num_fields,
        cutout_px=args.cutout_px,
        dataset_version=args.dataset_version,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    if report["verdict"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

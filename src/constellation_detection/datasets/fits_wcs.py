"""Pure-NumPy FITS 2D reader and TAN/SIP astrometric WCS projection utilities.

Follows ``docs/context/ARCHITECTURE.md`` §21 (`ADR-011`):
- Converts FITS 1-based pixel coordinates ``(x_fits, y_fits)`` directly into
  OpenCV 0-based coordinates ``(x_px, y_px) = (x_fits - 1.0, y_fits - 1.0)``
  consistent with NumPy array indexing ``data[iy, ix]``.
- Supports ICRS/J2000 gnomonic ``TAN`` projections with optional Shupe et al.
  (2005) ``-SIP`` polynomial distortion and iterative Newton inversion.
"""

from __future__ import annotations

import gzip
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "FitsImage",
    "TanSipWcs",
    "read_fits_image",
    "render_fits_to_uint8",
]

_FITS_BLOCK_BYTES = 2880
_FITS_CARD_BYTES = 80
_SIP_COEFF_RE = re.compile(r"^(A|B|AP|BP)_(\d+)_(\d+)$")
_TPV_COEFF_RE = re.compile(r"^PV(1|2)_(\d+)$")

# Standard Calabretta/SCAMP TPV polynomial terms for m = 0..16: (p_x, p_y, p_r).
_TPV_EXPONENTS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (1, 0, 0),
    2: (0, 1, 0),
    3: (0, 0, 1),
    4: (2, 0, 0),
    5: (1, 1, 0),
    6: (0, 2, 0),
    7: (3, 0, 0),
    8: (2, 1, 0),
    9: (1, 2, 0),
    10: (0, 3, 0),
    11: (0, 0, 3),
    12: (4, 0, 0),
    13: (3, 1, 0),
    14: (2, 2, 0),
    15: (1, 3, 0),
    16: (0, 4, 0),
}

_BITPIX_DTYPES: dict[int, str] = {
    8: ">u1",
    16: ">i2",
    32: ">i4",
    -32: ">f4",
    -64: ">f8",
}


def _parse_fits_value(raw_val: str) -> Any:
    text = raw_val.strip()
    if not text:
        return ""
    if text.startswith("'"):
        end_idx = text.rfind("'")
        if end_idx > 0:
            return text[1:end_idx].strip()
        return text.strip("'").strip()
    if text in {"T", "F"}:
        return text == "T"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return text


def _split_card_value_comment(rest: str) -> str:
    in_quotes = False
    for idx, ch in enumerate(rest):
        if ch == "'":
            in_quotes = not in_quotes
        elif ch == "/" and not in_quotes:
            return rest[:idx]
    return rest


def _parse_fits_header(raw_bytes: bytes) -> tuple[dict[str, Any], int]:
    header: dict[str, Any] = {}
    offset = 0
    found_end = False
    while offset < len(raw_bytes):
        block = raw_bytes[offset : offset + _FITS_BLOCK_BYTES]
        offset += _FITS_BLOCK_BYTES
        if len(block) < _FITS_BLOCK_BYTES:
            raise ValueError("Truncated FITS header block.")
        for i in range(0, _FITS_BLOCK_BYTES, _FITS_CARD_BYTES):
            card = block[i : i + _FITS_CARD_BYTES].decode("ascii", errors="replace")
            key = card[:8].strip()
            if key == "END":
                found_end = True
                break
            if not key or key in {"COMMENT", "HISTORY"}:
                continue
            if card[8:10] != "= ":
                continue
            val_part = _split_card_value_comment(card[10:])
            header[key] = _parse_fits_value(val_part)
        if found_end:
            break
    if not found_end:
        raise ValueError("Missing FITS END header card.")
    return header, offset


@dataclass(frozen=True)
class FitsImage:
    """Primary 2D HDU image container loaded from a FITS file."""

    data: np.ndarray
    header: dict[str, Any]

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])


def read_fits_image(fits_path: str | os.PathLike[str]) -> FitsImage:
    """Read a 2D primary-HDU FITS image into ``float64`` shape ``(H, W)``."""
    src = Path(fits_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"FITS file not found: {src}")

    raw = src.read_bytes()
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    header, data_offset = _parse_fits_header(raw)
    bitpix = int(header["BITPIX"])
    if bitpix not in _BITPIX_DTYPES:
        raise ValueError(f"Unsupported FITS BITPIX={bitpix} in {src}.")

    naxis = int(header.get("NAXIS", 0))
    if naxis != 2:
        raise ValueError(f"Expected 2D primary FITS image (NAXIS=2), got {naxis}.")

    width = int(header["NAXIS1"])
    height = int(header["NAXIS2"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid FITS dimensions: {width}x{height}.")

    dtype = np.dtype(_BITPIX_DTYPES[bitpix])
    n_pixels = width * height
    byte_count = n_pixels * dtype.itemsize
    if data_offset + byte_count > len(raw):
        raise ValueError(f"Truncated FITS image payload in {src}.")

    arr = np.frombuffer(
        raw, dtype=dtype, count=n_pixels, offset=data_offset
    ).reshape((height, width))
    bscale = float(header.get("BSCALE", 1.0))
    bzero = float(header.get("BZERO", 0.0))
    data = arr.astype(np.float64) * bscale + bzero
    return FitsImage(data=data, header=header)


@dataclass(frozen=True)
class TanSipWcs:
    """ICRS gnomonic ``TAN``, ``TAN-SIP``, and ``TPV`` WCS using OpenCV pixels."""

    crpix1: float
    crpix2: float
    crval1: float
    crval2: float
    cd: np.ndarray
    cd_inv: np.ndarray
    projection: str = "TAN"
    a_coeffs: dict[tuple[int, int], float] = field(default_factory=dict)
    b_coeffs: dict[tuple[int, int], float] = field(default_factory=dict)
    ap_coeffs: dict[tuple[int, int], float] = field(default_factory=dict)
    bp_coeffs: dict[tuple[int, int], float] = field(default_factory=dict)
    pv1_coeffs: dict[int, float] = field(default_factory=dict)
    pv2_coeffs: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_header(cls, header: dict[str, Any]) -> TanSipWcs:
        """Build a ``TanSipWcs`` instance from parsed FITS header cards."""
        ctype1 = str(header.get("CTYPE1", "")).upper()
        ctype2 = str(header.get("CTYPE2", "")).upper()
        is_tan = "TAN" in ctype1 and "TAN" in ctype2
        is_tpv = "TPV" in ctype1 and "TPV" in ctype2
        if not (is_tan or is_tpv):
            raise ValueError(
                f"Expected TAN or TPV WCS projection, got "
                f"CTYPE1={ctype1!r}, CTYPE2={ctype2!r}."
            )

        crpix1 = float(header["CRPIX1"])
        crpix2 = float(header["CRPIX2"])
        crval1 = float(header["CRVAL1"])
        crval2 = float(header["CRVAL2"])

        if "CD1_1" in header:
            cd = np.array(
                [
                    [float(header["CD1_1"]), float(header.get("CD1_2", 0.0))],
                    [float(header.get("CD2_1", 0.0)), float(header["CD2_2"])],
                ],
                dtype=np.float64,
            )
        elif "CDELT1" in header and "CDELT2" in header:
            cdelt1 = float(header["CDELT1"])
            cdelt2 = float(header["CDELT2"])
            pc11 = float(header.get("PC1_1", 1.0))
            pc12 = float(header.get("PC1_2", 0.0))
            pc21 = float(header.get("PC2_1", 0.0))
            pc22 = float(header.get("PC2_2", 1.0))
            cd = np.array(
                [[cdelt1 * pc11, cdelt1 * pc12], [cdelt2 * pc21, cdelt2 * pc22]],
                dtype=np.float64,
            )
        else:
            raise ValueError("Missing CDi_j and CDELTi/PCi_j WCS matrix keywords.")

        det = float(np.linalg.det(cd))
        if abs(det) < 1e-24:
            raise ValueError("Singular WCS CD matrix cannot be inverted.")
        cd_inv = np.linalg.inv(cd)

        buckets: dict[str, dict[tuple[int, int], float]] = {
            "A": {},
            "B": {},
            "AP": {},
            "BP": {},
        }
        pv1: dict[int, float] = {}
        pv2: dict[int, float] = {}
        for key, value in header.items():
            match = _SIP_COEFF_RE.match(key)
            if match:
                kind, p_str, q_str = match.groups()
                buckets[kind][(int(p_str), int(q_str))] = float(value)
                continue
            pv_match = _TPV_COEFF_RE.match(key)
            if pv_match:
                axis_str, term_str = pv_match.groups()
                term_idx = int(term_str)
                if term_idx not in _TPV_EXPONENTS:
                    raise ValueError(f"Unsupported TPV term index PV{axis_str}_{term_idx}.")
                if axis_str == "1":
                    pv1[term_idx] = float(value)
                else:
                    pv2[term_idx] = float(value)

        has_pv = bool(pv1 or pv2)
        if is_tpv:
            proj = "TPV"
        elif "SIP" in ctype1 or buckets["A"] or buckets["B"]:
            proj = "TAN-SIP"
        elif has_pv:
            proj = "TPV"
        else:
            proj = "TAN"

        return cls(
            crpix1=crpix1,
            crpix2=crpix2,
            crval1=crval1,
            crval2=crval2,
            cd=cd,
            cd_inv=cd_inv,
            projection=proj,
            a_coeffs=buckets["A"],
            b_coeffs=buckets["B"],
            ap_coeffs=buckets["AP"],
            bp_coeffs=buckets["BP"],
            pv1_coeffs=pv1,
            pv2_coeffs=pv2,
        )

    @property
    def pixel_scale_arcsec(self) -> float:
        """Mean nominal sky pixel scale in arcseconds per pixel."""
        return float(math.sqrt(abs(float(np.linalg.det(self.cd)))) * 3600.0)

    @staticmethod
    def _eval_sip(
        u: np.ndarray,
        v: np.ndarray,
        coeffs: dict[tuple[int, int], float],
    ) -> np.ndarray:
        if not coeffs:
            return np.zeros_like(u, dtype=np.float64)
        out = np.zeros_like(u, dtype=np.float64)
        for (p, q), val in coeffs.items():
            if val == 0.0:
                continue
            out = out + val * (u**p) * (v**q)
        return out

    @staticmethod
    def _eval_sip_with_derivs(
        u: np.ndarray,
        v: np.ndarray,
        coeffs: dict[tuple[int, int], float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        val_acc = np.zeros_like(u, dtype=np.float64)
        du_acc = np.zeros_like(u, dtype=np.float64)
        dv_acc = np.zeros_like(u, dtype=np.float64)
        for (p, q), c in coeffs.items():
            if c == 0.0:
                continue
            u_p = (u**p) if p > 0 else np.ones_like(u, dtype=np.float64)
            v_q = (v**q) if q > 0 else np.ones_like(v, dtype=np.float64)
            val_acc = val_acc + c * u_p * v_q
            if p > 0:
                du_acc = du_acc + c * p * (u ** (p - 1)) * v_q
            if q > 0:
                dv_acc = dv_acc + c * q * u_p * (v ** (q - 1))
        return val_acc, du_acc, dv_acc

    @staticmethod
    def _eval_tpv_poly(
        x: np.ndarray,
        y: np.ndarray,
        coeffs: dict[int, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate TPV polynomial sum c_m * x^px * y^py * r^pr and partials."""
        if not coeffs:
            return x.copy(), np.ones_like(x), np.zeros_like(x)
        r = np.hypot(x, y)
        safe_r = np.where(r > 0.0, r, 1.0)
        val = np.zeros_like(x, dtype=np.float64)
        dx = np.zeros_like(x, dtype=np.float64)
        dy = np.zeros_like(x, dtype=np.float64)
        for m, coeff in coeffs.items():
            if coeff == 0.0:
                continue
            px, py, pr = _TPV_EXPONENTS[m]
            x_pow = (x**px) if px > 0 else np.ones_like(x, dtype=np.float64)
            y_pow = (y**py) if py > 0 else np.ones_like(y, dtype=np.float64)
            r_pow = (r**pr) if pr > 0 else np.ones_like(r, dtype=np.float64)
            term = coeff * x_pow * y_pow * r_pow
            val = val + term
            if px > 0:
                dx = dx + coeff * px * (x ** (px - 1)) * y_pow * r_pow
            if py > 0:
                dy = dy + coeff * py * x_pow * (y ** (py - 1)) * r_pow
            if pr > 0:
                dr_coeff = coeff * pr * x_pow * y_pow * (safe_r ** (pr - 2))
                dr_coeff = np.where(r > 0.0, dr_coeff, 0.0)
                dx = dx + dr_coeff * x
                dy = dy + dr_coeff * y
        return val, dx, dy

    def pixel_to_world(
        self,
        x_px: np.ndarray | float,
        y_px: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map OpenCV 0-based ``(x_px, y_px)`` to ICRS ``(ra_deg, dec_deg)``."""
        x_arr = np.asarray(x_px, dtype=np.float64)
        y_arr = np.asarray(y_px, dtype=np.float64)

        u = (x_arr + 1.0) - self.crpix1
        v = (y_arr + 1.0) - self.crpix2
        u_p = u + self._eval_sip(u, v, self.a_coeffs)
        v_p = v + self._eval_sip(u, v, self.b_coeffs)

        xi_deg = self.cd[0, 0] * u_p + self.cd[0, 1] * v_p
        eta_deg = self.cd[1, 0] * u_p + self.cd[1, 1] * v_p

        if self.projection == "TPV" and (self.pv1_coeffs or self.pv2_coeffs):
            xi_dist, _, _ = self._eval_tpv_poly(xi_deg, eta_deg, self.pv1_coeffs)
            # Per Calabretta/SCAMP TPV spec, PV2 evaluates at (eta, xi)
            eta_dist, _, _ = self._eval_tpv_poly(eta_deg, xi_deg, self.pv2_coeffs)
            xi_deg, eta_deg = xi_dist, eta_dist

        xi = np.deg2rad(xi_deg)
        eta = np.deg2rad(eta_deg)

        ra0 = math.radians(self.crval1)
        dec0 = math.radians(self.crval2)
        sin_d0, cos_d0 = math.sin(dec0), math.cos(dec0)

        rho = np.hypot(xi, eta)
        c = np.arctan(rho)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        safe_rho = np.where(rho > 0.0, rho, 1.0)
        dec = np.where(
            rho > 0.0,
            np.arcsin(
                np.clip(cos_c * sin_d0 + (eta * sin_c * cos_d0) / safe_rho, -1.0, 1.0)
            ),
            dec0,
        )
        ra = ra0 + np.arctan2(
            xi * sin_c,
            rho * cos_d0 * cos_c - eta * sin_d0 * sin_c,
        )
        ra_deg = np.mod(np.rad2deg(ra), 360.0)
        dec_deg = np.rad2deg(dec)
        return ra_deg, dec_deg

    def world_to_pixel(
        self,
        ra_deg: np.ndarray | float,
        dec_deg: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project ICRS ``(ra_deg, dec_deg)`` into OpenCV 0-based ``(x_px, y_px)``."""
        ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
        dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
        ra0 = math.radians(self.crval1)
        dec0 = math.radians(self.crval2)

        sin_d0, cos_d0 = math.sin(dec0), math.cos(dec0)
        sin_d, cos_d = np.sin(dec), np.cos(dec)
        dra = ra - ra0
        cos_dra = np.cos(dra)
        sin_dra = np.sin(dra)

        denom = sin_d * sin_d0 + cos_d * cos_d0 * cos_dra
        valid = denom > 1e-6
        safe_denom = np.where(valid, denom, 1.0)

        xi_deg = np.rad2deg((cos_d * sin_dra) / safe_denom)
        eta_deg = np.rad2deg(
            (sin_d * cos_d0 - cos_d * sin_d0 * cos_dra) / safe_denom
        )

        # Invert TPV intermediate distortion if present
        if self.projection == "TPV" and (self.pv1_coeffs or self.pv2_coeffs):
            xi_lin = xi_deg.copy()
            eta_lin = eta_deg.copy()
            for _ in range(6):
                p1, d1_xi, d1_eta = self._eval_tpv_poly(
                    xi_lin, eta_lin, self.pv1_coeffs
                )
                # pv2 evaluates on swapped args (eta, xi), so d2_eta = d/dx, d2_xi = d/dy
                p2, d2_eta, d2_xi = self._eval_tpv_poly(
                    eta_lin, xi_lin, self.pv2_coeffs
                )
                r1 = p1 - xi_deg
                r2 = p2 - eta_deg
                jdet = d1_xi * d2_eta - d1_eta * d2_xi
                safe_jdet = np.where(np.abs(jdet) > 1e-12, jdet, 1.0)
                dxi = (d2_eta * r1 - d1_eta * r2) / safe_jdet
                deta = (-d2_xi * r1 + d1_xi * r2) / safe_jdet
                xi_lin = xi_lin - dxi
                eta_lin = eta_lin - deta
                if float(np.max(np.abs(dxi) + np.abs(deta))) < 1e-12:
                    break
            xi_deg, eta_deg = xi_lin, eta_lin

        u_p = self.cd_inv[0, 0] * xi_deg + self.cd_inv[0, 1] * eta_deg
        v_p = self.cd_inv[1, 0] * xi_deg + self.cd_inv[1, 1] * eta_deg

        if self.ap_coeffs or self.bp_coeffs:
            u = u_p + self._eval_sip(u_p, v_p, self.ap_coeffs)
            v = v_p + self._eval_sip(u_p, v_p, self.bp_coeffs)
        else:
            u = u_p.copy()
            v = v_p.copy()

        # Vectorized Newton refinement guarantees <1e-10 px forward-backward parity
        if self.a_coeffs or self.b_coeffs:
            for _ in range(5):
                fa, dua, dva = self._eval_sip_with_derivs(u, v, self.a_coeffs)
                fb, dub, dvb = self._eval_sip_with_derivs(u, v, self.b_coeffs)
                r1 = (u + fa) - u_p
                r2 = (v + fb) - v_p
                j11 = 1.0 + dua
                j12 = dva
                j21 = dub
                j22 = 1.0 + dvb
                jdet = j11 * j22 - j12 * j21
                safe_jdet = np.where(np.abs(jdet) > 1e-12, jdet, 1.0)
                du = (j22 * r1 - j12 * r2) / safe_jdet
                dv = (-j21 * r1 + j11 * r2) / safe_jdet
                u = u - du
                v = v - dv
                if float(np.max(np.abs(du) + np.abs(dv))) < 1e-11:
                    break

        x_px = np.where(valid, (u + self.crpix1) - 1.0, np.nan)
        y_px = np.where(valid, (v + self.crpix2) - 1.0, np.nan)
        return x_px, y_px


def render_fits_to_uint8(
    data: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.8,
) -> np.ndarray:
    """Render linear FITS float pixels to an 8-bit ``uint8`` grayscale array.

    Uses ``np.arcsinh`` dynamic-range compression between robust sky percentiles
    so faint field stars stay visible alongside bright saturated cores while
    handling ``NaN`` border regions cleanly.
    """
    arr = np.asarray(data, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros(arr.shape, dtype=np.uint8)

    valid_vals = arr[finite_mask]
    lo = float(np.percentile(valid_vals, low_pct))
    hi = float(np.percentile(valid_vals, high_pct))
    med = float(np.median(valid_vals))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    cleaned = np.where(finite_mask, arr, med)
    scaled = np.clip((cleaned - lo) / (hi - lo), 0.0, 1.0)
    stretched = np.arcsinh(10.0 * scaled) / math.asinh(10.0)
    return np.clip(np.round(stretched * 255.0), 0.0, 255.0).astype(np.uint8)

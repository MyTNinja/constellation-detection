"""Optional downloader for the five ESA/Hubble Phase 1 sample images.

Provenance:
    Public-domain / CC-BY ESA/Hubble archive images used in the Phase 1
    traditional computer-vision star-detection comparison notebook:
    - Cassiopeia: https://cdn.esahubble.org/archives/images/large/opo0613c.jpg
    - Ursa Major: https://cdn.esahubble.org/archives/images/large/opo0706b.jpg
    - Ursa Minor: https://cdn.esahubble.org/archives/images/large/heic0401g.jpg
    - Orion:      https://cdn.esahubble.org/archives/images/large/opo0205b.jpg
    - Cygnus:     https://cdn.esahubble.org/archives/images/large/heic0720c.jpg

Usage:
    python scripts/download_phase1_samples.py [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from constellation_detection.config import ProjectPaths

ESA_HUBBLE_SAMPLES: dict[str, str] = {
    "cassiopeia_opo0613c.jpg": "https://cdn.esahubble.org/archives/images/large/opo0613c.jpg",
    "ursa_major_opo0706b.jpg": "https://cdn.esahubble.org/archives/images/large/opo0706b.jpg",
    "ursa_minor_heic0401g.jpg": "https://cdn.esahubble.org/archives/images/large/heic0401g.jpg",
    "orion_opo0205b.jpg": "https://cdn.esahubble.org/archives/images/large/opo0205b.jpg",
    "cygnus_heic0720c.jpg": "https://cdn.esahubble.org/archives/images/large/heic0720c.jpg",
}


def download_phase1_samples(
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download the five ESA/Hubble reference starfield images to ``output_dir``.

    Args:
        output_dir: Destination directory. Defaults to ``ProjectPaths.from_paths().data_dir / "raw"``.
        overwrite: If ``True``, re-download files even if they already exist locally.

    Returns:
        List of local ``Path`` objects for the downloaded or existing image files.
    """
    if output_dir is None:
        target_dir = ProjectPaths.from_paths().data_dir / "raw"
    else:
        target_dir = Path(output_dir).expanduser().resolve()

    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for filename, url in ESA_HUBBLE_SAMPLES.items():
        dest = target_dir / filename
        if dest.exists() and not overwrite:
            print(f"[skip] Already exists: {dest}")
            saved_paths.append(dest)
            continue

        print(f"[download] {url} -> {dest}")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "constellation-detection-research/0.1 (Python urllib)"},
        )
        with urllib.request.urlopen(req, timeout=30) as response, dest.open("wb") as fh:
            fh.write(response.read())
        saved_paths.append(dest)

    return saved_paths


def main() -> None:
    """CLI entrypoint for downloading optional ESA/Hubble Phase 1 sample images."""
    parser = argparse.ArgumentParser(
        description="Download optional ESA/Hubble Phase 1 sample starfield images."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target directory for raw images (default: <CONSTELLATION_DATA_DIR>/raw).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download images even if they already exist.",
    )
    args = parser.parse_args()
    download_phase1_samples(output_dir=args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

"""Four-panel diagnostic visualization and star detection overlays."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from constellation_detection.evaluation.compare import resize_to_max_dim
from constellation_detection.evaluation.export import build_image_output_stem
from constellation_detection.image.preprocessing import to_grayscale

__all__ = [
    "DETECTOR_COLORS",
    "plot_detector_diagnostics",
]

DETECTOR_COLORS: dict[str, str] = {
    "global_threshold": "cyan",
    "CV1_GlobalThreshold": "cyan",
    "CV2_AdaptiveTopHat": "green",
    "CV3_MultiScaleDoG": "magenta",
}


def _to_rgb_display(image: np.ndarray) -> np.ndarray:
    """Convert a 2D grayscale or 3D BGR/BGRA image into 3-channel RGB for Matplotlib."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported image shape for RGB display: {image.shape}.")


def plot_detector_diagnostics(
    image_bgr: np.ndarray,
    result: dict[str, Any],
    output_dir: str | os.PathLike[str] | None = None,
    filename: str | None = None,
    image_name: str | None = None,
    max_dim: int = 800,
    top_k_show: int = 35,
    connect_top_n: int = 14,
    connect_rad_ratio: float = 0.22,
    color: str | None = None,
) -> tuple[Figure, Path | None]:
    """Render the Phase 1 four-panel diagnostic figure for a star detector output.

    Panels:
        1. Input RGB image annotated with sky grayscale mean and standard deviation.
        2. Detector preprocessed image.
        3. Segmentation or blob mask.
        4. Detection overlay showing top-K ranked star markers and proximity connections
           among the top ``connect_top_n`` stars within ``connect_rad_ratio * max(H, W)``.

    Uses Matplotlib's object-oriented ``Figure`` and ``FigureCanvasAgg`` so execution is
    completely headless-safe without requiring an interactive display backend.

    Args:
        image_bgr: Input BGR or grayscale image at working or original resolution.
        result: Standardized detector output dict with ``detector``, ``stars_xyf``,
            ``preprocessed``, ``mask``, and ``latency_ms``.
        output_dir: Optional directory in which to save the rendered PNG.
        filename: Optional PNG filename (defaults to ``"<detector>_diagnostics.png"``).
        max_dim: Maximum dimension used if ``image_bgr`` does not match the detector's
            ``preprocessed`` resolution. Defaults to ``800``.
        top_k_show: Maximum number of ranked star markers to plot. Defaults to ``35``.
        connect_top_n: Number of brightest stars to connect by proximity. Defaults to ``14``.
        connect_rad_ratio: Proximity connection threshold as a fraction of ``max(H, W)``.
            Defaults to ``0.22``.
        color: Optional Matplotlib marker/line color override. Defaults to detector-mapped
            colors (``cyan``, ``green``, ``magenta``).

    Returns:
        Tuple ``(figure, saved_path)`` where ``saved_path`` is ``None`` if ``output_dir``
        was not provided.
    """
    detector_name = str(result["detector"])
    stars_xyf = np.asarray(result["stars_xyf"], dtype=np.float32)
    preprocessed = np.asarray(result["preprocessed"])
    mask = np.asarray(result["mask"])
    latency_ms = float(result["latency_ms"])

    working_img = image_bgr
    if working_img.shape[:2] != preprocessed.shape[:2]:
        working_img = resize_to_max_dim(image_bgr, max_dim=max_dim)

    rgb_img = _to_rgb_display(working_img)
    gray_img = to_grayscale(working_img)
    sky_mean = float(np.mean(gray_img))
    sky_std = float(np.std(gray_img))

    marker_color = color or DETECTOR_COLORS.get(detector_name, "cyan")

    fig = Figure(figsize=(20, 5), dpi=120)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 4)

    # Panel 1: Input RGB image with sky mean and std
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Input RGB\nSky mean={sky_mean:.1f}, std={sky_std:.1f}")
    axes[0].axis("off")

    # Panel 2: Preprocessed image
    axes[1].imshow(preprocessed, cmap="gray")
    axes[1].set_title(f"{detector_name}\nPreprocessed")
    axes[1].axis("off")

    # Panel 3: Segmentation / blob mask
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"{detector_name}\nMask")
    axes[2].axis("off")

    # Panel 4: Detection overlay with ranked markers and proximity connections
    axes[3].imshow(rgb_img)
    total_stars = len(stars_xyf)
    axes[3].set_title(
        f"{detector_name} Overlay\n"
        f"Detections={total_stars} | {latency_ms:.1f} ms"
    )
    axes[3].axis("off")

    if total_stars > 0:
        height, width = gray_img.shape
        connect_radius = float(connect_rad_ratio) * float(max(height, width))

        top_connect = stars_xyf[: min(int(connect_top_n), total_stars)]
        n_conn = len(top_connect)
        for i in range(n_conn):
            for j in range(i + 1, n_conn):
                dx = float(top_connect[i, 0] - top_connect[j, 0])
                dy = float(top_connect[i, 1] - top_connect[j, 1])
                if (dx * dx + dy * dy) <= connect_radius * connect_radius:
                    axes[3].plot(
                        [top_connect[i, 0], top_connect[j, 0]],
                        [top_connect[i, 1], top_connect[j, 1]],
                        color=marker_color,
                        linewidth=0.8,
                        alpha=0.55,
                    )

        top_show = stars_xyf[: min(int(top_k_show), total_stars)]
        axes[3].scatter(
            top_show[:, 0],
            top_show[:, 1],
            s=45,
            facecolors="none",
            edgecolors=marker_color,
            linewidths=1.2,
        )
        for rank_idx, star in enumerate(top_show, start=1):
            axes[3].text(
                float(star[0]) + 4.0,
                float(star[1]) - 4.0,
                str(rank_idx),
                color=marker_color,
                fontsize=7,
                weight="bold",
            )

    fig.tight_layout()

    saved_path: Path | None = None
    if output_dir is not None:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if filename is not None:
            png_name = filename
        elif image_name is not None:
            image_id = build_image_output_stem(image_name)
            png_name = f"{image_id}_{detector_name}_diagnostics.png"
        else:
            png_name = f"{detector_name}_diagnostics.png"
        saved_path = out_dir / png_name
        fig.savefig(saved_path, bbox_inches="tight")

    return fig, saved_path

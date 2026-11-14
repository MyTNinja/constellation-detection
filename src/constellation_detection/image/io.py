"""Image loading utilities using OpenCV."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

__all__ = ["load_image"]


def load_image(
    path: str | os.PathLike[str],
    grayscale: bool = False,
) -> np.ndarray:
    """Load an image from a filesystem path using OpenCV.

    Args:
        path: Filesystem path to the image file.
        grayscale: If ``True``, loads the image directly as a 2D single-channel
            grayscale array (``cv2.IMREAD_GRAYSCALE``). If ``False`` (default),
            loads the image as a 3-channel BGR array (``cv2.IMREAD_COLOR``).

    Returns:
        Decoded image as a NumPy ``ndarray`` with shape ``(H, W, 3)`` in BGR order
        when ``grayscale=False``, or ``(H, W)`` when ``grayscale=True``.

    Raises:
        TypeError: If ``path`` is not a string or ``os.PathLike`` object.
        FileNotFoundError: If the path does not exist on the filesystem.
        ValueError: If the path points to a directory rather than a file, or if
            OpenCV fails to decode the file as a valid image.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"Expected path to be str or PathLike, got {type(path).__name__}.")

    path_str = str(path).strip()
    if not path_str:
        raise ValueError("Image path must not be empty.")

    image_path = Path(path).expanduser()

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Image path is not a regular file: {image_path}")

    if image_path.stat().st_size == 0:
        raise ValueError(f"Image file is empty (0 bytes): {image_path}")

    read_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), read_flag)

    if image is None or image.size == 0:
        raise ValueError(f"Failed to decode image file: {image_path}")

    return image

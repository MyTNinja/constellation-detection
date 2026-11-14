"""Unit tests for constellation_detection.image.io."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from constellation_detection.image import load_image as load_image_pkg
from constellation_detection.image.io import load_image


def test_load_image_reexport() -> None:
    """Verify load_image is re-exported by constellation_detection.image."""
    assert load_image is load_image_pkg


def test_load_valid_color_image(tmp_path: Path) -> None:
    """Valid BGR image on disk is loaded with expected shape and values."""
    synth_bgr = np.zeros((32, 48, 3), dtype=np.uint8)
    synth_bgr[10, 20] = [255, 128, 64]

    img_path = tmp_path / "synthetic_starfield.png"
    assert cv2.imwrite(str(img_path), synth_bgr)

    loaded = load_image(img_path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (32, 48, 3)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, synth_bgr)


def test_load_valid_image_as_grayscale(tmp_path: Path) -> None:
    """Valid image loaded with grayscale=True returns a 2D uint8 array."""
    synth_gray = np.full((24, 36), 180, dtype=np.uint8)
    img_path = tmp_path / "gray_starfield.png"
    assert cv2.imwrite(str(img_path), synth_gray)

    loaded = load_image(str(img_path), grayscale=True)

    assert isinstance(loaded, np.ndarray)
    assert loaded.ndim == 2
    assert loaded.shape == (24, 36)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, synth_gray)


def test_load_missing_image_raises_file_not_found(tmp_path: Path) -> None:
    """Missing image path raises FileNotFoundError."""
    missing_path = tmp_path / "nonexistent_image.png"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_image(missing_path)


def test_load_directory_path_raises_value_error(tmp_path: Path) -> None:
    """Passing a directory path instead of a file raises ValueError."""
    with pytest.raises(ValueError, match="not a regular file"):
        load_image(tmp_path)


def test_load_empty_file_raises_value_error(tmp_path: Path) -> None:
    """Zero-byte file raises ValueError."""
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        load_image(empty_file)


def test_load_invalid_corrupt_image_raises_value_error(tmp_path: Path) -> None:
    """Corrupt/non-image file content raises ValueError when OpenCV cannot decode it."""
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"not a valid image payload")

    with pytest.raises(ValueError, match="Failed to decode"):
        load_image(corrupt_file)


def test_load_invalid_path_type_raises_type_error() -> None:
    """Non-path input raises TypeError."""
    with pytest.raises(TypeError, match="Expected path"):
        load_image(12345)  # type: ignore[arg-type]


def test_load_blank_path_string_raises_value_error() -> None:
    """Blank string path raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        load_image("   ")

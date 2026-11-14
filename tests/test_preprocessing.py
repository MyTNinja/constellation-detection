"""Unit tests for constellation_detection.image.preprocessing."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from constellation_detection.image.preprocessing import (
    apply_gaussian_blur,
    preprocess_image,
    to_grayscale,
)


def test_to_grayscale_from_bgr() -> None:
    """BGR 3-channel image is converted to a 2D grayscale array."""
    bgr = np.zeros((20, 30, 3), dtype=np.uint8)
    bgr[:, :] = [255, 255, 255]  # Pure white in BGR should map to 255 in grayscale

    gray = to_grayscale(bgr)

    assert isinstance(gray, np.ndarray)
    assert gray.ndim == 2
    assert gray.shape == (20, 30)
    assert gray.dtype == np.uint8
    np.testing.assert_array_equal(gray, np.full((20, 30), 255, dtype=np.uint8))


def test_to_grayscale_accepts_already_grayscale_2d() -> None:
    """Already-grayscale 2D image is returned as an independent 2D copy."""
    original = np.arange(0, 64, dtype=np.uint8).reshape((8, 8))

    gray = to_grayscale(original)

    assert gray.ndim == 2
    assert gray.shape == (8, 8)
    assert gray.dtype == np.uint8
    np.testing.assert_array_equal(gray, original)
    assert gray is not original


def test_to_grayscale_accepts_single_channel_3d() -> None:
    """Single-channel 3D image (H, W, 1) is squeezed to 2D (H, W)."""
    single_ch = np.full((12, 16, 1), 127, dtype=np.uint8)

    gray = to_grayscale(single_ch)

    assert gray.ndim == 2
    assert gray.shape == (12, 16)
    np.testing.assert_array_equal(gray, np.full((12, 16), 127, dtype=np.uint8))


def test_to_grayscale_from_bgra() -> None:
    """BGRA 4-channel image is converted to 2D grayscale."""
    bgra = np.zeros((10, 10, 4), dtype=np.uint8)
    bgra[:, :] = [0, 0, 0, 255]

    gray = to_grayscale(bgra)

    assert gray.ndim == 2
    assert gray.shape == (10, 10)
    np.testing.assert_array_equal(gray, np.zeros((10, 10), dtype=np.uint8))


def test_apply_gaussian_blur_with_tuple_and_int_kernel() -> None:
    """Gaussian blur smooths a sharp impulse peak and supports int or tuple kernel sizes."""
    impulse = np.zeros((21, 21), dtype=np.float32)
    impulse[10, 10] = 100.0

    blurred_tuple = apply_gaussian_blur(impulse, kernel_size=(5, 5), sigma_x=1.0)
    blurred_int = apply_gaussian_blur(impulse, kernel_size=5, sigma_x=1.0)

    assert blurred_tuple.shape == impulse.shape
    assert blurred_tuple.dtype == np.float32
    np.testing.assert_allclose(blurred_tuple, blurred_int, rtol=1e-6)

    # Peak intensity is reduced and spread to immediate neighbors
    assert blurred_tuple[10, 10] < 100.0
    assert blurred_tuple[10, 10] > blurred_tuple[10, 11] > 0.0
    # Total energy is conserved in the interior
    np.testing.assert_allclose(blurred_tuple.sum(), 100.0, rtol=1e-4)


def test_preprocess_image_pipeline() -> None:
    """preprocess_image converts BGR input to 2D grayscale and applies Gaussian blur."""
    bgr = np.zeros((15, 15, 3), dtype=np.uint8)
    bgr[7, 7] = [255, 255, 255]

    result = preprocess_image(bgr, kernel_size=(3, 3), sigma_x=0.0)
    expected = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), sigmaX=0.0)

    assert result.ndim == 2
    assert result.shape == (15, 15)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "invalid_input,expected_exc",
    [
        ([[1, 2], [3, 4]], TypeError),
        (np.array([1, 2, 3], dtype=np.uint8), ValueError),
        (np.zeros((2, 2, 3, 1), dtype=np.uint8), ValueError),
        (np.zeros((0, 10), dtype=np.uint8), ValueError),
        (np.zeros((10, 10, 2), dtype=np.uint8), ValueError),
    ],
)
def test_to_grayscale_invalid_inputs(
    invalid_input: object,
    expected_exc: type[Exception],
) -> None:
    """Invalid image types, dimensions, or channel counts raise clear exceptions."""
    with pytest.raises(expected_exc):
        to_grayscale(invalid_input)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_kernel,expected_exc",
    [
        (4, ValueError),
        (0, ValueError),
        (-3, ValueError),
        ((3, 4), ValueError),
        ((5, 5, 5), ValueError),
        (True, TypeError),
        ((3, True), TypeError),
    ],
)
def test_apply_gaussian_blur_invalid_kernel_size(
    invalid_kernel: object,
    expected_exc: type[Exception],
) -> None:
    """Even, non-positive, or malformed kernel sizes raise ValueError or TypeError."""
    image = np.zeros((11, 11), dtype=np.uint8)
    with pytest.raises(expected_exc):
        apply_gaussian_blur(image, kernel_size=invalid_kernel)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_sigma",
    [
        -1.0,
        -0.001,
        float("nan"),
        float("inf"),
        float("-inf"),
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_apply_gaussian_blur_non_finite_or_negative_sigma_raises_value_error(
    invalid_sigma: float,
) -> None:
    """Negative, NaN, and positive/negative infinite sigma_x values raise ValueError."""
    image = np.zeros((11, 11), dtype=np.uint8)
    with pytest.raises(ValueError, match="sigma_x must be finite and non-negative"):
        apply_gaussian_blur(image, kernel_size=(3, 3), sigma_x=invalid_sigma)


@pytest.mark.parametrize(
    "invalid_sigma_type",
    [
        True,
        False,
        np.bool_(True),
        1 + 2j,
        np.complex128(1.0 + 0.0j),
        "1.0",
        None,
    ],
)
def test_apply_gaussian_blur_non_real_sigma_raises_type_error(
    invalid_sigma_type: object,
) -> None:
    """Boolean and non-real numeric sigma_x inputs raise TypeError."""
    image = np.zeros((11, 11), dtype=np.uint8)
    with pytest.raises(TypeError, match="sigma_x must be a real number"):
        apply_gaussian_blur(image, kernel_size=(3, 3), sigma_x=invalid_sigma_type)  # type: ignore[arg-type]


def test_apply_gaussian_blur_accepts_numpy_numeric_scalars() -> None:
    """Valid non-negative NumPy integer and floating scalars are accepted for sigma_x."""
    image = np.zeros((11, 11), dtype=np.float32)
    image[5, 5] = 50.0

    out_float = apply_gaussian_blur(image, kernel_size=3, sigma_x=np.float32(1.25))
    out_int = apply_gaussian_blur(image, kernel_size=3, sigma_x=np.int64(1))

    assert out_float.shape == (11, 11)
    assert out_int.shape == (11, 11)


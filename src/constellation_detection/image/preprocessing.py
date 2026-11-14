"""Image preprocessing utilities for star and constellation detection."""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "apply_gaussian_blur",
    "preprocess_image",
    "to_grayscale",
]


def _validate_image_array(image: np.ndarray) -> None:
    """Validate that the input is a non-empty 2D or 3D NumPy array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be a numpy.ndarray, got {type(image).__name__}.")

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D (H, W) or 3D (H, W, C) image array, got {image.ndim}D array "
            f"with shape {image.shape}."
        )

    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Image array must not be empty, got shape {image.shape}.")


def _normalize_kernel_size(kernel_size: int | tuple[int, int]) -> tuple[int, int]:
    """Validate and normalize Gaussian kernel size to a ``(kx, ky)`` pair of odd positive ints."""
    if isinstance(kernel_size, bool):
        raise TypeError("kernel_size must be an int or a 2-tuple of ints, not bool.")

    if isinstance(kernel_size, (int, np.integer)):
        k = int(kernel_size)
        kx, ky = k, k
    elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
        if any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer))
            for v in kernel_size
        ):
            raise TypeError("kernel_size tuple elements must be integers.")
        kx, ky = int(kernel_size[0]), int(kernel_size[1])
    else:
        raise ValueError(
            f"kernel_size must be an odd positive integer or a 2-tuple of odd positive "
            f"integers, got {kernel_size!r}."
        )

    if kx <= 0 or ky <= 0 or kx % 2 == 0 or ky % 2 == 0:
        raise ValueError(
            f"Gaussian kernel dimensions must be positive odd integers, got ({kx}, {ky})."
        )

    return kx, ky


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR (or BGRA / single-channel 3D) image to a 2D grayscale array.

    If the input image is already a 2D grayscale array, returns a copy of the array.

    Args:
        image: Input image as a NumPy ``ndarray`` of shape ``(H, W)`` or ``(H, W, C)``.
            3-channel images are assumed to follow OpenCV's BGR channel ordering.

    Returns:
        2D grayscale NumPy ``ndarray`` of shape ``(H, W)`` with the same dtype as ``image``.

    Raises:
        TypeError: If ``image`` is not a NumPy ``ndarray``.
        ValueError: If ``image`` is empty, is not 2D or 3D, or has an unsupported
            number of channels.
    """
    _validate_image_array(image)

    if image.ndim == 2:
        return image.copy()

    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0].copy()
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(
        f"Unsupported number of channels for grayscale conversion: {channels}. "
        "Expected 1, 3 (BGR), or 4 (BGRA)."
    )


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int | tuple[int, int] = (5, 5),
    sigma_x: float = 0.0,
) -> np.ndarray:
    """Apply Gaussian smoothing to an image.

    Args:
        image: Input 2D ``(H, W)`` or 3D ``(H, W, C)`` NumPy ``ndarray``.
        kernel_size: Gaussian kernel size in pixels, specified either as a single
            positive odd integer ``k`` (for a ``(k, k)`` kernel) or a 2-tuple of
            positive odd integers ``(kx, ky)``. Defaults to ``(5, 5)``.
        sigma_x: Gaussian kernel standard deviation in the X direction. When ``0.0``
            (default), OpenCV computes sigma automatically from ``kernel_size``.

    Returns:
        Smoothed NumPy ``ndarray`` with the same shape and dtype as ``image``.

    Raises:
        TypeError: If ``image`` is not a ``numpy.ndarray``, ``kernel_size`` has an
            invalid type, or ``sigma_x`` is not a real number.
        ValueError: If ``image`` has invalid dimensions, ``kernel_size`` is not
            positive and odd, or ``sigma_x`` is not finite and non-negative.
    """
    _validate_image_array(image)
    kx, ky = _normalize_kernel_size(kernel_size)

    if isinstance(sigma_x, (bool, np.bool_)) or not isinstance(
        sigma_x, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"sigma_x must be a real number, got {type(sigma_x).__name__}.")

    sigma_val = float(sigma_x)
    if not np.isfinite(sigma_val) or sigma_val < 0.0:
        raise ValueError(f"sigma_x must be finite and non-negative, got {sigma_x}.")

    return cv2.GaussianBlur(image, (kx, ky), sigmaX=sigma_val)


def preprocess_image(
    image: np.ndarray,
    kernel_size: int | tuple[int, int] = (5, 5),
    sigma_x: float = 0.0,
) -> np.ndarray:
    """Convert an image to 2D grayscale and apply Gaussian smoothing.

    Args:
        image: Input 2D grayscale or 3D BGR image array.
        kernel_size: Positive odd integer or 2-tuple ``(kx, ky)`` for Gaussian blur.
            Defaults to ``(5, 5)``.
        sigma_x: Gaussian standard deviation along X. Defaults to ``0.0``.

    Returns:
        2D blurred grayscale ``np.ndarray`` of shape ``(H, W)``.
    """
    gray = to_grayscale(image)
    return apply_gaussian_blur(gray, kernel_size=kernel_size, sigma_x=sigma_x)

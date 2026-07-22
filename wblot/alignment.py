"""Rotation-based scan alignment for Western blot images.

The alignment preview is intentionally separate from measurement: the grid is
only a visual guide, while :func:`rotate_loaded` creates the corrected working
image that downstream detection and densitometry receive.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, rotate

from .image_io import LoadedImage


def _border_background(gray: np.ndarray) -> float:
    """Return a robust fill value from the outer edge of an image."""
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("Alignment requires a non-empty grayscale image.")
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    return float(np.median(border))


def rotate_loaded(image: LoadedImage, correction_angle: float) -> LoadedImage:
    """Rotate a working blot without changing its canvas or native range.

    Positive values rotate counter-clockwise. Corners exposed by the rotation
    are filled with the scan's border median rather than black, preventing the
    new edges from being mistaken for bands.
    """
    angle = float(np.clip(correction_angle, -15.0, 15.0))
    if abs(angle) < 1e-9:
        gray = image.gray.copy()
    else:
        gray = rotate(
            image.gray,
            angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=_border_background(image.gray),
            prefilter=False,
        )
        gray = np.clip(gray, 0.0, image.max_value)
    return LoadedImage(
        gray=np.asarray(gray, dtype=np.float64),
        max_value=image.max_value,
        bit_depth=image.bit_depth,
        name=image.name,
    )


def _alignment_score(detail: np.ndarray, angle: float) -> float:
    """Score how sharply band signal concentrates into horizontal rows."""
    rotated = rotate(
        detail,
        float(angle),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    height, width = rotated.shape
    y0, y1 = int(height * 0.08), max(int(height * 0.92), 1)
    x0, x1 = int(width * 0.08), max(int(width * 0.92), 1)
    core = rotated[y0:y1, x0:x1]
    if core.size == 0:
        core = rotated
    row_profile = core.sum(axis=1)
    mean = float(np.mean(row_profile))
    return float(np.var(row_profile) / (mean * mean + 1e-12))


def estimate_correction_angle(
    gray: np.ndarray,
    max_angle: float = 12.0,
) -> float:
    """Estimate the rotation that makes the dominant bands horizontal.

    The estimator is polarity-independent. It removes slowly varying
    background, then searches for the angle that gives the sharpest horizontal
    intensity projection. The result is rounded to the app's 0.1-degree control.
    """
    arr = np.asarray(gray, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Alignment requires a non-empty grayscale image.")
    max_angle = float(np.clip(abs(max_angle), 0.1, 15.0))

    # Keep estimation responsive on multi-megapixel TIFF scans. Striding does
    # not change line angle and avoids another interpolation before scoring.
    stride = max(1, int(np.ceil(max(arr.shape) / 600.0)))
    small = arr[::stride, ::stride]
    if min(small.shape) < 8 or float(np.ptp(small)) <= 1e-12:
        return 0.0

    sigma = max(2.0, min(small.shape) / 35.0)
    detail = np.abs(small - gaussian_filter(small, sigma=sigma))
    ceiling = float(np.percentile(detail, 99.5))
    if ceiling <= 1e-12:
        return 0.0
    detail = np.clip(detail, 0.0, ceiling)

    coarse_angles = np.arange(-max_angle, max_angle + 0.001, 0.5)
    coarse_scores = np.asarray([_alignment_score(detail, a) for a in coarse_angles])
    coarse_best = float(coarse_angles[int(np.argmax(coarse_scores))])

    refine_lo = max(-max_angle, coarse_best - 0.5)
    refine_hi = min(max_angle, coarse_best + 0.5)
    fine_angles = np.arange(refine_lo, refine_hi + 0.001, 0.1)
    fine_scores = np.asarray([_alignment_score(detail, a) for a in fine_angles])
    best = float(fine_angles[int(np.argmax(fine_scores))])
    return round(best, 1)

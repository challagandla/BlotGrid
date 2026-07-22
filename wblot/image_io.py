"""Image loading and signal-polarity handling for Western blot densitometry.

Key idea: we always work internally with a *signal* image where **higher pixel
values mean more protein**. Blots come in two flavours:

* dark bands on a light background (scanned film)  -> signal = max - pixel
* light bands on a dark background (chemi/fluor)    -> signal = pixel

The native bit depth is preserved for measurement (we never downcast 16-bit to
8-bit before measuring). Display images are contrast-stretched copies only.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class LoadedImage:
    """A loaded blot kept in its native intensity range."""

    gray: np.ndarray          # float64, native range (e.g. 0..255 or 0..65535)
    max_value: float          # theoretical max for the bit depth (255 or 65535)
    bit_depth: int            # 8 or 16
    name: str = "blot"

    @property
    def height(self) -> int:
        return int(self.gray.shape[0])

    @property
    def width(self) -> int:
        return int(self.gray.shape[1])


def load_image(source, name: str = "blot") -> LoadedImage:
    """Load an image from a path, bytes, or file-like object.

    Colour images are converted to grayscale using luminance weighting.
    16-bit TIFFs keep their full range.
    """
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)

    arr = np.asarray(img)

    # Determine bit depth from dtype width, not from a single dtype match, so
    # that big-endian, signed and 32-bit TIFFs are not silently treated as 8-bit
    # (which would put max_value at 255 and mis-flag saturation on every pixel).
    if arr.dtype.kind in "ui":
        itemsize = arr.dtype.itemsize
        if itemsize == 1:
            bit_depth, max_value = 8, 255.0
        elif itemsize == 2:
            bit_depth, max_value = 16, 65535.0
        else:
            bit_depth, max_value = 32, float(max(np.max(arr), 1))
    else:  # float images carry no declared range; take it from the data
        bit_depth = 32
        max_value = float(max(np.max(arr), 1.0))

    arr = arr.astype(np.float64)

    # Convert to single-channel grayscale.
    if arr.ndim == 3:
        if arr.shape[2] == 4:  # drop alpha
            arr = arr[:, :, :3]
        if arr.shape[2] == 3:
            arr = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        else:
            arr = arr[:, :, 0]

    return LoadedImage(gray=arr, max_value=max_value, bit_depth=bit_depth, name=name)


def to_signal(loaded: LoadedImage, bands_are_dark: bool) -> np.ndarray:
    """Return a signal image where higher = more protein.

    bands_are_dark=True  -> dark bands on light background (film): invert.
    bands_are_dark=False -> light bands on dark background: use as-is.
    """
    if bands_are_dark:
        return loaded.max_value - loaded.gray
    return loaded.gray.copy()


def to_display(loaded: LoadedImage) -> np.ndarray:
    """Contrast-stretched uint8 copy for on-screen display ONLY (never measured)."""
    g = loaded.gray
    lo, hi = np.percentile(g, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    disp = np.clip((g - lo) / (hi - lo), 0, 1) * 255.0
    return disp.astype(np.uint8)


def saturated_mask(
    loaded: LoadedImage,
    bands_are_dark: bool = False,
) -> np.ndarray:
    """Boolean mask of pixels clipped in the band-signal direction.

    A light-on-dark acquisition clips at the top of the native range.  Film and
    other dark-on-light images clip at zero, because inversion maps those dark
    pixels to maximum signal.  Treating both polarities as top-clipped missed
    saturated film bands in the upload quality check.
    """
    if bands_are_dark:
        return loaded.gray <= 0
    return loaded.gray >= loaded.max_value

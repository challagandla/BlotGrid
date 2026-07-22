"""Band densitometry: background subtraction and integrated-density measurement.

All functions operate on a *signal* image (higher = more protein), produced by
`image_io.to_signal`. Integrated density (net_intden) is the quantity used for
downstream normalization.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

try:  # rolling-ball is optional; degrade gracefully if unavailable
    from skimage.restoration import rolling_ball as _rolling_ball
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False


@dataclass
class BandMeasurement:
    target: str
    lane: str
    x: int
    y: int
    w: int
    h: int
    area: int
    raw_intden: float          # sum of signal in ROI (no background removed)
    background_per_px: float   # estimated background level per pixel
    net_intden: float          # background-subtracted integrated density (>=0)
    mean_signal: float
    saturated: bool
    measurement_mode: str = "rectangle"
    profile_y0: int | None = None
    profile_y1: int | None = None

    def as_row(self) -> dict:
        return asdict(self)


def rolling_ball_background(signal: np.ndarray, radius: float = 50.0) -> np.ndarray:
    """Estimate a smooth background via the rolling-ball algorithm.

    Returns a background image the same shape as `signal`. Falls back to a
    constant (image minimum) if scikit-image is not installed.
    """
    if not _HAS_SKIMAGE:
        return np.full_like(signal, float(np.min(signal)))
    return _rolling_ball(signal, radius=radius)


def _background_level(
    signal: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    method: str,
    margin: int = 4,
    bg_image: np.ndarray | None = None,
    bg_value: float | None = None,
) -> float:
    """Estimate the per-pixel background for a single ROI."""
    H, W = signal.shape
    if method == "rolling_ball":
        if bg_image is None:
            raise ValueError("method='rolling_ball' requires bg_image.")
        return float(np.mean(bg_image[y : y + h, x : x + w]))
    if method == "fixed":
        if bg_value is None:
            raise ValueError("method='fixed' requires bg_value.")
        return float(bg_value)
    if method != "local":
        raise ValueError(f"Unknown background method {method!r}.")
    # method == "local": median of a margin frame around the ROI.
    y0, y1 = max(0, y - margin), min(H, y + h + margin)
    x0, x1 = max(0, x - margin), min(W, x + w + margin)
    frame = signal[y0:y1, x0:x1].copy().astype(np.float64)
    # blank out the band itself so only surrounding pixels contribute
    iy0, iy1 = y - y0, y - y0 + h
    ix0, ix1 = x - x0, x - x0 + w
    frame[iy0:iy1, ix0:ix1] = np.nan
    vals = frame[~np.isnan(frame)]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


def _robust_profile_noise(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    noise = 1.4826 * float(np.median(np.abs(values - median)))
    if noise <= np.finfo(float).eps:
        noise = max(
            float(np.percentile(values, 75) - np.percentile(values, 25)) / 1.349,
            np.finfo(float).eps,
        )
    return noise


def profile_peak_interval(roi: np.ndarray, background_per_px: float) -> tuple[int, int, np.ndarray]:
    """Return the dominant band interval within an ROI and its row profile.

    The threshold is used only to choose reproducible valley-like row
    boundaries.  Every pixel across the locked lane width is still integrated,
    avoiding the intensity-dependent bias caused by selecting only the darkest
    pixels of strong bands.
    """
    values = np.asarray(roi, dtype=float)
    if values.ndim != 2 or values.size == 0:
        return 0, 0, np.zeros(0, dtype=float)
    corrected = values - float(background_per_px)
    profile = corrected.mean(axis=1)
    smooth = gaussian_filter1d(profile, sigma=max(0.65, len(profile) / 45.0))
    baseline = float(np.percentile(smooth, 20.0))
    lower = smooth[smooth <= np.percentile(smooth, 55.0)]
    noise = _robust_profile_noise(lower if lower.size else smooth)
    dynamic = max(float(np.max(smooth) - baseline), 0.0)
    if dynamic <= max(noise, np.finfo(float).eps):
        return 0, len(profile), smooth

    minimum = baseline + max(1.15 * noise, dynamic * 0.10)
    peaks, _ = find_peaks(smooth, height=minimum)
    centre = (len(smooth) - 1) / 2.0
    if len(peaks):
        # The editable ROI is centred on its intended band.  Distance from the
        # ROI centre therefore wins ties between a band and an edge artefact.
        scale = max(len(smooth) * 0.35, 1.0)
        merit = (smooth[peaks] - baseline) / max(dynamic, np.finfo(float).eps)
        merit -= np.abs(peaks - centre) / scale * 0.90
        peak = int(peaks[int(np.argmax(merit))])
    else:
        peak = int(np.argmax(smooth))

    boundary = baseline + max(0.75 * noise, (float(smooth[peak]) - baseline) * 0.075)
    left = peak
    while left > 0 and smooth[left - 1] > boundary:
        left -= 1
    right = peak + 1
    while right < len(smooth) and smooth[right] > boundary:
        right += 1

    # Preserve faint tails and keep a minimum measurable peak thickness.
    left = max(0, left - 1)
    right = min(len(smooth), right + 1)
    minimum_rows = min(len(smooth), 3)
    if right - left < minimum_rows:
        left = max(0, peak - minimum_rows // 2)
        right = min(len(smooth), left + minimum_rows)
        left = max(0, right - minimum_rows)
    return int(left), int(right), smooth


def measure_band(
    signal: np.ndarray,
    target: str,
    lane: str,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    method: str = "local",
    margin: int = 4,
    bg_image: np.ndarray | None = None,
    bg_value: float | None = None,
    max_value: float = 255.0,
    measurement_mode: str = "rectangle",
) -> BandMeasurement:
    """Measure one rectangular band ROI.

    method: 'local' (median frame around band), 'rolling_ball' (needs bg_image),
    or 'fixed' (needs bg_value). ``measurement_mode='profile'`` integrates the
    dominant peak between locally estimated row boundaries; ``'rectangle'``
    preserves classic full-ROI integration.
    """
    H, W = signal.shape
    # Clip the original rectangle endpoints, not the already-clipped origin.
    # Otherwise x=-5, w=10 incorrectly measures x=0..10 instead of x=0..5.
    x_in, y_in, w_in, h_in = int(x), int(y), int(w), int(h)
    x, y = max(0, x_in), max(0, y_in)
    x2, y2 = min(W, x_in + w_in), min(H, y_in + h_in)
    w, h = max(0, x2 - x), max(0, y2 - y)
    roi = signal[y:y2, x:x2]
    area = int(roi.size)
    if area == 0:
        return BandMeasurement(target, lane, x, y, w, h, 0, 0.0, 0.0, 0.0, 0.0, False)

    if measurement_mode not in {"rectangle", "profile"}:
        raise ValueError("measurement_mode must be 'rectangle' or 'profile'.")
    bg = _background_level(signal, x, y, w, h, method, margin, bg_image, bg_value)
    profile_y0, profile_y1 = y, y + h
    measured_roi = roi
    if measurement_mode == "profile":
        local_y0, local_y1, _ = profile_peak_interval(roi, bg)
        measured_roi = roi[local_y0:local_y1, :]
        profile_y0, profile_y1 = y + local_y0, y + local_y1

    area = int(measured_roi.size)
    raw_intden = float(np.sum(measured_roi))
    net = raw_intden - bg * area
    net = max(net, 0.0)
    mean_signal = float(np.mean(measured_roi)) if area else 0.0

    # Saturation means the detector clipped where the protein is. Both polarities
    # map a clipped band pixel to signal == max_value (film: gray 0 -> max - 0;
    # chemi: gray max -> max). A signal of 0 is just blank background, not
    # saturation, so it must not be flagged.
    saturated = bool(np.any(measured_roi >= max_value))

    return BandMeasurement(
        target=target, lane=lane, x=x, y=y, w=w, h=h, area=area,
        raw_intden=raw_intden, background_per_px=bg, net_intden=net,
        mean_signal=mean_signal, saturated=saturated,
        measurement_mode=measurement_mode,
        profile_y0=profile_y0,
        profile_y1=profile_y1,
    )


def lane_profile(signal: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Mean signal per row within a lane ROI (ImageJ-style profile).

    Peaks in the returned 1-D array correspond to bands along the lane.
    """
    H, W = signal.shape
    x, y = max(0, int(x)), max(0, int(y))
    x2, y2 = min(W, x + int(w)), min(H, y + int(h))
    roi = signal[y:y2, x:x2]
    if roi.size == 0:
        return np.zeros(0)
    return roi.mean(axis=1)

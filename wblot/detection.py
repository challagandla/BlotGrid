"""Automatic Western-blot lane and band detection.

The detector deliberately uses the experiment structure supplied by the user
(ordered lane and target labels), because protein identities cannot be inferred
from pixels alone.  Everything that *can* be inferred from the image is
automatic: signal polarity, lane centres, target-row centres, the number of
bands in each target/lane cell, and an individual rectangle for every band.

The implementation is based on background-corrected intensity projections.  A
wide Gaussian removes illumination gradients while a narrow Gaussian suppresses
pixel noise.  One-dimensional peak detection is then applied within every
automatically located lane/target cell.  This works well for both compact and
diffuse bands and, unlike connected-component-only approaches, does not lose
faint bands when a stronger band sets the global threshold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import (binary_dilation, binary_opening, gaussian_filter,
                           gaussian_filter1d, label)
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_prominences, peak_widths


@dataclass
class DetectedBand:
    """One automatically detected, editable band rectangle."""

    target: str
    lane: str
    x: int
    y: int
    w: int
    h: int
    confidence: float
    use: bool = True
    source: str = "detected"
    status: str = "accepted"
    review_required: bool = False
    component_count: int = 1
    profile_span: float = 0.0

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class DetectionResult:
    bands: list[DetectedBand]
    lane_centers: list[int]
    target_centers: list[int]
    bands_are_dark: bool
    polarity_confidence: float
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RawGridInference:
    """Pixel-only row/lane grid inferred when a scan has no readable labels."""

    lane_centers: tuple[int, ...]
    target_centers: tuple[int, ...]
    bands_are_dark: bool
    polarity_confidence: float
    grid_confidence: float
    warning: str = ""


def _standardize_boxes(
    bands: list[DetectedBand],
    image_shape: tuple[int, int],
    quantile: float = 0.90,
) -> None:
    """Give every detected band on the blot one shared rectangle size.

    Equal ROI geometry prevents differences in the automatically estimated box
    size from becoming a source of densitometry bias.  The 90th percentile keeps
    enough room for broad bands without letting a single extreme artefact make
    every rectangle unnecessarily large.  Centres remain independently located.
    """
    H, W = image_shape
    if not bands:
        return
    shared_w = int(np.clip(np.ceil(np.quantile([band.w for band in bands], quantile)), 1, W))
    shared_h = int(np.clip(np.ceil(np.quantile([band.h for band in bands], quantile)), 1, H))
    for band in bands:
        cx, cy = band.x + band.w / 2.0, band.y + band.h / 2.0
        band.x = int(np.clip(round(cx - shared_w / 2.0), 0, W - shared_w))
        band.y = int(np.clip(round(cy - shared_h / 2.0), 0, H - shared_h))
        band.w, band.h = shared_w, shared_h


def _finite_image(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Band detection requires a two-dimensional grayscale image.")
    if min(arr.shape) < 8:
        raise ValueError("The image is too small for automatic band detection.")
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("The image contains no finite pixel values.")
    if not finite.all():
        arr = arr.copy()
        arr[~finite] = float(np.median(arr[finite]))
    return arr


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _enhance(signal: np.ndarray, n_lanes: int, n_targets: int) -> np.ndarray:
    """Remove smooth background and return positive band-like signal."""
    H, W = signal.shape
    # The narrow blur rejects isolated hot pixels.  The broad blur is sized from
    # the expected grid so that a horizontal band remains foreground while slow
    # illumination changes disappear.
    small = gaussian_filter(signal, sigma=(max(0.6, H / 500), max(0.8, W / 500)))
    broad = gaussian_filter(
        signal,
        sigma=(max(4.0, H / max(8.0, n_targets * 5.0)),
               max(7.0, W / max(8.0, n_lanes * 3.0))),
    )
    return small - broad


def _tail_score(residual: np.ndarray) -> float:
    """Score the strength and horizontal coherence of positive residuals."""
    positive = np.clip(residual, 0.0, None)
    if not np.any(positive):
        return 0.0
    cutoff = float(np.percentile(positive, 97.0))
    strong = positive[positive >= cutoff]
    strength = float(np.mean(strong)) if strong.size else 0.0

    # Western bands are horizontally coherent.  Averaging across a short
    # horizontal window preserves a band but suppresses point artefacts.
    horizontal = gaussian_filter(positive, sigma=(1.0, 4.0))
    hcut = float(np.percentile(horizontal, 97.0))
    coherent = horizontal[horizontal >= hcut]
    coherence = float(np.mean(coherent)) if coherent.size else 0.0
    return strength + coherence


def infer_band_polarity(
    gray: np.ndarray,
    n_lanes: int,
    n_targets: int,
) -> tuple[bool, float]:
    """Return ``(bands_are_dark, confidence)`` from image morphology.

    Confidence is a 0..1 separation score.  Low confidence is not an error; the
    app exposes an explicit polarity override for ambiguous images.
    """
    arr = _finite_image(gray)
    norm = _normalize(arr)
    light_residual = _enhance(norm, n_lanes, n_targets)
    light_score = _tail_score(light_residual)
    dark_score = _tail_score(-light_residual)
    denom = max(light_score, dark_score, np.finfo(float).eps)
    confidence = float(np.clip(abs(dark_score - light_score) / denom, 0.0, 1.0))
    return bool(dark_score >= light_score), confidence


def _unknown_projection_centers(
    enhanced: np.ndarray,
    *,
    axis: int,
    max_count: int,
    min_structure_ratio: float = 6.5,
) -> tuple[tuple[int, ...], float]:
    """Infer an unknown number of repeated band centres from one projection."""
    length = enhanced.shape[axis]
    profile = np.mean(np.clip(enhanced, 0.0, None), axis=1 - axis)
    sigma = max(1.0, length / max(max_count * 13.0, 1.0))
    profile = gaussian_filter1d(profile, sigma=sigma)
    dynamic = float(np.ptp(profile))
    if dynamic <= np.finfo(float).eps:
        return (length // 2,), 0.0
    lower = profile[profile <= np.percentile(profile, 60.0)]
    projection_noise = _robust_noise(lower if lower.size else profile)
    structure_ratio = dynamic / max(projection_noise, np.finfo(float).eps)
    if structure_ratio < min_structure_ratio:
        return (length // 2,), 0.0

    # The separation limit deliberately merges close shoulders. They can still
    # be recovered later as multiple bands inside one target/lane cell, whereas
    # treating a shoulder as a new target row corrupts the entire raw grid.
    distance = max(3, int(round(length / max(max_count * 1.25, 1.0))))
    peaks, properties = find_peaks(profile, distance=distance, prominence=0.0)
    if not len(peaks):
        return (int(np.argmax(profile)),), 0.0
    prominences = properties.get("prominences")
    if prominences is None or len(prominences) != len(peaks):
        prominences = peak_prominences(profile, peaks)[0]
    floor = float(np.percentile(profile, 10.0))
    excess = np.maximum(profile[peaks] - floor, 0.0)
    maximum_prominence = max(float(np.max(prominences)), np.finfo(float).eps)
    maximum_excess = max(float(np.max(excess)), np.finfo(float).eps)

    credible = (
        (prominences >= maximum_prominence * 0.055)
        & (excess >= maximum_excess * 0.055)
    )
    candidates = peaks[credible]
    candidate_prominences = prominences[credible]
    candidate_excess = excess[credible]
    if not len(candidates):
        strongest = int(np.argmax(prominences + excess))
        candidates = peaks[[strongest]]
        candidate_prominences = prominences[[strongest]]
        candidate_excess = excess[[strongest]]

    # A cropped scan can contain a real band close to an edge, so edge peaks
    # are not prohibited. Only discard a weak edge speck when stronger internal
    # peaks establish a credible blot grid.
    edge_margin = max(2, int(round(length * 0.018)))
    internal = (candidates >= edge_margin) & (candidates < length - edge_margin)
    merit = candidate_prominences / maximum_prominence
    merit += candidate_excess / maximum_excess
    if np.any(internal):
        keep = internal | (merit >= 0.55)
        candidates = candidates[keep]
        merit = merit[keep]

    if len(candidates) > max_count:
        chosen = np.argsort(merit)[-max_count:]
        candidates = candidates[chosen]
        merit = merit[chosen]
    order = np.argsort(candidates)
    candidates = candidates[order]
    merit = merit[order]
    structure_confidence = float(np.clip((structure_ratio - 4.0) / 12.0, 0.0, 1.0))
    confidence = float(np.clip(np.median(merit) / 2.0, 0.0, 1.0))
    confidence *= structure_confidence
    return tuple(int(value) for value in candidates), confidence


def _target_guided_lane_centers(
    signal: np.ndarray,
    target_centers: tuple[int, ...],
    *,
    max_count: int,
) -> tuple[tuple[int, ...], float]:
    """Infer broad lane centres from narrow strips around trusted target rows.

    Averaging a sparse one-row blot over the full image height can make its lane
    projection weaker than dust or scanner-edge artefacts.  Within the target
    row, however, broad flat-topped bands form well separated one-dimensional
    components.  Their intensity-weighted centroids are substantially more
    stable than local maxima, which often land on one shoulder of a saturated
    band.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 2 or not target_centers:
        return (), 0.0
    height, width = values.shape
    if width < 3:
        return (), 0.0

    if len(target_centers) > 1:
        row_gap = float(np.median(np.diff(target_centers)))
        half_window = min(height * 0.035, row_gap * 0.20)
    else:
        half_window = height * 0.035
    half_window = max(2, int(round(half_window)))

    row_profiles: list[np.ndarray] = []
    for center in target_centers:
        y0 = max(0, int(center) - half_window)
        y1 = min(height, int(center) + half_window + 1)
        if y1 <= y0:
            continue
        profile = np.percentile(values[y0:y1], 80.0, axis=0)
        profile = gaussian_filter1d(
            profile,
            sigma=max(0.8, width / max(max_count * 100.0, 1.0)),
        )
        baseline = float(np.percentile(profile, 20.0))
        profile = np.clip(profile - baseline, 0.0, None)
        scale = float(np.percentile(profile, 99.5))
        if scale <= np.finfo(float).eps:
            continue
        row_profiles.append(np.clip(profile / scale, 0.0, 1.0))
    if not row_profiles:
        return (), 0.0

    guided_profile = np.mean(np.stack(row_profiles), axis=0)
    component_labels, component_count = label(guided_profile >= 0.35)
    components: list[tuple[float, int, float, float]] = []
    for component_id in range(1, component_count + 1):
        indices = np.flatnonzero(component_labels == component_id)
        if len(indices) < 2:
            continue
        weights = guided_profile[indices]
        total = float(np.sum(weights))
        if total <= np.finfo(float).eps:
            continue
        center = float(np.sum(indices * weights) / total)
        components.append((center, len(indices), float(np.max(weights)), total))
    if not components:
        return (), 0.0

    if len(components) > max_count:
        components = sorted(components, key=lambda item: item[3])[-max_count:]
    components.sort(key=lambda item: item[0])
    centers = tuple(int(round(item[0])) for item in components)
    widths = np.asarray([item[1] for item in components], dtype=float)
    peak_strength = float(np.median([item[2] for item in components]))

    if len(centers) > 1:
        gaps = np.diff(centers).astype(float)
        typical_gap = max(float(np.median(gaps)), 1.0)
        gap_deviation = float(np.median(np.abs(gaps - typical_gap))) / typical_gap
        spacing_confidence = float(np.clip(1.0 - 3.0 * gap_deviation, 0.0, 1.0))
    else:
        spacing_confidence = 0.45
    typical_width = max(float(np.median(widths)), 1.0)
    width_deviation = float(np.median(np.abs(widths - typical_width))) / typical_width
    width_confidence = float(np.clip(1.0 - 2.0 * width_deviation, 0.0, 1.0))
    confidence = peak_strength * (
        0.55 + 0.30 * spacing_confidence + 0.15 * width_confidence
    )
    if len(centers) == 1 and widths[0] > width * 0.35:
        confidence *= 0.25
    return centers, float(np.clip(confidence, 0.0, 1.0))


def infer_raw_grid(
    gray: np.ndarray,
    *,
    max_lanes: int = 16,
    max_targets: int = 16,
) -> RawGridInference:
    """Infer a target-row/lane grid directly from an unlabeled raw scan.

    This is a fallback for metadata discovery only. Once neutral labels and
    centre hints have been created, the normal detector still performs the
    full per-cell faint-band, overlap, and border analysis.
    """
    if max_lanes < 1 or max_targets < 1:
        raise ValueError("Raw-grid limits must be positive.")
    arr = _finite_image(gray)
    norm = _normalize(arr)
    bands_are_dark, polarity_confidence = infer_band_polarity(arr, 3, 3)
    signal = 1.0 - norm if bands_are_dark else norm
    enhanced = np.clip(_enhance(signal, 4, 4), 0.0, None)
    masked_enhanced = enhanced.copy()
    masked_enhanced[_line_artifact_mask(enhanced, 3, 3)] = 0.0

    structure_floor = 12.0 if polarity_confidence < 0.08 else 6.5
    masked_targets, masked_target_confidence = _unknown_projection_centers(
        masked_enhanced, axis=0, max_count=max_targets,
        min_structure_ratio=structure_floor)
    unmasked_targets, unmasked_target_confidence = _unknown_projection_centers(
        enhanced, axis=0, max_count=max_targets,
        min_structure_ratio=structure_floor)

    # A row of broad, closely spaced real bands can satisfy the horizontal-line
    # opening and disappear from the masked projection.  Prefer the unmasked row
    # only when its structure is materially more convincing; ordinary bordered
    # multi-row panels continue to use the artefact-cleaned inference.
    prefer_unmasked_targets = (
        unmasked_target_confidence >= 0.25
        and unmasked_target_confidence
        > masked_target_confidence + max(0.10, masked_target_confidence * 0.25)
    )
    if prefer_unmasked_targets:
        target_centers = unmasked_targets
        target_confidence = unmasked_target_confidence
    else:
        target_centers = masked_targets
        target_confidence = masked_target_confidence

    lane_centers, lane_confidence = _unknown_projection_centers(
        masked_enhanced, axis=1, max_count=max_lanes,
        min_structure_ratio=structure_floor)
    guided_lanes, guided_lane_confidence = _target_guided_lane_centers(
        signal, target_centers, max_count=max_lanes)
    if (
        guided_lanes
        and guided_lane_confidence >= 0.45
        and guided_lane_confidence > lane_confidence + 0.08
    ):
        lane_centers = guided_lanes
        lane_confidence = guided_lane_confidence
    if (polarity_confidence < 0.04
            and (target_confidence == 0.0 or lane_confidence == 0.0)):
        target_centers = (arr.shape[0] // 2,)
        lane_centers = (arr.shape[1] // 2,)
        target_confidence = lane_confidence = 0.0
    grid_confidence = float(np.sqrt(target_confidence * lane_confidence))
    warning = ""
    if grid_confidence < 0.18:
        warning = (
            "The raw grid is faint or irregular; neutral labels were added, "
            "so review the inferred boxes before quantifying."
        )
    return RawGridInference(
        lane_centers=lane_centers,
        target_centers=target_centers,
        bands_are_dark=bands_are_dark,
        polarity_confidence=polarity_confidence,
        grid_confidence=grid_confidence,
        warning=warning,
    )


def _projection_centers(
    enhanced: np.ndarray,
    *,
    axis: int,
    count: int,
) -> list[int]:
    """Locate a requested number of well-separated projection peaks."""
    length = enhanced.shape[axis]
    if count == 1:
        profile = np.mean(np.clip(enhanced, 0.0, None), axis=1 - axis)
        return [int(np.argmax(profile))]

    profile = np.mean(np.clip(enhanced, 0.0, None), axis=1 - axis)
    # Lane projections need substantially more horizontal smoothing than row
    # projections.  This preserves broad bands but suppresses the thin vertical
    # borders commonly left around cropped blot panels (and narrow label glyphs).
    # Without it, a panel edge can be selected as a lane centre.
    divisor = 12.0 if axis == 1 else 70.0
    profile = gaussian_filter1d(profile, sigma=max(1.0, length / (count * divisor)))
    distance = max(2, int(length / (count * 2.4)))
    peaks, props = find_peaks(profile, distance=distance, prominence=0.0)

    if len(peaks) >= count:
        prominences = props.get("prominences")
        if prominences is None or len(prominences) != len(peaks):
            prominences = peak_prominences(profile, peaks)[0]
        # Peak height matters for a broad low-prominence band, while prominence
        # stops a global gradient from dominating the choice.
        merit = prominences + np.maximum(profile[peaks] - np.median(profile), 0.0)
        chosen = peaks[np.argsort(merit)[-count:]]
        return sorted(int(v) for v in chosen)

    # Sparse/very faint images may not expose enough formal local maxima.  A
    # weighted 1-D k-means fallback still follows the foreground mass and is
    # much safer than silently reverting to fixed, evenly spaced boxes.
    weights = np.maximum(profile - np.percentile(profile, 20.0), 0.0)
    coords = np.arange(length, dtype=float)
    if float(weights.sum()) <= 0:
        return list(np.linspace(length * 0.15, length * 0.85, count).astype(int))

    cumulative = np.cumsum(weights)
    quantiles = (np.arange(count) + 0.5) / count * cumulative[-1]
    centers = np.interp(quantiles, cumulative, coords)
    for _ in range(30):
        labels = np.argmin(np.abs(coords[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for i in range(count):
            mask = labels == i
            total = float(weights[mask].sum())
            if total > 0:
                updated[i] = float(np.sum(coords[mask] * weights[mask]) / total)
        if np.max(np.abs(updated - centers)) < 0.1:
            centers = updated
            break
        centers = updated
    return sorted(int(round(v)) for v in centers)


def _cell_bounds(centers: list[int], limit: int) -> list[tuple[int, int]]:
    if len(centers) == 1:
        return [(0, limit)]
    cuts = [0]
    cuts.extend(int(round((a + b) / 2)) for a, b in zip(centers[:-1], centers[1:]))
    cuts.append(limit)
    return [(cuts[i], cuts[i + 1]) for i in range(len(centers))]


def _lane_cell_bounds(centers: list[int], limit: int) -> list[tuple[int, int]]:
    """Bound lane cells to the repeated blot area, excluding side labels.

    Figures exported from papers often place protein names immediately to the
    right of a two-lane blot.  Extending the last lane cell to the image edge
    therefore makes letters look like bands.  The inter-lane distance provides
    a scale-free estimate of the actual panel width.
    """
    if len(centers) < 2:
        return _cell_bounds(centers, limit)
    spacing = float(np.median(np.diff(centers)))
    outer = max(3.0, 0.48 * spacing)
    cuts = [max(0, int(round(centers[0] - outer)))]
    cuts.extend(int(round((a + b) / 2)) for a, b in zip(centers[:-1], centers[1:]))
    cuts.append(min(limit, int(round(centers[-1] + outer))))
    return [(cuts[i], cuts[i + 1]) for i in range(len(centers))]


def _validated_centers_hint(
    values: list[int] | tuple[int, ...] | None,
    *,
    count: int,
    limit: int,
    label: str,
) -> list[int] | None:
    """Validate optional OCR/grid coordinates before they steer detection."""
    if values is None or len(values) == 0:
        return None
    if len(values) != count:
        raise ValueError(f"{label} centre hint count does not match the experiment setup.")
    try:
        centers = [int(round(float(value))) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} centre hints must be finite pixel positions.") from exc
    if any(not np.isfinite(float(value)) for value in values):
        raise ValueError(f"{label} centre hints must be finite pixel positions.")
    if any(value < 0 or value >= limit for value in centers):
        raise ValueError(f"{label} centre hints fall outside the image.")
    if any(b <= a for a, b in zip(centers[:-1], centers[1:])):
        raise ValueError(f"{label} centre hints must be strictly ordered.")
    return centers


def _excluded_lane_keys(
    excluded_lanes: list[str] | tuple[str, ...] | set[str] | str | None,
) -> set[str]:
    """Return normalized lane names that must never produce measurements."""
    if excluded_lanes is None:
        return set()
    values = (excluded_lanes,) if isinstance(excluded_lanes, str) else excluded_lanes
    return {str(value).strip().casefold() for value in values if str(value).strip()}


def _robust_noise(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    noise = 1.4826 * float(np.median(np.abs(values - median)))
    if noise <= np.finfo(float).eps:
        noise = max(float(np.percentile(values, 75) - np.percentile(values, 25)) / 1.349,
                    np.finfo(float).eps)
    return noise


def _line_artifact_mask(
    positive: np.ndarray,
    n_lanes: int,
    n_targets: int,
) -> np.ndarray:
    """Return a mask for long, continuous panel borders and scanner streaks.

    A real Western band is horizontally elongated, so rejecting every thin
    horizontal object would also reject genuine signal.  Panel borders differ
    because they remain connected across the spaces between multiple lanes.
    Long morphological openings capture that global continuity before the
    image is divided into lane/target cells.
    """
    values = np.asarray(positive, dtype=float)
    H, W = values.shape
    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(np.max(finite)) <= 0:
        return np.zeros((H, W), dtype=bool)

    baseline = float(np.percentile(finite, 50.0))
    lower = finite[finite <= np.percentile(finite, 75.0)]
    noise = _robust_noise(lower if lower.size else finite)
    dynamic = max(float(np.percentile(finite, 99.5)) - baseline, 0.0)
    threshold = baseline + max(2.4 * noise, dynamic * 0.18)
    foreground = values >= threshold

    # With one lane/target, require an especially long line so a broad genuine
    # band cannot be mistaken for a frame.  Multi-lane borders only need to
    # cross several inter-lane gaps to be unambiguously artificial.
    horizontal_fraction = 0.72 if n_lanes <= 1 else 0.40
    vertical_fraction = 0.72 if n_targets <= 1 else 0.40
    horizontal_length = max(12, int(round(W * horizontal_fraction)))
    vertical_length = max(12, int(round(H * vertical_fraction)))
    horizontal = binary_opening(
        foreground, structure=np.ones((1, horizontal_length), dtype=bool))
    vertical = binary_opening(
        foreground, structure=np.ones((vertical_length, 1), dtype=bool))

    # A smeared or overlapping biological band can also be long. Only retain
    # opened components that stay thin in the perpendicular direction; panel
    # rules are line-like, whereas real bands have appreciable thickness.
    horizontal_lines = np.zeros_like(horizontal)
    horizontal_labels, horizontal_count = label(horizontal)
    max_horizontal_thickness = max(
        2, int(round(H / max(n_targets * 16.0, 1.0))))
    for component_id in range(1, horizontal_count + 1):
        component = horizontal_labels == component_id
        yy, xx = np.nonzero(component)
        if len(xx) and int(np.ptp(yy)) + 1 <= max_horizontal_thickness:
            horizontal_lines |= component

    vertical_lines = np.zeros_like(vertical)
    vertical_labels, vertical_count = label(vertical)
    max_vertical_thickness = max(
        2, int(round(W / max(n_lanes * 16.0, 1.0))))
    for component_id in range(1, vertical_count + 1):
        component = vertical_labels == component_id
        yy, xx = np.nonzero(component)
        if len(xx) and int(np.ptp(xx)) + 1 <= max_vertical_thickness:
            vertical_lines |= component

    lines = horizontal_lines | vertical_lines
    if not np.any(lines):
        return lines

    thickness = max(1, int(round(min(H, W) / 180.0)))
    return binary_dilation(lines, iterations=thickness)


def _component_quality(
    positive: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    peak_y: int,
    expected_x: int,
) -> tuple[float, int]:
    """Score locally connected, band-shaped foreground inside a candidate ROI."""
    x, y, w, h = box
    local = np.asarray(positive[y:y + h, x:x + w], dtype=float)
    if local.size == 0 or float(np.max(local)) <= 0:
        return 0.0, 0

    baseline = float(np.percentile(local, 20.0))
    lower = local[local <= np.percentile(local, 55.0)]
    noise = _robust_noise(lower if lower.size else local.ravel())
    dynamic = max(float(np.max(local) - baseline), 0.0)
    weak = local >= baseline + max(1.15 * noise, dynamic * 0.08)
    strong = local >= baseline + max(2.4 * noise, dynamic * 0.24)
    components, count = label(weak)
    strong_labels = set(int(value) for value in np.unique(components[strong]) if value)
    if not strong_labels:
        return 0.15, 0

    expected_local_x = float(expected_x - x)
    expected_local_y = float(peak_y - y)
    candidates: list[tuple[float, int, np.ndarray]] = []
    for component_id in strong_labels:
        mask = components == component_id
        yy, xx = np.nonzero(mask)
        if not len(xx):
            continue
        mass = float(np.sum(local[mask]))
        distance = abs(float(np.mean(xx)) - expected_local_x) / max(w, 1)
        distance += abs(float(np.mean(yy)) - expected_local_y) / max(h, 1)
        candidates.append((mass / (1.0 + distance), component_id, mask))
    if not candidates:
        return 0.15, 0

    _, _, chosen = max(candidates, key=lambda item: item[0])
    yy, xx = np.nonzero(chosen)
    component_w = int(np.ptp(xx)) + 1
    component_h = int(np.ptp(yy)) + 1
    aspect = component_w / max(component_h, 1)
    occupancy = float(chosen.sum()) / max(component_w * component_h, 1)
    horizontal_score = float(np.clip(aspect / 2.0, 0.15, 1.0))
    fill_score = float(np.clip(occupancy / 0.38, 0.15, 1.0))
    centre_distance = abs(float(np.mean(xx)) - expected_local_x) / max(w * 0.5, 1)
    centre_score = float(np.clip(1.0 - centre_distance, 0.1, 1.0))
    quality = horizontal_score * 0.45 + fill_score * 0.25 + centre_score * 0.30
    return float(np.clip(quality, 0.0, 1.0)), len(strong_labels)


def _faint_candidate_confidence(
    positive: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    expected_x: int,
) -> tuple[float, int]:
    """Score a missed grid cell using a deliberately relaxed local test.

    The primary detector remains conservative. This second pass is only used
    inside an already known target/lane cell, where the spatial prior is much
    stronger and a faint but coherent peak should not be treated like an empty
    fallback rectangle.
    """
    x, y, w, h = box
    patch = np.asarray(positive[y:y + h, x:x + w], dtype=float)
    if patch.size == 0 or float(np.max(patch)) <= 0:
        return 0.0, 0

    vertical = gaussian_filter1d(
        patch.mean(axis=1), sigma=max(0.65, h / 45.0))
    baseline = float(np.percentile(vertical, 20.0))
    dynamic = max(float(np.max(vertical) - baseline), 0.0)
    if dynamic <= np.finfo(float).eps:
        return 0.0, 0
    profile_noise = max(
        _robust_noise(vertical),
        float(np.percentile(vertical, 75.0) - baseline) * 0.10,
        np.finfo(float).eps,
    )
    centre_lo = max(0, int(round(h * 0.18)))
    centre_hi = min(h, max(centre_lo + 1, int(round(h * 0.82))))
    peak_local = centre_lo + int(np.argmax(vertical[centre_lo:centre_hi]))
    peak_height = max(float(vertical[peak_local] - baseline), 0.0)
    snr = peak_height / profile_noise

    if 0 < peak_local < len(vertical) - 1:
        span = float(peak_widths(vertical, [peak_local], rel_height=0.75)[0][0])
    else:
        span = 1.0
    band_y = y + peak_local
    component_quality, component_count = _component_quality(
        positive, box, peak_y=band_y, expected_x=expected_x)
    centre_offset = abs(peak_local - (h - 1) / 2.0) / max(h * 0.5, 1.0)
    centre_score = float(np.clip(1.0 - centre_offset, 0.2, 1.0))
    thickness_score = float(np.clip(span / max(h * 0.10, 1.0), 0.25, 1.0))
    snr_score = float(1.0 - np.exp(-max(snr - 0.45, 0.0) / 2.2))
    confidence = (
        snr_score * 0.50
        + component_quality * 0.30
        + centre_score * thickness_score * 0.20
    )
    return float(np.clip(confidence, 0.0, 1.0)), component_count


def _deconvolve_overlapping_peak(
    profile: np.ndarray,
    peak: int,
    noise: float,
) -> tuple[list[int], bool]:
    """Use a conservative one-vs-two Gaussian fit for a merged peak.

    Returns new peak centres when two components are sufficiently resolved.
    When a two-component model is strongly preferred but the centres cannot be
    separated reliably, ``unresolved`` is true and the combined peak remains a
    single measurement.
    """
    values = np.asarray(profile, dtype=float)
    if len(values) < 9 or peak <= 1 or peak >= len(values) - 2:
        return [], False
    baseline = float(np.percentile(values, 20.0))
    y = np.clip(values - baseline, 0.0, None)
    amplitude = float(y[peak])
    if amplitude <= max(3.0 * noise, np.finfo(float).eps):
        return [], False

    measured_width = float(peak_widths(values, [peak], rel_height=0.5)[0][0])
    if measured_width < max(4.0, len(values) * 0.08):
        return [], False
    sigma0 = max(1.0, measured_width / 2.355)
    shoulder_offset = max(2, int(round(sigma0 * 0.65)))
    if peak - shoulder_offset < 0 or peak + shoulder_offset >= len(y):
        return [], False
    left_shoulder = float(y[peak - shoulder_offset])
    right_shoulder = float(y[peak + shoulder_offset])
    relative_curvature = (
        amplitude - (left_shoulder + right_shoulder) / 2.0) / amplitude
    relative_asymmetry = abs(left_shoulder - right_shoulder) / amplitude
    # A single Gaussian has a distinctly curved, symmetric summit.  Only spend
    # degrees of freedom on a second component when the observed peak has a
    # flat top or a strong shoulder/asymmetry.
    if relative_curvature >= 0.075 and relative_asymmetry <= 0.20:
        return [], False
    x = np.arange(len(values), dtype=float)
    centre_lo = max(0.0, peak - 2.5 * sigma0)
    centre_hi = min(float(len(values) - 1), peak + 2.5 * sigma0)
    sigma_hi = max(2.0, min(float(len(values)) / 2.0, sigma0 * 3.0))

    def one_model(params):
        offset, amp, centre, sigma = params
        return offset + amp * np.exp(-0.5 * ((x - centre) / sigma) ** 2)

    def two_model(params):
        offset, amp1, centre1, sigma1, amp2, centre2, sigma2 = params
        return (offset
                + amp1 * np.exp(-0.5 * ((x - centre1) / sigma1) ** 2)
                + amp2 * np.exp(-0.5 * ((x - centre2) / sigma2) ** 2))

    upper_amp = max(amplitude * 2.5, np.finfo(float).eps)
    try:
        one = least_squares(
            lambda params: one_model(params) - y,
            x0=[0.0, amplitude, float(peak), sigma0],
            bounds=([0.0, 0.0, centre_lo, 0.65],
                    [upper_amp, upper_amp, centre_hi, sigma_hi]),
            max_nfev=350,
        )
        separation0 = max(1.2, sigma0 * 0.75)
        two = least_squares(
            lambda params: two_model(params) - y,
            x0=[0.0, amplitude * 0.58, max(centre_lo, peak - separation0), sigma0 * 0.75,
                amplitude * 0.58, min(centre_hi, peak + separation0), sigma0 * 0.75],
            bounds=([0.0, 0.0, centre_lo, 0.65, 0.0, centre_lo, 0.65],
                    [upper_amp, upper_amp, centre_hi, sigma_hi,
                     upper_amp, centre_hi, sigma_hi]),
            max_nfev=650,
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return [], False
    if not one.success or not two.success:
        return [], False

    n = max(len(values), 1)
    rss_one = max(float(np.sum(one.fun ** 2)), np.finfo(float).eps)
    rss_two = max(float(np.sum(two.fun ** 2)), np.finfo(float).eps)
    bic_one = n * np.log(rss_one / n) + 4 * np.log(n)
    bic_two = n * np.log(rss_two / n) + 7 * np.log(n)
    if bic_two + 12.0 >= bic_one:
        return [], False

    _, amp1, centre1, sigma1, amp2, centre2, sigma2 = two.x
    components = sorted([(centre1, amp1, sigma1), (centre2, amp2, sigma2)])
    (centre1, amp1, sigma1), (centre2, amp2, sigma2) = components
    if min(amp1, amp2) < max(2.2 * noise, amplitude * 0.16):
        return [], False
    separation = float(centre2 - centre1)
    left_index = int(np.clip(round(centre1), 0, len(y) - 1))
    right_index = int(np.clip(round(centre2), 0, len(y) - 1))
    interior = y[left_index + 1:right_index]
    shoulder_level = min(float(y[left_index]), float(y[right_index]))
    has_observed_valley = bool(
        interior.size and shoulder_level > 0
        and float(np.min(interior)) <= shoulder_level * 0.90)
    resolvable = (
        has_observed_valley
        and separation >= max(3.0, 1.10 * max(sigma1, sigma2))
    )
    if not resolvable:
        return [], True
    centres = sorted(set([int(round(centre1)), int(round(centre2))]))
    return (centres, False) if len(centres) == 2 else ([], True)


def _peak_box(
    enhanced: np.ndarray,
    x_bounds: tuple[int, int],
    y_bounds: tuple[int, int],
    peak_y: int,
    vertical_profile: np.ndarray,
    prominence: float,
    noise: float,
    expected_x: int | None = None,
    model_component: bool = False,
) -> tuple[int, int, int, int, float]:
    """Estimate a padded 2-D rectangle and confidence for a profile peak."""
    H, W = enhanced.shape
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    local_peak = peak_y - y0

    min_h = max(4, int(round(H / 150)))
    max_h = max(min_h, int(round((y1 - y0) * 0.55)))
    if model_component:
        # A fitted Gaussian centre may be a shoulder rather than a formal local
        # maximum.  Keep it centred on the fitted component and let the shared
        # ROI size plus profile-guided integration provide the final extent.
        cy = float(peak_y)
        height = int(np.clip(round((y1 - y0) * 0.18), min_h, max_h))
    else:
        vy = peak_widths(vertical_profile, [local_peak], rel_height=0.75)
        left_y, right_y = float(vy[2][0]) + y0, float(vy[3][0]) + y0
        height = int(np.clip(np.ceil((right_y - left_y) * 1.15), min_h, max_h))
        cy = float((left_y + right_y) / 2)

    by0 = max(0, int(round(cy - height / 2)))
    by1 = min(H, by0 + height)
    band_slice = np.clip(enhanced[by0:by1, x0:x1], 0.0, None)
    horizontal = band_slice.mean(axis=0) if band_slice.size else np.zeros(x1 - x0)
    # Smooth on the lane-cell scale so narrow vertical panel borders and dust
    # specks cannot win over a broad, faint band when choosing its X centre.
    horizontal = gaussian_filter1d(horizontal, sigma=max(0.8, (x1 - x0) / 12.0))
    search_lo, search_hi = 0, len(horizontal)
    if expected_x is not None:
        radius = max(3, int(round((x1 - x0) * 0.38)))
        expected_local = int(round(expected_x - x0))
        search_lo = max(0, expected_local - radius)
        search_hi = min(len(horizontal), expected_local + radius + 1)
    local_peaks, _ = find_peaks(horizontal)
    local_peaks = local_peaks[(local_peaks >= search_lo) & (local_peaks < search_hi)]
    if len(local_peaks):
        local_x = int(local_peaks[np.argmax(horizontal[local_peaks])])
        is_formal_peak = True
    else:
        local_x = search_lo + int(np.argmax(horizontal[search_lo:search_hi]))
        is_formal_peak = False

    if len(horizontal) >= 3 and horizontal[local_x] > 0 and is_formal_peak:
        vx = peak_widths(horizontal, [local_x], rel_height=0.75)
        left_x, right_x = float(vx[2][0]) + x0, float(vx[3][0]) + x0
    else:
        left_x = x0 + (x1 - x0) * 0.3
        right_x = x0 + (x1 - x0) * 0.7

    min_w = max(6, int(round((x1 - x0) * 0.16)))
    max_w = max(min_w, int(round((x1 - x0) * 0.95)))
    width = int(np.clip(np.ceil((right_x - left_x) * 1.12), min_w, max_w))
    cx = float((left_x + right_x) / 2)

    bx0 = max(0, int(round(cx - width / 2)))
    if bx0 + width > W:
        bx0 = max(0, W - width)
    if by0 + height > H:
        by0 = max(0, H - height)

    snr = max(0.0, float(prominence) / max(noise, np.finfo(float).eps))
    aspect = width / max(height, 1)
    shape_factor = float(np.clip(aspect / 2.0, 0.35, 1.0))
    confidence = float(np.clip((1.0 - np.exp(-snr / 3.0)) * shape_factor, 0.0, 1.0))
    return bx0, by0, width, height, confidence


def detect_bands(
    gray: np.ndarray,
    targets: list[str] | tuple[str, ...],
    lanes: list[str] | tuple[str, ...],
    *,
    polarity: str = "auto",
    sensitivity: float = 0.65,
    lane_centers_hint: list[int] | tuple[int, ...] | None = None,
    target_centers_hint: list[int] | tuple[int, ...] | None = None,
    excluded_lanes: list[str] | tuple[str, ...] | set[str] | str | None = None,
) -> DetectionResult:
    """Automatically detect every band and assign it to a target and lane.

    Parameters
    ----------
    gray:
        Native-range, two-dimensional grayscale image.
    targets, lanes:
        Ordered labels (top-to-bottom and left-to-right).  Labels are required
        for biological assignment; their pixel positions are detected.
    polarity:
        ``"auto"``, ``"dark"`` (dark bands on light background), or ``"light"``.
    sensitivity:
        0..1.  Higher values keep fainter peaks but may admit more artefacts.
    lane_centers_hint, target_centers_hint:
        Optional trusted spatial coordinates, typically read from the image's
        panel grid and target-label positions by the local OCR stage.
    excluded_lanes:
        Optional physical lane names, such as a molecular-weight ladder, that
        remain part of lane-centre geometry but never receive band proposals.
    """
    arr = _finite_image(gray)
    targets = [str(v) for v in targets]
    lanes = [str(v) for v in lanes]
    excluded_lane_keys = _excluded_lane_keys(excluded_lanes)
    quantifiable_lanes = [
        lane for lane in lanes if lane.strip().casefold() not in excluded_lane_keys
    ]
    if not targets or not lanes:
        raise ValueError("At least one target and one lane are required.")
    if polarity not in {"auto", "dark", "light"}:
        raise ValueError("polarity must be 'auto', 'dark', or 'light'.")
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))

    if polarity == "auto":
        bands_are_dark, polarity_confidence = infer_band_polarity(
            arr, len(lanes), len(targets))
    else:
        bands_are_dark = polarity == "dark"
        polarity_confidence = 1.0

    norm = _normalize(arr)
    signal = 1.0 - norm if bands_are_dark else norm
    residual = _enhance(signal, len(lanes), len(targets))
    positive = np.clip(residual, 0.0, None)
    line_mask = _line_artifact_mask(positive, len(lanes), len(targets))
    analysis_positive = positive.copy()
    analysis_positive[line_mask] = 0.0

    lane_centers = _validated_centers_hint(
        lane_centers_hint, count=len(lanes), limit=arr.shape[1], label="Lane")
    if lane_centers is None:
        lane_centers = _projection_centers(analysis_positive, axis=1, count=len(lanes))
    target_centers = _validated_centers_hint(
        target_centers_hint, count=len(targets), limit=arr.shape[0], label="Target")
    if target_centers is None:
        target_centers = _projection_centers(analysis_positive, axis=0, count=len(targets))
    x_cells = _lane_cell_bounds(lane_centers, arr.shape[1])
    y_cells = _cell_bounds(target_centers, arr.shape[0])

    bands: list[DetectedBand] = []
    lane_spacing = (float(np.median(np.diff(lane_centers)))
                    if len(lane_centers) > 1 else float(arr.shape[1]))
    for lane, lane_center, x_bounds in zip(lanes, lane_centers, x_cells):
        if lane.strip().casefold() in excluded_lane_keys:
            continue
        x0, x1 = x_bounds
        # Trim only the extreme cell edges; using most of the cell makes the
        # method tolerant of curved or slightly diagonal lanes.
        xpad = max(0, int((x1 - x0) * 0.04))
        sx0, sx1 = x0 + xpad, x1 - xpad
        lane_profile = analysis_positive[:, sx0:sx1].mean(axis=1)
        lane_profile = gaussian_filter1d(lane_profile, sigma=max(0.7, arr.shape[0] / 650.0))

        for target_index, (target, target_center, y_bounds) in enumerate(
                zip(targets, target_centers, y_cells)):
            y0, y1 = y_bounds
            profile = lane_profile[y0:y1]
            if len(profile) < 3:
                continue
            # Estimate background from the lower half.  A broad strong band can
            # occupy more than half of a small panel; using the full-cell median
            # then places the detection threshold above the band itself.
            baseline = float(np.percentile(profile, 20.0))
            lower_half = profile[profile <= np.percentile(profile, 50.0)]
            noise = _robust_noise(lower_half)
            dynamic = max(float(np.max(profile) - baseline), 0.0)
            # At the default sensitivity, a peak needs roughly 2 robust noise
            # units of height/prominence.  The dynamic-range term prevents a
            # near-flat cell from manufacturing peaks when MAD is tiny.
            noise_factor = 3.1 - 1.8 * sensitivity
            min_height = baseline + noise_factor * noise
            min_prominence = max((2.7 - 1.6 * sensitivity) * noise,
                                 dynamic * (0.14 - 0.08 * sensitivity))
            min_distance = max(3, int((y1 - y0) * (0.08 - 0.035 * sensitivity)))
            peaks, props = find_peaks(
                profile,
                height=min_height,
                prominence=min_prominence,
                distance=min_distance,
            )

            if len(peaks) == 0:
                continue
            prominences = props["prominences"]
            if len(peaks) > 1:
                relative_floor = float(np.max(prominences)) * (0.42 - 0.20 * sensitivity)
                keep = prominences >= relative_floor
                peaks, prominences = peaks[keep], prominences[keep]
            # Limit pathological noisy cells without discarding legitimate
            # doublets/triplets.  The strongest eight per target/lane remain.
            if len(peaks) > 8:
                keep = np.argsort(prominences)[-8:]
                peaks, prominences = peaks[keep], prominences[keep]

            unresolved_peak = False
            deconvolved_peaks: set[int] = set()
            if len(peaks) == 1:
                model_centres, unresolved_peak = _deconvolve_overlapping_peak(
                    profile, int(peaks[0]), noise)
                if model_centres:
                    peaks = np.asarray(model_centres, dtype=int)
                    prominences = np.asarray([
                        max(float(profile[model_peak] - baseline), 2.8 * noise)
                        for model_peak in peaks
                    ])
                    deconvolved_peaks = set(model_centres)

            overlap_peaks: set[int] = set()
            if len(peaks) > 1:
                peak_spans = np.asarray([
                    (max(3.0, (y1 - y0) * 0.18)
                     if int(candidate_peak) in deconvolved_peaks
                     else peak_widths(
                         profile, [candidate_peak], rel_height=0.75)[0][0])
                    for candidate_peak in peaks
                ])
                order = np.argsort(peaks)
                for left_index, right_index in zip(order[:-1], order[1:]):
                    separation = float(peaks[right_index] - peaks[left_index])
                    overlap_limit = 0.52 * float(
                        peak_spans[left_index] + peak_spans[right_index])
                    if separation < max(3.0, overlap_limit):
                        overlap_peaks.update(
                            [int(peaks[left_index]), int(peaks[right_index])])

            for peak, prominence in sorted(zip(peaks, prominences), key=lambda p: p[0]):
                model_component = int(peak) in deconvolved_peaks
                peak_span = (
                    max(3.0, (y1 - y0) * 0.18)
                    if model_component
                    else float(peak_widths(profile, [peak], rel_height=0.75)[0][0])
                )
                # Horizontal rectangle borders form extremely narrow peaks in
                # the vertical profile.  Real bands have a measurable vertical
                # spread, even in small JPEG figures.  Keep a high-sensitivity
                # escape hatch for genuinely razor-thin electrophoresis bands.
                min_span = max(2.2, (y1 - y0) * 0.055)
                if peak_span < min_span and sensitivity < 0.95:
                    continue
                peak_y = y0 + int(peak)
                if len(target_centers) > 1:
                    adjacent_gaps = []
                    if target_index > 0:
                        adjacent_gaps.append(target_centers[target_index] - target_centers[target_index - 1])
                    if target_index + 1 < len(target_centers):
                        adjacent_gaps.append(target_centers[target_index + 1] - target_centers[target_index])
                    row_scale = float(min(adjacent_gaps))
                    allowed_offset = row_scale * (0.30 + 0.08 * sensitivity)
                    if abs(peak_y - target_center) > allowed_offset:
                        continue
                bx, by, bw, bh, confidence = _peak_box(
                    analysis_positive, (sx0, sx1), y_bounds, peak_y, profile,
                    float(prominence), noise, expected_x=lane_center,
                    model_component=model_component,
                )
                # Reject point-like dust unless the user deliberately selected
                # very high sensitivity.  Real blot bands are predominantly
                # horizontal, though diffuse bands may only be mildly so.
                if bw < bh and sensitivity < 0.9:
                    continue
                if (len(lane_centers) > 1 and
                        abs((bx + bw / 2) - lane_center) > lane_spacing * 0.36):
                    continue
                component_quality, component_count = _component_quality(
                    analysis_positive, (bx, by, bw, bh),
                    peak_y=peak_y, expected_x=lane_center)
                if component_quality < 0.18 and sensitivity < 0.9:
                    continue
                confidence = float(np.clip(
                    confidence * (0.58 + 0.42 * component_quality), 0.0, 1.0))
                is_overlap = int(peak) in overlap_peaks
                review_required = bool(
                    confidence < 0.28 or component_quality < 0.38
                    or unresolved_peak or is_overlap)
                if unresolved_peak:
                    status = "unresolved overlap"
                elif is_overlap:
                    status = "overlapping"
                elif review_required:
                    status = "low confidence"
                else:
                    status = "accepted"
                bands.append(DetectedBand(
                    target=target, lane=lane, x=bx, y=by, w=bw, h=bh,
                    confidence=round(confidence, 3),
                    source=("deconvolved" if int(peak) in deconvolved_peaks
                            else "detected"),
                    status=status,
                    review_required=review_required,
                    component_count=component_count,
                    profile_span=round(float(peak_span), 3),
                ))

    # A fully merged doublet may look like one smooth Gaussian.  It cannot be
    # deconvolved honestly from that cell alone, but it often remains much
    # broader than bands elsewhere on the same blot.  Flag only strong robust
    # outliers; do not split them or silently assign separate quantities.
    spans = np.asarray([band.profile_span for band in bands if band.profile_span > 0])
    if len(spans) >= 4:
        typical_span = float(np.median(spans))
        span_noise = _robust_noise(spans)
        broad_limit = max(typical_span * 1.35, typical_span + 3.5 * span_noise)
        for band in bands:
            if band.status == "accepted" and band.profile_span > broad_limit:
                band.status = "unresolved overlap"
                band.review_required = True

    _standardize_boxes(bands, arr.shape)
    bands.sort(key=lambda b: (targets.index(b.target), lanes.index(b.lane), b.y, b.x))
    warnings: list[str] = []
    expected = len(targets) * len(quantifiable_lanes)
    if not bands and expected:
        warnings.append("No bands passed the automatic detector; try higher sensitivity or override polarity.")
    elif len(bands) < expected:
        warnings.append(
            f"Detected {len(bands)} bands for {expected} target/lane cells; empty cells may be true negatives or faint bands."
        )
    if polarity == "auto" and polarity_confidence < 0.12:
        warnings.append("Automatic polarity is uncertain; verify the overlay or choose a polarity explicitly.")
    review_count = sum(band.review_required for band in bands)
    if review_count:
        warnings.append(
            f"{review_count} detected band(s) need review because they are faint or overlapping."
        )

    return DetectionResult(
        bands=bands,
        lane_centers=lane_centers,
        target_centers=target_centers,
        bands_are_dark=bands_are_dark,
        polarity_confidence=polarity_confidence,
        warnings=warnings,
    )


def ensure_complete_grid(
    result: DetectionResult,
    targets: list[str] | tuple[str, ...],
    lanes: list[str] | tuple[str, ...],
    image_shape: tuple[int, int],
    *,
    gray: np.ndarray | None = None,
    sensitivity: float = 0.65,
    excluded_lanes: list[str] | tuple[str, ...] | set[str] | str | None = None,
) -> int:
    """Add a reviewable equal-size ROI to every otherwise empty grid cell.

    A missing signal peak must not make a target/lane measurement disappear.
    Low-confidence fallback rectangles are centred on the OCR-derived grid and
    remain editable in the same way as directly detected bands. Physical lanes
    named by ``excluded_lanes`` retain their centres but cannot retain or gain a
    proposal.
    """
    targets = [str(value) for value in targets]
    lanes = [str(value) for value in lanes]
    excluded_lane_keys = _excluded_lane_keys(excluded_lanes)
    H, W = image_shape
    if len(result.target_centers) != len(targets) or len(result.lane_centers) != len(lanes):
        raise ValueError("Detected grid dimensions do not match the experiment setup.")

    # Other proposal backends can feed this completion pass. Enforce the same
    # fail-closed ladder rule even if such a backend emitted an excluded lane.
    result.bands = [
        band for band in result.bands
        if band.lane.strip().casefold() not in excluded_lane_keys
    ]

    if result.bands:
        shared_w, shared_h = result.bands[0].w, result.bands[0].h
    else:
        lane_scale = (float(np.median(np.diff(result.lane_centers)))
                      if len(result.lane_centers) > 1 else W * 0.55)
        target_scale = (float(np.median(np.diff(result.target_centers)))
                        if len(result.target_centers) > 1 else H * 0.35)
        shared_w = int(np.clip(round(lane_scale * 0.72), 6, W))
        shared_h = int(np.clip(round(target_scale * 0.34), 4, H))

    recovery_positive: np.ndarray | None = None
    if gray is not None:
        recovery_arr = _finite_image(gray)
        if recovery_arr.shape != (H, W):
            raise ValueError("Recovery image dimensions do not match the detected grid.")
        recovery_norm = _normalize(recovery_arr)
        recovery_signal = (1.0 - recovery_norm
                           if result.bands_are_dark else recovery_norm)
        recovery_residual = _enhance(recovery_signal, len(lanes), len(targets))
        recovery_positive = np.clip(recovery_residual, 0.0, None)
        recovery_positive[_line_artifact_mask(
            recovery_positive, len(lanes), len(targets))] = 0.0

    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    recovery_threshold = 0.25 - 0.10 * sensitivity
    occupied = {(band.target, band.lane) for band in result.bands}
    added = 0
    recovered = 0
    excluded = 0
    for target, cy in zip(targets, result.target_centers):
        for lane, cx in zip(lanes, result.lane_centers):
            if lane.strip().casefold() in excluded_lane_keys:
                continue
            if (target, lane) in occupied:
                continue
            x = int(np.clip(round(cx - shared_w / 2.0), 0, W - shared_w))
            y = int(np.clip(round(cy - shared_h / 2.0), 0, H - shared_h))
            confidence, component_count = 0.0, 0
            if recovery_positive is not None:
                confidence, component_count = _faint_candidate_confidence(
                    recovery_positive, (x, y, shared_w, shared_h),
                    expected_x=int(cx))
            auto_include = confidence >= recovery_threshold
            if auto_include:
                recovered += 1
            else:
                excluded += 1
            result.bands.append(DetectedBand(
                target=target,
                lane=lane,
                x=x,
                y=y,
                w=shared_w,
                h=shared_h,
                confidence=round(confidence, 3),
                use=auto_include,
                source="faint recovery" if auto_include else "fallback",
                status="faint — review" if auto_include else "missing — review",
                review_required=True,
                component_count=component_count,
                profile_span=0.0,
            ))
            occupied.add((target, lane))
            added += 1

    _standardize_boxes(result.bands, image_shape)
    result.bands.sort(key=lambda band: (
        targets.index(band.target), lanes.index(band.lane), band.y, band.x))
    if added:
        result.warnings = [warning for warning in result.warnings
                           if not warning.startswith("Detected ")]
        if recovered:
            result.warnings.append(
                f"Recovered {recovered} probable faint band(s) with relaxed local scoring; "
                "they remain active but are marked for review."
            )
        if excluded:
            result.warnings.append(
                f"Excluded {excluded} empty low-confidence ROI(s); enable them manually "
                "only if the profile confirms a real band."
            )
    return added

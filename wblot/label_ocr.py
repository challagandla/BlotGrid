"""Local OCR and spatial classification for blot target/lane labels."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from . import detection, image_io


class OCRUnavailable(RuntimeError):
    """Raised when the optional local OCR runtime is not installed."""


@dataclass(frozen=True)
class TextToken:
    text: str
    score: float
    box: tuple[tuple[float, float], ...]

    @property
    def x0(self) -> float:
        return min(point[0] for point in self.box)

    @property
    def x1(self) -> float:
        return max(point[0] for point in self.box)

    @property
    def y0(self) -> float:
        return min(point[1] for point in self.box)

    @property
    def y1(self) -> float:
        return max(point[1] for point in self.box)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class ExperimentLabels:
    targets: tuple[str, ...]
    lanes: tuple[str, ...]
    target_confidence: float
    lane_confidence: float
    lane_source: str
    tokens: tuple[TextToken, ...]
    warning: str = ""
    blot_bounds: tuple[int, int] | None = None
    lane_centers: tuple[int, ...] = ()
    target_centers: tuple[int, ...] = ()


def _clean_label(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    text = text.strip("|_—–,:;=\"'`~")
    # Keep punctuation used in biological names and condition labels while
    # rejecting OCR fragments made entirely from rules/panel borders.
    if not re.search(r"[A-Za-z0-9Α-Ωα-ω]", text):
        return ""
    return text[:48]


def _deduplicate(tokens: list[TextToken], *, axis: str) -> list[TextToken]:
    tokens = sorted(tokens, key=(lambda item: item.cy) if axis == "y" else (lambda item: item.cx))
    seen: set[str] = set()
    result: list[TextToken] = []
    for token in tokens:
        key = token.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(token)
    return result


def _best_vertical_group(tokens: list[TextToken], width: int, height: int) -> list[TextToken]:
    x_tolerance = max(6.0, width * 0.065)
    candidates: list[tuple[float, list[TextToken]]] = []
    for side in ("left", "right"):
        side_tokens = [
            token for token in tokens
            if ((token.x1 < width * 0.48) if side == "left" else (token.x0 > width * 0.52))
            and (token.x1 - token.x0) < width * 0.46
        ]
        for anchor in side_tokens:
            group = [token for token in side_tokens if abs(token.x0 - anchor.x0) <= x_tolerance]
            group = _deduplicate(group, axis="y")
            if len(group) < 2:
                continue
            y_span = group[-1].cy - group[0].cy
            if y_span < max(height * 0.18, 20.0):
                continue
            alignment = float(np.std([token.x0 for token in group])) / max(width, 1)
            score = len(group) * 2.0 + y_span / height + np.mean([t.score for t in group]) - alignment
            candidates.append((score, group))
    return max(candidates, key=lambda item: item[0])[1] if candidates else []


def _best_horizontal_group(
    tokens: list[TextToken],
    excluded: set[TextToken],
    width: int,
    height: int,
) -> tuple[list[TextToken], list[str]]:
    remaining = [token for token in tokens if token not in excluded]
    y_tolerance = max(6.0, height * 0.055)
    candidates: list[tuple[float, list[TextToken]]] = []
    for region in ("top", "bottom"):
        region_tokens = [
            token for token in remaining
            if (token.cy < height * 0.36 if region == "top" else token.cy > height * 0.64)
        ]
        for anchor in region_tokens:
            group = [token for token in region_tokens if abs(token.y0 - anchor.y0) <= y_tolerance]
            group = _deduplicate(group, axis="x")
            if len(group) < 2:
                continue
            x_span = group[-1].cx - group[0].cx
            if x_span < max(width * 0.18, 20.0):
                continue
            alignment = float(np.std([token.y0 for token in group])) / max(height, 1)
            score = len(group) * 2.0 + x_span / width + np.mean([t.score for t in group]) - alignment
            candidates.append((score, group))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1], []

    # Some OCR engines return a whole lane-label row as one text box.
    for token in sorted(remaining, key=lambda item: item.score, reverse=True):
        near_edge = token.cy < height * 0.30 or token.cy > height * 0.70
        words = [_clean_label(word) for word in re.split(r"\s+", token.text)]
        words = [word for word in words if word]
        if near_edge and 2 <= len(words) <= 12 and (token.x1 - token.x0) > width * 0.20:
            return [], words
    return [], []


def _display_uint8(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray, dtype=float)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def _infer_lane_grid(
    gray: np.ndarray,
    target_tokens: list[TextToken],
) -> tuple[int, tuple[int, ...], tuple[int, int] | None]:
    """Infer repeated unlabeled columns and the bordered blot-panel extent."""
    arr = _display_uint8(gray).astype(float)
    height, width = arr.shape[:2]
    if not target_tokens:
        return 1, (), None

    median_x = float(np.median([token.cx for token in target_tokens]))
    margin = max(1, int(round(width * 0.005)))
    if median_x >= width / 2:
        search_x0, search_x1 = 0, max(1, int(min(token.x0 for token in target_tokens)) - margin)
    else:
        search_x0 = min(width - 1, int(max(token.x1 for token in target_tokens)) + margin)
        search_x1 = width
    search = arr[:, search_x0:search_x1]
    search_width = search.shape[1]
    if search_width < 16:
        return 1, (), None

    # Thin rectangular panel borders repeat through most target rows. Their
    # column-darkness persistence is therefore stronger than individual bands.
    edge_profile = gaussian_filter1d(
        np.percentile(255.0 - search, 65, axis=0), sigma=0.7)
    zone = max(4, int(round(search_width * 0.30)))
    edge_peaks, _ = find_peaks(
        edge_profile,
        distance=max(2, search_width // 80),
        prominence=max(0.5, float(np.ptp(edge_profile)) * 0.008),
    )

    # A dark band near a panel edge can be fractionally stronger than the thin
    # rectangular frame.  Among comparably persistent peaks, choose the
    # outermost one: the frame encloses the bands while remaining well clear of
    # the OCR labels.  This is important for both compact two-lane figures and
    # wide multi-lane TIFFs.
    left_candidates = [int(peak) for peak in edge_peaks if peak < zone]
    left_candidates.append(int(np.argmax(edge_profile[:zone])))
    left_candidates = sorted(set(left_candidates))
    if left_candidates:
        left_strength = max(float(edge_profile[peak]) for peak in left_candidates)
        credible = [peak for peak in left_candidates
                    if edge_profile[peak] >= left_strength * 0.80]
        left_local = min(credible)
    else:
        left_local = int(np.argmax(edge_profile[:zone]))

    right_start = search_width - zone
    right_candidates = [int(peak) for peak in edge_peaks if peak >= right_start]
    right_candidates.append(right_start + int(np.argmax(edge_profile[-zone:])))
    right_candidates = sorted(set(right_candidates))
    if right_candidates:
        right_strength = max(float(edge_profile[peak]) for peak in right_candidates)
        credible = [peak for peak in right_candidates
                    if edge_profile[peak] >= right_strength * 0.80]
        right_local = max(credible)
    else:
        right_local = right_start + int(np.argmax(edge_profile[-zone:]))
    panel_width = right_local - left_local
    # If the two selected peaks enclose only the band centres themselves, this
    # is an unframed blot rather than a trustworthy panel boundary.  Fall back
    # to the full label-free side and let periodicity determine the lane count.
    if panel_width < search_width * 0.60:
        left_local, right_local = 0, search_width - 1
        panel_width = right_local - left_local
    x0 = search_x0 + left_local
    x1 = search_x0 + right_local + 1
    if panel_width < 12:
        return 1, (), (x0, x1)

    row_centres = sorted(token.cy for token in target_tokens)
    gaps = np.diff(row_centres)
    typical_gap = float(np.median(gaps)) if len(gaps) else height * 0.15
    half_window = int(np.clip(round(typical_gap * 0.22), 3, max(3, height * 0.06)))
    bands_are_dark, _ = detection.infer_band_polarity(
        arr[:, x0:x1], n_lanes=2, n_targets=max(1, len(row_centres)))
    profiles: list[np.ndarray] = []
    for centre in row_centres:
        y = int(round(centre))
        strip = arr[max(0, y - half_window):min(height, y + half_window + 1), x0:x1]
        if not strip.size:
            continue
        signal = 255.0 - strip if bands_are_dark else strip
        profiles.append(np.percentile(signal, 70, axis=0))
    if not profiles:
        return 1, (), (x0, x1)

    # Autocorrelation detects the repeating lane pitch even where neighbouring
    # strong bands merge into one broad dark region. Prefer the smallest peak
    # close to the dominant correlation so a two-lane harmonic is not mistaken
    # for half the true lane count.
    profile = np.mean(np.stack(profiles), axis=0)
    narrow_sigma = max(1.2, panel_width * 0.005)
    broad_sigma = max(6.0, panel_width * 0.06)
    periodic = gaussian_filter1d(profile, narrow_sigma) - gaussian_filter1d(profile, broad_sigma)
    periodic -= float(np.mean(periodic))
    correlation = np.correlate(periodic, periodic, mode="full")[len(periodic) - 1:]
    correlation /= np.arange(len(periodic), 0, -1)
    lag_min = max(5, int(round(panel_width / 30.0)))
    lag_max = min(len(correlation) - 1, max(lag_min + 2, int(round(panel_width * 0.80))))
    peaks, _ = find_peaks(correlation[lag_min:lag_max], distance=max(3, lag_min // 2))
    peaks = peaks + lag_min
    positive_peaks = [int(peak) for peak in peaks if correlation[peak] > 0]
    if not positive_peaks:
        return 1, (int(round((x0 + x1 - 1) / 2)),), (x0, x1)
    strongest = max(float(correlation[peak]) for peak in positive_peaks)
    plausible = [peak for peak in positive_peaks if correlation[peak] >= strongest * 0.70]
    spacing = min(plausible) if plausible else max(positive_peaks, key=lambda peak: correlation[peak])
    count = int(np.clip(round(panel_width / max(spacing, 1)), 1, 12))
    centres = tuple(
        int(round(x0 + (index + 0.5) * panel_width / count))
        for index in range(count)
    )
    return count, centres, (x0, x1)


def _estimate_lane_count(gray: np.ndarray, target_tokens: list[TextToken]) -> int:
    return _infer_lane_grid(gray, target_tokens)[0]


def classify_experiment_labels(
    gray: np.ndarray,
    tokens: list[TextToken] | tuple[TextToken, ...],
) -> ExperimentLabels:
    """Classify OCR boxes into vertical target and horizontal lane labels."""
    height, width = np.asarray(gray).shape[:2]
    valid = [token for token in tokens if token.text and token.score >= 0.50]
    targets = _best_vertical_group(valid, width, height)
    lane_tokens, combined_lane_words = _best_horizontal_group(
        valid, set(targets), width, height)

    target_names = tuple(token.text for token in targets)
    if lane_tokens:
        lane_names = tuple(token.text for token in lane_tokens)
        lane_source = "printed"
        lane_confidence = float(np.mean([token.score for token in lane_tokens]))
    elif combined_lane_words:
        lane_names = tuple(combined_lane_words)
        lane_source = "printed"
        lane_confidence = 0.65
    elif target_names:
        lane_count, inferred_centres, blot_bounds = _infer_lane_grid(gray, targets)
        lane_names = tuple(f"Lane {index}" for index in range(1, lane_count + 1))
        lane_source = "inferred"
        lane_confidence = 0.0
    else:
        lane_names = ()
        lane_source = "none"
        lane_confidence = 0.0
        inferred_centres, blot_bounds = (), None

    if lane_tokens or combined_lane_words:
        inferred_centres, blot_bounds = (), None

    return ExperimentLabels(
        targets=target_names,
        lanes=lane_names,
        target_confidence=float(np.mean([token.score for token in targets])) if targets else 0.0,
        lane_confidence=lane_confidence,
        lane_source=lane_source,
        tokens=tuple(valid),
        warning="" if target_names else "No reliable vertical target-label list was found.",
        blot_bounds=blot_bounds,
        lane_centers=inferred_centres,
        target_centers=tuple(int(round(token.cy)) for token in targets),
    )


@lru_cache(maxsize=1)
def _ocr_engine():
    # The bundled project-local runtime is a fallback for workstations whose
    # managed Python installation cannot accept additional packages.
    vendor = Path(__file__).resolve().parents[1] / ".vendor"
    if vendor.is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    try:
        from rapidocr import RapidOCR
    except (ImportError, OSError) as exc:
        raise OCRUnavailable(
            "Local OCR is unavailable. Install rapidocr and onnxruntime, then retry label recognition."
        ) from exc
    # Pin every model to the vendored on-disk copy. Passing explicit paths
    # bypasses RapidOCR's optional model downloader and guarantees that label
    # recognition never needs a network connection.
    model_dir = vendor / "rapidocr" / "models"
    model_paths = {
        "Det.model_path": model_dir / "PP-OCRv6_det_small.onnx",
        "Cls.model_path": model_dir / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "Rec.model_path": model_dir / "PP-OCRv6_rec_small.onnx",
    }
    missing = [path.name for path in model_paths.values() if not path.is_file()]
    if missing:
        raise OCRUnavailable(
            "Local OCR model files are missing: " + ", ".join(missing)
        )
    try:
        return RapidOCR(params={
            **{key: str(path) for key, path in model_paths.items()},
            "Global.log_level": "warning",
        })
    except (ImportError, OSError, RuntimeError) as exc:
        raise OCRUnavailable(f"Local OCR could not start: {exc}") from exc


def read_experiment_labels(image: image_io.LoadedImage) -> ExperimentLabels:
    """Read printed target/lane labels from an uploaded blot, fully locally."""
    display = image_io.to_display(image)
    rgb = np.repeat(display[:, :, None], 3, axis=2)
    try:
        output = _ocr_engine()(rgb)
    except OCRUnavailable:
        raise
    except Exception as exc:
        raise OCRUnavailable(f"Label recognition failed: {exc}") from exc

    boxes = getattr(output, "boxes", None)
    texts = getattr(output, "txts", None)
    scores = getattr(output, "scores", None)
    if boxes is None or texts is None or scores is None:
        return ExperimentLabels((), (), 0.0, 0.0, "none", (), "No readable text was found.")

    tokens: list[TextToken] = []
    for box, text, score in zip(boxes, texts, scores):
        cleaned = _clean_label(text)
        if not cleaned:
            continue
        points = tuple((float(point[0]), float(point[1])) for point in np.asarray(box))
        tokens.append(TextToken(cleaned, float(score), points))
    return classify_experiment_labels(image.gray, tokens)


def infer_raw_scan_labels(
    image: image_io.LoadedImage,
    prior: ExperimentLabels | None = None,
) -> ExperimentLabels:
    """Fill missing metadata from a label-free scan without discarding OCR."""
    grid = detection.infer_raw_grid(image.gray)
    prior_targets = tuple(prior.targets) if prior is not None else ()
    prior_lanes = tuple(prior.lanes) if prior is not None else ()
    targets = prior_targets or tuple(
        f"Target {index}" for index in range(1, len(grid.target_centers) + 1))
    lanes = prior_lanes or tuple(
        f"Lane {index}" for index in range(1, len(grid.lane_centers) + 1))

    prior_target_centers = tuple(getattr(prior, "target_centers", ())) if prior else ()
    prior_lane_centers = tuple(getattr(prior, "lane_centers", ())) if prior else ()
    target_centers = (prior_target_centers
                      if len(prior_target_centers) == len(targets)
                      else grid.target_centers)
    lane_centers = (prior_lane_centers
                    if len(prior_lane_centers) == len(lanes)
                    else grid.lane_centers)
    # If one axis was printed but its OCR centres are unavailable, ask the
    # ordinary detector to locate exactly that many centres instead of passing
    # a mismatched raw hint.
    if len(target_centers) != len(targets):
        target_centers = ()
    if len(lane_centers) != len(lanes):
        lane_centers = ()

    retained = []
    if prior_targets:
        retained.append("target names")
    if prior_lanes:
        retained.append("lane names")
    retained_note = f" Retained recognized {' and '.join(retained)}." if retained else ""
    warning = (
        f"No complete printed label set was found; inferred a "
        f"{len(targets)} target × {len(lanes)} lane raw grid and assigned neutral labels."
        f"{retained_note}"
    )
    if grid.warning:
        warning += f" {grid.warning}"
    return ExperimentLabels(
        targets=targets,
        lanes=lanes,
        target_confidence=(prior.target_confidence if prior_targets and prior else grid.grid_confidence),
        lane_confidence=(prior.lane_confidence if prior_lanes and prior else grid.grid_confidence),
        lane_source="raw_scan",
        tokens=tuple(prior.tokens) if prior is not None else (),
        warning=warning,
        blot_bounds=getattr(prior, "blot_bounds", None) if prior else None,
        lane_centers=lane_centers,
        target_centers=target_centers,
    )


def read_experiment_labels_or_grid(image: image_io.LoadedImage) -> ExperimentLabels:
    """Prefer printed labels, falling back locally to a pixel-only raw grid."""
    try:
        labels = read_experiment_labels(image)
    except OCRUnavailable:
        return infer_raw_scan_labels(image)
    if labels.targets and labels.lanes:
        return labels
    return infer_raw_scan_labels(image, prior=labels)


def suggested_control(targets: list[str] | tuple[str, ...]) -> str:
    """Return a recognized loading control, or blank when none is known."""
    if not targets:
        return ""
    preferred = (
        "gapdh", "actb", "actin", "bactin", "betaactin", "tubb", "tubulin",
        "hprt1", "rplp0", "vinculin", "laminb1", "h3", "h4",
        "histoneh3", "histoneh4", "eef1a1",
    )
    for wanted in preferred:
        for target in targets:
            normalised = re.sub(r"[^a-z0-9]", "", target.casefold())
            without_size = re.sub(r"\d+(?:kda)?$", "", normalised)
            if normalised == wanted or without_size == wanted:
                return target
    return ""

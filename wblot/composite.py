"""Quantify composite Western blot figures made of many framed panels.

A published figure is often not one blot but a grid of small cropped
tiles, each outlined by a thin black frame, with cell-line headers, target
labels and lane labels printed around them.  BlotGrid's whole-image detector
expects one continuous blot, so this module adds a front end that

1. finds every framed tile (closed rectangles of dark line pixels),
2. assigns each tile to a figure row/column by gap clustering of centres,
3. models each tile as two neighbouring, roughly mirror-image bands:

   * the split between the bands is the column-profile valley closest to the
     mirror-symmetry prior (midpoint of the combined band extent), so a
     faint knockdown band beside a strong control stays in its own lane;
   * each band's horizontal extent is the outermost above-threshold span in
     its lane, so a "dumbbell" band (dark ends, pale middle) stays ONE band;
   * vertical extent is the contiguous peak around the band row, so a
     second band stacked above/below (a vertical doublet) is not merged;
   * both lanes get ONE box size, set by the longer and thicker band; the
     smaller band's box covers empty background around it, and no box may
     reach into the other band;
   * an absent band's box is placed at the mirror position of the present
     band about the split,

4. measures every box with an ImageJ "Gel Analyzer"-style peak area (lane
   profile, straight baseline between the valleys either side of the band,
   area above it), plus a rectangle integrated density whose background
   comes from the strips just above and below the box (never from the
   neighbouring band).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d

from .densitometry import measure_band


@dataclass(frozen=True)
class Panel:
    x: int
    y: int
    w: int
    h: int
    row: int = -1
    col: int = -1


@dataclass
class LaneMeasurement:
    row: int
    col: int
    target: str
    lane: str
    lane_index: int
    roi_x: int
    roi_y: int
    roi_w: int
    roi_h: int
    band_x0: int                # detected band extent (absolute pixels)
    band_x1: int
    band_y0: int
    band_y1: int
    split_x: int                # boundary between the two bands
    peak_area: float            # ImageJ-style area above the valley baseline
    net_intden: float           # rectangle IntDen, bg from strips above/below
    background_per_px: float
    baseline_y0: int
    baseline_y1: int
    saturated_fraction: float   # fraction of ROI pixels at 8-bit black
    source: str                 # detected / faint / absent
    status: str
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        row = asdict(self)
        row["notes"] = "; ".join(self.notes)
        return row


# ---------------------------------------------------------------------------
# Panel discovery
# ---------------------------------------------------------------------------

def _line_mask(gray: np.ndarray, dark_threshold: float, min_h: int, min_v: int) -> np.ndarray:
    dark = gray < dark_threshold
    horizontal = ndi.binary_opening(dark, structure=np.ones((1, min_h), bool))
    vertical = ndi.binary_opening(dark, structure=np.ones((min_v, 1), bool))
    return horizontal | vertical


def find_framed_panels(
    gray: np.ndarray,
    *,
    dark_threshold: float = 110.0,
    min_line_h: int = 40,
    min_line_v: int = 20,
    size_range: tuple[int, int, int, int] = (40, 20, 400, 200),
) -> list[Panel]:
    """Return the bounding boxes of closed, dark rectangular frames."""
    lines = _line_mask(gray, dark_threshold, min_line_h, min_line_v)
    filled = ndi.binary_fill_holes(lines)
    labels, _ = ndi.label(filled)
    min_w, min_h, max_w, max_h = size_range
    panels: list[Panel] = []
    for sl in ndi.find_objects(labels):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        w, h = x1 - x0, y1 - y0
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        # A frame is mostly hollow; a solid dark blob (text, a saturated
        # smear) is not a panel.
        if lines[y0:y1, x0:x1].mean() > 0.6:
            continue
        panels.append(Panel(int(x0), int(y0), int(w), int(h)))
    return panels


def _gap_clusters(values: np.ndarray, min_gap: float) -> np.ndarray:
    order = np.argsort(values)
    cluster = np.zeros(len(values), dtype=int)
    current = 0
    for prev, nxt in zip(order[:-1], order[1:]):
        if values[nxt] - values[prev] > min_gap:
            current += 1
        cluster[nxt] = current
    return cluster


def assign_grid(panels: list[Panel]) -> list[Panel]:
    """Assign row/column indices from panel centres (top-left = 0, 0)."""
    if not panels:
        return []
    cx = np.array([p.x + p.w / 2 for p in panels])
    cy = np.array([p.y + p.h / 2 for p in panels])
    min_w = float(np.median([p.w for p in panels]))
    min_h = float(np.median([p.h for p in panels]))
    cols = _gap_clusters(cx, 0.5 * min_w)
    rows = _gap_clusters(cy, 0.5 * min_h)
    out = [Panel(p.x, p.y, p.w, p.h, int(r), int(c)) for p, r, c in zip(panels, rows, cols)]
    return sorted(out, key=lambda p: (p.row, p.col))


def panel_interior(gray: np.ndarray, panel: Panel, extra_inset: int = 1,
                   dark_threshold: float = 110.0) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) inside the frame lines of ``panel``."""
    crop = gray[panel.y:panel.y + panel.h, panel.x:panel.x + panel.w] < dark_threshold

    def edge_depth(lines: np.ndarray) -> int:
        depth = 0
        for line in lines:
            if line.mean() > 0.5:
                depth += 1
            else:
                break
        return depth

    top = edge_depth(crop)
    bottom = edge_depth(crop[::-1])
    left = edge_depth(crop.T)
    right = edge_depth(crop.T[::-1])
    x0 = panel.x + left + extra_inset
    y0 = panel.y + top + extra_inset
    x1 = panel.x + panel.w - right - extra_inset
    y1 = panel.y + panel.h - bottom - extra_inset
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def valley_baseline_area(profile: np.ndarray, peak: int, *, lo: int | None = None,
                         hi: int | None = None) -> tuple[float, int, int]:
    """ImageJ Gel-Analyzer-style peak area.

    Draws a straight baseline between the lowest points of the smoothed
    profile on each side of ``peak`` (searched within ``lo``..``hi``) and
    integrates the profile above it.  Returns (area, left_index, right_index).
    """
    n = len(profile)
    if n < 3:
        return 0.0, 0, max(n - 1, 0)
    smooth = gaussian_filter1d(profile.astype(float), sigma=0.8)
    peak = int(np.clip(peak, 0, n - 1))
    lo = 0 if lo is None else int(np.clip(lo, 0, peak))
    hi = n if hi is None else int(np.clip(hi, peak + 1, n))
    left = lo + int(np.argmin(smooth[lo:peak + 1]))
    right = peak + int(np.argmin(smooth[peak:hi]))
    if right <= left:
        return 0.0, left, right
    xs = np.arange(left, right + 1)
    base = np.interp(xs, [left, right], [smooth[left], smooth[right]])
    area = float(np.clip(profile[left:right + 1] - base, 0.0, None).sum())
    return area, int(left), int(right)


# ---------------------------------------------------------------------------
# Two-band tile model
# ---------------------------------------------------------------------------

@dataclass
class TileSettings:
    """Band-detection settings.  Keep these FIXED across repeat measurements.

    Changing them between repeats changes what is measured (e.g. the box
    height), which is a method change, not a re-measurement; use
    :class:`Redraw` for repeat measurements instead.
    """

    extent_frac: float = 0.25    # band CORE = above this fraction of the band's peak
    tail_frac: float = 0.08      # core grows along faded tails down to this fraction
    guard_frac: float = 0.12     # neighbour band "keep-out" edge (wider)
    pad: int = 1                 # px added on every side of the largest band
    jitter: tuple[int, int] = (0, 0)
    extra_inset: int = 1         # px inside the tile frame


@dataclass(frozen=True)
class Redraw:
    """Box perturbation for one repeat measurement.

    ``shifts`` moves each lane's box independently by (dx, dy) px; ``dw`` and
    ``dh`` change the tile's shared box size (all boxes stay equal in size).
    Band detection, split and baseline window are not re-run, so a repeat
    differs from the nominal measurement only in box placement.
    """

    shifts: tuple[tuple[int, int], ...] = ((0, 0), (0, 0))
    dw: int = 0
    dh: int = 0

    @staticmethod
    def random(rng: np.random.Generator, n_lanes: int = 2, max_px: int = 1) -> "Redraw":
        shifts = tuple(tuple(int(v) for v in rng.integers(-max_px, max_px + 1, size=2))
                       for _ in range(n_lanes))
        dw, dh = (int(v) for v in rng.integers(-max_px, max_px + 1, size=2))
        return Redraw(shifts, dw, dh)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def span_extent(profile: np.ndarray, lo: int, hi: int, frac: float,
                *, min_run: int = 3, keep_frac: float = 0.4) -> tuple[int, int] | None:
    """Outermost above-threshold span of ``profile[lo:hi]``.

    All runs above ``frac`` x peak that are at least ``min_run`` px long and
    reach ``keep_frac`` x peak are merged into one span, so the two dark lobes
    of a dumbbell band (and any pale middle between them) form ONE band.
    Short or weak specks are ignored.
    """
    lo, hi = max(0, lo), min(len(profile), hi)
    seg = profile[lo:hi]
    if seg.size == 0:
        return None
    peak = float(seg.max())
    if peak <= 0:
        return None
    runs = [(a, b) for a, b in _runs(seg >= frac * peak)
            if (b - a) >= min_run and float(seg[a:b].max()) >= keep_frac * peak]
    if not runs:
        k = int(np.argmax(seg))
        return lo + k, lo + k + 1
    return lo + runs[0][0], lo + runs[-1][1]


def contiguous_extent(profile: np.ndarray, peak: int, frac: float) -> tuple[int, int]:
    """Contiguous span around ``peak`` above ``frac`` x peak height.

    The walk also stops where the profile starts rising again, so a second
    band stacked above or below (vertical doublet) is not swallowed.
    """
    top = float(profile[peak])
    thr = frac * top
    tol = 0.05 * max(top, 1e-9)
    a = peak
    while a > 0 and profile[a - 1] >= thr and profile[a - 1] <= profile[a] + tol:
        a -= 1
    b = peak
    while b < len(profile) - 1 and profile[b + 1] >= thr and profile[b + 1] <= profile[b] + tol:
        b += 1
    return a, b + 1


def local_baseline_corrected(profile: np.ndarray, peak: int, lo: int, hi: int) -> np.ndarray:
    """Profile minus a straight baseline between the minima either side of ``peak``.

    This is the ImageJ "draw a line under the peak" step.  It handles shaded or
    gradient tile backgrounds that a single global baseline gets wrong.  Values
    outside the two minima are set to 0 so an extent can never run past them.
    """
    lo, hi = max(0, lo), min(len(profile), hi)
    peak = int(np.clip(peak, lo, hi - 1))
    left = lo + int(np.argmin(profile[lo:peak + 1]))
    right = peak + int(np.argmin(profile[peak:hi]))
    out = np.zeros_like(profile, dtype=float)
    if right <= left:
        out[peak] = max(float(profile[peak]), 0.0)
        return out
    xs = np.arange(left, right + 1)
    base = np.interp(xs, [left, right], [profile[left], profile[right]])
    out[left:right + 1] = np.clip(profile[left:right + 1] - base, 0, None)
    return out


def _detrended(signal: np.ndarray) -> np.ndarray:
    """Signal minus each column's 30th percentile (removes streaks/background)."""
    return signal - np.percentile(signal, 30, axis=0, keepdims=True)


def background_removed(signal: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove shading, streaks and mottled background, keep thin bands.

    Bands are thin horizontal features, while blotchy background patches are
    larger.  A per-column grey opening with a vertical window several band
    thicknesses tall (a 1-D rolling ball) estimates that background; bands
    shorter than the window are kept.  Returns (band signal, pixel noise).
    """
    first = _detrended(signal)
    _, thick = _main_row(first)
    ih = signal.shape[0]
    length = int(np.clip(4 * thick + 1, 9, max(9, ih - 2)))
    smooth = ndi.gaussian_filter(signal.astype(float), 0.7)
    background = ndi.grey_opening(smooth, size=(length, 1))
    background = ndi.uniform_filter1d(background, 3, axis=1)  # no column jitter
    det = signal - background
    resid = det - ndi.median_filter(det, size=(1, 5))
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-6
    return det, noise


def hysteresis_extent(profile: np.ndarray, lo: int, hi: int, core_frac: float,
                      tail_frac: float, floor: float) -> tuple[int, int] | None:
    """Band span = merged core runs, grown along faded tails.

    The core is every run above ``core_frac`` x peak (dumbbell lobes merge
    into one span, short specks are ignored).  The span then grows outward
    while the profile stays above max(``tail_frac`` x peak, ``floor``), so a
    band with one dark end and a long faded tail is captured in full.
    """
    core = span_extent(profile, lo, hi, core_frac)
    if core is None:
        return None
    c0, c1 = core
    peak = float(profile[lo:hi].max())
    low = max(tail_frac * peak, floor)
    while c0 > lo and profile[c0 - 1] >= low:
        c0 -= 1
    while c1 < hi and profile[c1] >= low:
        c1 += 1
    return c0, c1


def _main_row(det: np.ndarray) -> tuple[int, int]:
    """Dominant band row and its (valley-stopped) half-height thickness."""
    prof = gaussian_filter1d(det.mean(axis=1), 0.8)
    r0 = int(np.argmax(prof))
    # Half-height measured above a local baseline, so shading or blotches
    # around a thin band do not make it look thick.
    corrected = local_baseline_corrected(prof, r0, 0, len(prof))
    a, b = contiguous_extent(corrected, r0, 0.5)
    return r0, max(1, b - a)


def find_split(col: np.ndarray, *, window: tuple[float, float] = (0.25, 0.75),
               prior_weight: float = 3.0) -> tuple[int, bool]:
    """Boundary between the two bands of a tile.

    Candidates are positions inside the central ``window``.  Each is scored
    by its *dip*: the profile value divided by the lower of the two band
    maxima on either side (within 0.35 x width).  A real gap between two
    bands scores near 0; a point on a plain ramp from a faint band into a
    strong one scores 1, because one side is never higher than the point
    itself.  The score adds the distance from the mirror-symmetry prior (the
    midpoint of the combined band extent), weighted strongly: a ripple inside
    a faint or dumbbell band can be as deep as a real gap, but a real gap
    lies near the mirror midpoint, so touching bands split at their junction
    and a dumbbell band's pale middle is never chosen.
    Returns (split, has_gap).
    """
    w = len(col)
    c = np.clip(col, 0, None)
    ext = span_extent(c, 0, w, 0.10, keep_frac=0.0)
    prior = (ext[0] + ext[1]) / 2.0 if ext else w / 2.0
    lo, hi = int(window[0] * w), int(window[1] * w)
    reach = max(3, int(0.35 * w))
    xs = np.arange(lo, hi)
    top = float(c.max()) or 1.0
    dip = np.ones(len(xs))
    for k, x in enumerate(xs):
        left = float(c[max(0, x - reach):x].max()) if x > 0 else 0.0
        right = float(c[x + 1:min(w, x + 1 + reach)].max()) if x + 1 < w else 0.0
        denom = min(left, right)
        if denom <= 0.05 * top:
            # One side is empty (absent band): low signal here = free space.
            dip[k] = min(1.0, float(c[x]) / top)
        else:
            dip[k] = min(1.0, float(c[x]) / denom)
    score = dip + prior_weight * np.abs(xs - prior) / w
    split = int(xs[int(np.argmin(score))])
    left_peak = float(col[:split].max()) if split > 0 else 0.0
    right_peak = float(col[split:].max())
    has_gap = col[split] < 0.6 * min(left_peak, right_peak) if min(left_peak, right_peak) > 0 else True
    return split, bool(has_gap)


def find_splits(col: np.ndarray, n: int, *, prior_weight: float = 3.0) -> tuple[list[int], list[bool]]:
    """Boundaries between ``n`` neighbouring bands (n > 2).

    Same scoring as :func:`find_split`, applied per boundary: the prior for
    boundary k is k / n of the way across the combined band extent (lanes of
    one gel are evenly spaced), and a real dip between bands is preferred.
    """
    w = len(col)
    c = np.clip(col, 0, None)
    ext = span_extent(c, 0, w, 0.10, keep_frac=0.0)
    left_edge, right_edge = ext if ext else (0, w)
    pitch = max(4.0, (right_edge - left_edge) / n)
    top = float(c.max()) or 1.0
    reach = max(3, int(0.8 * pitch))
    bounds: list[int] = []
    gaps: list[bool] = []
    prev = 0
    for k in range(1, n):
        prior = left_edge + k * pitch
        lo = max(prev + 2, int(prior - 0.5 * pitch))
        hi = min(w - 1, int(prior + 0.5 * pitch) + 1)
        best, best_score = int(round(prior)), np.inf
        for x in range(lo, max(lo + 1, hi)):
            left = float(c[max(0, x - reach):x].max()) if x > 0 else 0.0
            right = float(c[x + 1:min(w, x + 1 + reach)].max()) if x + 1 < w else 0.0
            denom = min(left, right)
            if denom <= 0.05 * top:
                dip = min(1.0, float(c[x]) / top)
            else:
                dip = min(1.0, float(c[x]) / denom)
            score = dip + prior_weight * abs(x - prior) / w
            if score < best_score:
                best, best_score = x, score
        best = int(np.clip(best, prev + 2, w - 2))
        left_peak = float(c[prev:best].max()) if best > prev else 0.0
        right_peak = float(c[best:min(w, best + reach)].max())
        gaps.append(bool(c[best] < 0.6 * min(left_peak, right_peak))
                    if min(left_peak, right_peak) > 0 else True)
        bounds.append(best)
        prev = best
    return bounds, gaps


@dataclass
class _Band:
    present: str                 # detected / faint / absent
    x0: int = 0
    x1: int = 0
    y0: int = 0
    y1: int = 0
    gx0: int = 0                 # guard (keep-out) extent
    gx1: int = 0
    peak: float = 0.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


def analyse_tile(signal: np.ndarray, n_lanes: int = 2, s: TileSettings | None = None,
                 *, return_allowed: bool = False):
    """Locate the bands of a tile (2 or more lanes) and their common box.

    Returns (bands, split, has_gap, box_w, box_h, notes_per_lane, boxes)
    where boxes are (x, y) top-left corners in tile coordinates; with
    ``return_allowed`` the per-lane x ranges a box may occupy (keep-out of
    the neighbouring bands) are appended.  For two lanes ``split`` is the
    single boundary and ``has_gap`` a bool; for more lanes they are lists
    (one entry per inner boundary).
    """
    if n_lanes < 2:
        raise ValueError("A tile needs at least two lanes.")
    s = s or TileSettings()
    ih, iw = signal.shape
    det, noise_px = background_removed(signal)
    r0, thick = _main_row(det)
    # Band rows: about two half-height thicknesses centred on the band row.
    half_win = max(3, thick)
    ry0, ry1 = max(0, r0 - half_win), min(ih, r0 + half_win + 1)
    col = gaussian_filter1d(det[ry0:ry1].mean(axis=0), 1.0)
    # Tail floor: how high the band-row column profile can rise on background
    # alone.  Pixel noise sets a minimum; blotchy background is measured from
    # band-free row blocks of the same height above and below the band.
    col_noise = 2.0 * noise_px / np.sqrt(max(1, ry1 - ry0))
    tail_floor = 3.0 * col_noise
    band_h = ry1 - ry0
    for b0, b1 in ((0, max(0, ry0 - half_win)), (min(ih, ry1 + half_win), ih)):
        if b1 - b0 >= 3:
            rows = det[b0:b1] if b1 - b0 <= band_h else (
                det[b1 - band_h:b1] if b0 == 0 else det[b0:b0 + band_h])
            bgcol = gaussian_filter1d(rows.mean(axis=0), 1.0)
            tail_floor = max(tail_floor, float(np.percentile(bgcol, 95)))
    # The split uses a taller row window so a band sitting slightly higher or
    # lower than the other (tilt) still shows up in the column profile.
    wide = max(3, 2 * thick)
    col_split = gaussian_filter1d(det[max(0, r0 - wide):min(ih, r0 + wide + 1)].mean(axis=0), 1.0)
    if n_lanes == 2:
        split, has_gap = find_split(col_split)
        segs = [(0, split), (split, iw)]
    else:
        split, has_gap = find_splits(col_split, n_lanes)
        edges = [0, *split, iw]
        segs = list(zip(edges[:-1], edges[1:]))
    # Each lane gets its own band-row window centred on its own band row.
    lane_cols = []
    for a, b in segs:
        lp = gaussian_filter1d(det[:, a:b].mean(axis=1), 0.8) if b > a else np.zeros(ih)
        lo_r, hi_r = max(0, r0 - wide), min(ih, r0 + wide + 1)
        rl = lo_r + int(np.argmax(lp[lo_r:hi_r]))
        lane_cols.append(gaussian_filter1d(
            det[max(0, rl - half_win):min(ih, rl + half_win + 1)].mean(axis=0), 1.0))
    peaks = [float(lc[a:b].max()) if b > a else 0.0 for lc, (a, b) in zip(lane_cols, segs)]
    strongest = max(peaks) if peaks else 0.0

    bands: list[_Band] = []
    for (a, b), pk, col in zip(segs, peaks, lane_cols):
        if strongest <= 0 or pk < 3.0:
            bands.append(_Band("absent", peak=pk))
            continue
        present = "detected" if pk >= 0.12 * strongest else "faint"
        # Horizontal: dark core(s) merged (dumbbell = one band), then grown
        # along faded tails while still above background noise.
        ccol = np.clip(col, 0, None)
        x0, x1 = hysteresis_extent(ccol, a, b, s.extent_frac, s.tail_frac, tail_floor)
        gx0, gx1 = hysteresis_extent(ccol, a, b, s.guard_frac, s.tail_frac, tail_floor)
        # Vertical: this band's own row profile, peak near the tile band row,
        # edges measured above a local straight baseline (gradient-safe).
        rprof = gaussian_filter1d(det[:, x0:x1].mean(axis=1), 0.8)
        lo_r, hi_r = max(0, r0 - 2 * thick), min(ih, r0 + 2 * thick + 1)
        pr = lo_r + int(np.argmax(rprof[lo_r:hi_r]))
        win = max(2 * thick, 6)
        rcor = local_baseline_corrected(rprof, pr, pr - win, pr + win + 1)
        y0, y1 = contiguous_extent(rcor, pr, s.extent_frac)
        bands.append(_Band(present, x0, x1, y0, y1, gx0, gx1, pk))

    # Only clearly visible bands (>= 30% of the stronger one) set the shared
    # box size: "the longer and thicker band decides".  A faint knockdown
    # band's noisy extent must not inflate it.
    sizing = [bd for bd in bands if bd.present != "absent" and bd.peak >= 0.30 * strongest]
    box_w = max(bd.x1 - bd.x0 for bd in sizing) + 2 * s.pad
    box_h = max(bd.y1 - bd.y0 for bd in sizing) + 2 * s.pad
    box_h = int(min(box_h, ih))
    ref = max(sizing, key=lambda bd: bd.peak)
    # A weak band sits level with the reference band unless its own peak row
    # is clearly within the reference band's height.
    # Only truly faint bands (<12%) are snapped; a weak but real band keeps
    # its own row (e.g. a sloping control band lower than the treated band).
    for bd in bands:
        if bd.present == "faint":
            if abs(bd.cy - ref.cy) > (ref.y1 - ref.y0) / 2.0:
                bd.y0, bd.y1 = ref.y0, ref.y1

    # Keep-out: a box may not reach into the OTHER band.  For a clearly
    # visible neighbour that is its band edge (plus 1 px); for a weak
    # neighbour, whose edges are noisy, the split line also counts as room.
    n = len(bands)
    allowed = []
    for i in range(n):
        lo_x, hi_x = 0, iw
        if i > 0 and bands[i - 1].present != "absent":
            left = bands[i - 1]
            lo_x = left.x1 + 1
            if left not in sizing:
                lo_x = min(lo_x, segs[i][0])
        if i < n - 1 and bands[i + 1].present != "absent":
            right = bands[i + 1]
            hi_x = right.x0 - 1
            if right not in sizing:
                hi_x = max(hi_x, segs[i][1])
        allowed.append((max(0, lo_x), min(iw, hi_x)))
    notes: list[list[str]] = [[] for _ in range(n)]
    fit_w = min(hi - lo for lo, hi in allowed)
    if box_w > fit_w:
        for i, bd in enumerate(bands):
            bw_i = bd.x1 - bd.x0
            # Only worth a note when the box misses >5% of the band itself.
            if bd.present != "absent" and fit_w < 0.95 * bw_i:
                notes[i].append(
                    f"box covers {fit_w / bw_i:.0%} of band width ({fit_w}/{bw_i}px): "
                    "widened box would reach the neighbouring band")
        box_w = fit_w
    box_w = int(max(4, box_w))

    # Centres: own band, or the mirror image of the other band about the split.
    centres = []
    for i, bd in enumerate(bands):
        if bd.present != "absent":
            cx, cy = bd.cx, bd.cy
        elif n == 2 and bands[1 - i].present != "absent":
            other = bands[1 - i]
            cx, cy = 2 * split - other.cx, other.cy
        else:
            cx, cy = (segs[i][0] + segs[i][1]) / 2.0, ref.cy
        centres.append((cx, cy))

    dx, dy = s.jitter
    boxes = []
    for i, (cx, cy) in enumerate(centres):
        lo_x, hi_x = allowed[i]
        x = int(round(cx - box_w / 2.0)) + dx
        x = int(np.clip(x, lo_x, max(lo_x, hi_x - box_w)))
        x = int(np.clip(x, 0, iw - box_w))
        y = int(round(cy - box_h / 2.0)) + dy
        y = int(np.clip(y, 0, ih - box_h))
        boxes.append([x, y])
    # Neighbouring boxes must not overlap each other either.
    if n > 2:
        min_gap = min(boxes[i + 1][0] - boxes[i][0] for i in range(n - 1))
        if min_gap < box_w:
            box_w = int(max(4, min_gap))
            for nl in notes:
                nl.append(f"box narrowed to {box_w}px so neighbouring boxes do not overlap")
    overlap = boxes[0][0] + box_w - boxes[1][0] if n == 2 else 0
    if overlap > 0:
        shift_l = min(overlap // 2 + overlap % 2, boxes[0][0] - allowed[0][0])
        boxes[0][0] -= max(0, shift_l)
        overlap = boxes[0][0] + box_w - boxes[1][0]
        if overlap > 0:
            room = (allowed[1][1] - box_w) - boxes[1][0]
            boxes[1][0] += max(0, min(overlap, room))
        overlap = boxes[0][0] + box_w - boxes[1][0]
        if overlap > 0:  # still no room: shrink the common width
            box_w -= overlap
            boxes[1][0] = boxes[0][0] + box_w
            for n in notes:
                n.append(f"box narrowed to {box_w}px so the two boxes do not overlap")
    out = (bands, split, has_gap, box_w, box_h, notes, [tuple(b) for b in boxes])
    return (*out, allowed) if return_allowed else out


def redraw_boxes(boxes, box_w: int, box_h: int, allowed, tile_w: int, tile_h: int,
                 redraw: Redraw) -> tuple[list[tuple[int, int]], int, int]:
    """Apply one hand-redraw perturbation while keeping every tile rule.

    Both boxes keep one shared size, stay inside their allowed (keep-out)
    ranges and never overlap; the size change is centred on the nominal box.
    """
    fit_w = min(hi - lo for lo, hi in allowed)
    w2 = int(np.clip(box_w + redraw.dw, 4, max(4, fit_w)))
    h2 = int(np.clip(box_h + redraw.dh, 4, tile_h))
    new = []
    for (x, y), (lo_x, hi_x), (dx, dy) in zip(boxes, allowed, redraw.shifts):
        nx = x - (w2 - box_w) // 2 + dx
        nx = int(np.clip(nx, lo_x, max(lo_x, hi_x - w2)))
        nx = int(np.clip(nx, 0, tile_w - w2))
        ny = y - (h2 - box_h) // 2 + dy
        ny = int(np.clip(ny, 0, tile_h - h2))
        new.append((nx, ny))
    gaps = [new[i + 1][0] - new[i][0] for i in range(len(new) - 1)]
    if gaps and min(gaps) < w2:
        w2 = min(gaps)
        if w2 < 4:  # cannot redraw without overlap: keep the nominal boxes
            return list(boxes), box_w, box_h
    return new, w2, h2


def quantify_panel(
    gray: np.ndarray,
    panel: Panel,
    target: str,
    lanes: list[str],
    *,
    settings: TileSettings | None = None,
    redraw: Redraw | None = None,
) -> list[LaneMeasurement]:
    """Measure the two bands of one framed tile with one shared box size.

    Band geometry is detected once with ``settings``.  A ``redraw`` perturbs
    box position and size for a repeat measurement; the baseline window stays
    anchored to the detected band.
    """
    s = settings or TileSettings()
    ix, iy, iw, ih = panel_interior(gray, panel, extra_inset=s.extra_inset)
    crop = gray[iy:iy + ih, ix:ix + iw]
    signal = 255.0 - crop  # dark bands on light background
    bands, split, has_gap, nbw, nbh, notes, nboxes, allowed = analyse_tile(
        signal, len(lanes), s, return_allowed=True)
    if redraw is not None:
        boxes, bw, bh = redraw_boxes(nboxes, nbw, nbh, allowed, iw, ih, redraw)
    else:
        boxes, bw, bh = nboxes, nbw, nbh

    results: list[LaneMeasurement] = []
    for li, lane in enumerate(lanes):
        bd = bands[li]
        rx, ry = boxes[li]
        ny = nboxes[li][1]
        lane_notes = list(notes[li])
        status = {"detected": "accepted",
                  "faint": "faint band (<12% of the other band)",
                  "absent": "no band; box at mirror position"}[bd.present]
        if bd.present == "absent":
            lane_notes.append("no band above background; box mirrored from the other band")
        if isinstance(split, list):  # >2 lanes: report the nearest inner boundary
            lane_split = split[min(li, len(split) - 1)]
            lane_gap = has_gap[min(li, len(has_gap) - 1)]
        else:
            lane_split, lane_gap = split, has_gap
        if not lane_gap:
            lane_notes.append("bands touch (no gap); split by mirror symmetry")

        # (a) ImageJ-style peak area on the box's vertical profile.
        prof = signal[:, rx:rx + bw].mean(axis=1)
        sm = gaussian_filter1d(prof, 0.8)
        peak = ry + int(np.argmax(sm[ry:ry + bh]))
        # Valleys are searched half a box height beyond the NOMINAL box, so a
        # tight box still gets a baseline under the whole band and a redrawn
        # box does not move the baseline window (which would change what is
        # measured rather than re-measure it).
        reach = max(3, nbh // 2)
        area, b0, b1 = valley_baseline_area(prof, peak, lo=max(0, ny - reach),
                                            hi=min(ih, ny + nbh + reach))
        area *= bw  # mean-profile area -> integrated-density units

        # (b) Rectangle IntDen; background = median of 3-px strips directly
        # above and below the box (same columns), never the neighbouring band.
        strips = [signal[max(0, ry - 3):ry, rx:rx + bw],
                  signal[ry + bh:min(ih, ry + bh + 3), rx:rx + bw]]
        vals = np.concatenate([st.ravel() for st in strips if st.size])
        bg = float(np.median(vals)) if vals.size else float(np.percentile(signal, 10))
        bm = measure_band(signal, target, lane, rx, ry, bw, bh, method="fixed",
                          bg_value=bg, max_value=255.0, measurement_mode="rectangle")

        roi = crop[ry:ry + bh, rx:rx + bw]
        # Figure exports resample clipped pixels to near-black, not exactly 0.
        sat = float((roi <= 8).mean())
        if sat > 0.02:
            lane_notes.append(f"{sat:.0%} of box pixels are near-black (saturated in the figure)")
        # Vertical doublet: two separate peaks inside the box's row profile.
        from scipy.signal import find_peaks
        box_prof = sm[max(0, ry - 1):min(ih, ry + bh + 1)]
        if box_prof.size >= 5:
            rng = float(box_prof.max() - box_prof.min())
            pk, _ = find_peaks(box_prof, prominence=max(1.0, 0.25 * rng))
            if len(pk) >= 2:
                lane_notes.append("box contains two stacked bands (vertical doublet); both measured")
        results.append(LaneMeasurement(
            row=panel.row, col=panel.col, target=target, lane=lane, lane_index=li,
            roi_x=ix + rx, roi_y=iy + ry, roi_w=bw, roi_h=bh,
            band_x0=ix + bd.x0, band_x1=ix + bd.x1, band_y0=iy + bd.y0, band_y1=iy + bd.y1,
            split_x=ix + lane_split,
            peak_area=round(area, 2), net_intden=round(bm.net_intden, 2),
            background_per_px=round(bg, 3),
            baseline_y0=iy + b0, baseline_y1=iy + b1,
            saturated_fraction=round(sat, 4), source=bd.present, status=status,
            notes=lane_notes,
        ))
    return results

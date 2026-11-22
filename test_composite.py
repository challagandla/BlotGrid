"""Tests for composite (multi-tile) figure quantification."""
import numpy as np

from wblot.composite import (TileSettings, analyse_tile, assign_grid, find_framed_panels,
                             find_split, quantify_panel, span_extent,
                             valley_baseline_area)


def _band(img, x0, x1, yc, depth, half_h=3.0, profile=None):
    """Darken a horizontal band; ``profile`` optionally shapes it along x."""
    xs = np.arange(x0, x1)
    shape = np.ones(len(xs)) if profile is None else profile(np.linspace(0, 1, len(xs)))
    for y in range(int(yc - 3 * half_h), int(yc + 3 * half_h) + 1):
        if 0 <= y < img.shape[0]:
            img[y, x0:x1] -= depth * shape * np.exp(-0.5 * ((y - yc) / half_h) ** 2)


def _tile_signal(bands, w=72, h=34, bg=215.0, gradient=0.0):
    img = np.full((h, w), bg) - gradient * np.linspace(1, 0, h)[:, None]
    for kw in bands:
        _band(img, **kw)
    return 255.0 - np.clip(img, 0, 255)


def _figure(depths):
    """Two rows x two columns of framed 2-band tiles on white."""
    img = np.full((140, 220), 245.0)
    for r in range(2):
        for c in range(2):
            x, y, w, h = 10 + c * 105, 10 + r * 65, 80, 40
            img[y:y + h, x:x + w] = 215.0
            img[y, x:x + w] = img[y + h - 1, x:x + w] = 0
            img[y:y + h, x] = img[y:y + h, x + w - 1] = 0
            d0, d1 = depths[r][c]
            _band(img, x + 8, x + 34, y + 20, d0)
            _band(img, x + 46, x + 72, y + 20, d1)
    return np.clip(img, 0, 255)


def test_panels_found_and_gridded():
    img = _figure([[(100, 100), (100, 50)], [(100, 100), (60, 120)]])
    panels = assign_grid(find_framed_panels(img))
    assert len(panels) == 4
    assert sorted((p.row, p.col) for p in panels) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_ratio_recovered_with_equal_boxes():
    img = _figure([[(100, 100), (100, 50)], [(100, 100), (60, 120)]])
    panels = assign_grid(find_framed_panels(img))
    p = [q for q in panels if (q.row, q.col) == (0, 1)][0]
    ctrl, trt = quantify_panel(img, p, "T", ["Control", "KD"])
    assert (ctrl.roi_w, ctrl.roi_h) == (trt.roi_w, trt.roi_h)
    assert ctrl.roi_x + ctrl.roi_w <= trt.roi_x
    assert abs(trt.peak_area / ctrl.peak_area - 0.5) < 0.08


def test_box_size_set_by_larger_band_and_no_overlap():
    sig = _tile_signal([dict(x0=4, x1=34, yc=17, depth=120, half_h=4.0),   # long, thick
                        dict(x0=44, x1=60, yc=17, depth=60, half_h=2.0)])  # short, thin
    bands, split, gap, bw, bh, notes, boxes = analyse_tile(sig)
    assert bw >= 30 - 2 and bh >= bands[0].y1 - bands[0].y0
    (x0, _), (x1, _) = boxes
    assert x0 + bw <= x1                      # boxes never overlap
    assert x1 >= bands[0].x1                  # small band's box stays off the big band


def test_dumbbell_band_is_one_band():
    # Dark ends, pale middle: must give one extent spanning both lobes.
    dumbbell = lambda t: 0.25 + 0.75 * (np.exp(-((t - 0.1) / 0.12) ** 2) + np.exp(-((t - 0.9) / 0.12) ** 2))
    sig = _tile_signal([dict(x0=4, x1=34, yc=17, depth=120, profile=dumbbell),
                        dict(x0=40, x1=68, yc=17, depth=110)])
    bands, split, *_ = analyse_tile(sig)
    assert 34 <= split <= 40
    left = bands[0]
    assert left.x0 <= 6 and left.x1 >= 31


def test_touching_bands_split_at_junction():
    sig = _tile_signal([dict(x0=4, x1=36, yc=17, depth=110),
                        dict(x0=36, x1=68, yc=17, depth=90)])
    bands, split, gap, *_ = analyse_tile(sig)
    assert 32 <= split <= 40


def test_faint_band_next_to_strong_keeps_own_lane():
    col = np.zeros(70)
    col[5:30] = 100.0
    col[36:46] = 15.0
    split, _ = find_split(col)
    assert 30 <= split <= 36


def test_gradient_background_does_not_inflate_height():
    sig = _tile_signal([dict(x0=4, x1=32, yc=14, depth=120, half_h=2.5),
                        dict(x0=40, x1=68, yc=14, depth=100, half_h=2.5)], gradient=25.0)
    bands, split, gap, bw, bh, *_ = analyse_tile(sig)
    assert bh <= 14


def test_absent_band_box_is_mirrored():
    sig = _tile_signal([dict(x0=6, x1=30, yc=17, depth=120)])
    bands, split, gap, bw, bh, notes, boxes = analyse_tile(sig)
    assert bands[1].present == "absent"
    mirror = 2 * split - bands[0].cx
    assert abs((boxes[1][0] + bw / 2) - mirror) <= 3


def test_span_extent_ignores_specks():
    prof = np.zeros(40)
    prof[10:25] = 10.0
    prof[33] = 9.0  # 1-px speck
    assert span_extent(prof, 0, 40, 0.25) == (10, 25)


def test_valley_baseline_area_removes_slope():
    x = np.arange(40, dtype=float)
    profile = 0.5 * x + 20 * np.exp(-0.5 * ((x - 20) / 2.5) ** 2)
    area, left, right = valley_baseline_area(profile, 20, lo=5, hi=35)
    expected = 20 * 2.5 * np.sqrt(2 * np.pi)
    assert abs(area - expected) / expected < 0.1


def test_settings_change_box_padding():
    sig = _tile_signal([dict(x0=6, x1=30, yc=17, depth=120),
                        dict(x0=42, x1=66, yc=17, depth=100)])
    _, _, _, bw1, _, _, _ = analyse_tile(sig)
    _, _, _, bw2, _, _, _ = analyse_tile(sig, 2, TileSettings(pad=2))
    assert bw2 >= bw1


def test_comet_band_tail_is_captured():
    # Dark head on the left, long faded tail to the right (comet-shaped band).
    comet = lambda t: np.clip(1.0 - 0.9 * t, 0.1, 1.0)
    sig = _tile_signal([dict(x0=4, x1=34, yc=17, depth=120, profile=comet),
                        dict(x0=40, x1=68, yc=17, depth=110, profile=comet)])
    bands, split, *_ = analyse_tile(sig)
    assert bands[0].x0 <= 6 and bands[0].x1 >= 30      # head AND tail
    assert bands[1].x0 <= 42 and bands[1].x1 >= 64


def test_faint_band_ramping_into_strong_band_splits_at_junction():
    ramp = lambda t: 0.15 + 0.1 * t
    sig = _tile_signal([dict(x0=6, x1=36, yc=17, depth=120, profile=ramp),
                        dict(x0=36, x1=66, yc=17, depth=120)])
    bands, split, *_ = analyse_tile(sig)
    assert 32 <= split <= 40
    assert bands[0].x0 <= 10                           # faint band kept whole


def test_blotchy_background_does_not_extend_bands():
    rng = np.random.default_rng(3)
    from scipy.ndimage import gaussian_filter
    noise = gaussian_filter(rng.normal(0, 1, (34, 72)), 4)
    noise = 30 * noise / np.abs(noise).max()
    sig = _tile_signal([dict(x0=16, x1=36, yc=17, depth=110, half_h=1.5),
                        dict(x0=42, x1=60, yc=17, depth=40, half_h=1.5)])
    sig = sig + np.clip(noise, 0, None)
    bands, split, *_ = analyse_tile(sig)
    assert 34 <= split <= 44
    assert bands[0].x0 >= 8 and bands[1].x1 <= 68


def test_redraw_keeps_equal_size_and_no_overlap():
    from wblot.composite import Redraw, redraw_boxes
    sig = _tile_signal([dict(x0=4, x1=34, yc=17, depth=120),
                        dict(x0=36, x1=68, yc=17, depth=100)])
    bands, split, gap, bw, bh, notes, boxes, allowed = analyse_tile(sig, return_allowed=True)
    rng = np.random.default_rng(0)
    for _ in range(200):
        new, w2, h2 = redraw_boxes(boxes, bw, bh, allowed, sig.shape[1], sig.shape[0],
                                   Redraw.random(rng))
        assert abs(w2 - bw) <= 1 and abs(h2 - bh) <= 1
        assert new[0][0] + w2 <= new[1][0]                       # never overlap
        for (x, _), (lo, hi) in zip(new, allowed):
            assert lo <= x and x + w2 <= max(hi, lo + w2)        # keep-out respected


def test_redraw_repeats_are_consistent():
    img = _figure([[(100, 100), (100, 50)], [(100, 100), (60, 120)]])
    panels = assign_grid(find_framed_panels(img))
    p = [q for q in panels if (q.row, q.col) == (0, 1)][0]
    from wblot.composite import Redraw
    rng = np.random.default_rng(1)
    ratios = []
    for _ in range(3):
        c, t = quantify_panel(img, p, "T", ["Control", "KD"], redraw=Redraw.random(rng))
        ratios.append(t.peak_area / c.peak_area)
    assert np.std(ratios) / np.mean(ratios) < 0.05            # re-measurement, not a new method
    assert abs(np.mean(ratios) - 0.5) < 0.08


def test_layout_per_block_cell_lines():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from scripts.quantify_composite import label_panels
    from wblot.composite import Panel
    panels = [Panel(0, 0, 10, 10, r, c) for r in range(4) for c in range(2)]
    layout = {
        "row_blocks": [
            {"name": "P", "targets": ["Target", "Loading control"], "cell_lines": ["A", "B"]},
            {"name": "P", "targets": ["Target", "Loading control"], "cell_lines": ["C", "D"]},
        ],
        "col_blocks": [{"condition": "OE", "cell_lines": ["left", "right"]}],
    }
    row_info, col_info = label_panels(panels, layout)
    assert row_info[0] == ("P", "Target", ["A", "B"])
    assert row_info[3] == ("P", "Loading control", ["C", "D"])
    assert col_info[1] == ("OE", "right")


def test_five_lane_tile_equal_boxes_no_overlap():
    bands = [dict(x0=x, x1=x + 30, yc=17, depth=d) for x, d in
             zip((6, 46, 86, 126, 166), (40, 120, 70, 110, 100))]
    sig = _tile_signal(bands, w=202, h=34)
    found, splits, gaps, bw, bh, notes, boxes = analyse_tile(sig, 5)
    assert len(splits) == 4 and len(boxes) == 5
    for b, x in zip(splits, (41, 81, 121, 161)):
        assert abs(b - x) <= 4                         # boundaries in the gaps
    for (x0, _), (x1, _) in zip(boxes[:-1], boxes[1:]):
        assert x0 + bw <= x1                           # neighbours never overlap
    assert all(bd.present == "detected" for bd in found)


def test_multilane_layout_becomes_pairwise_comparisons():
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent))
    from scripts.quantify_composite import pairwise_vs_reference
    raw = pd.DataFrame({
        "lane_index": [0, 1, 2], "lane_label": ["Reference", "L1", "L2"],
        "cell_line": ["panel"] * 3, "lane_role": ["Control", "Treated", "Treated"],
        "target": ["Target"] * 3, "peak_area": [1.0, 2.0, 3.0]})
    out = pairwise_vs_reference(raw)
    assert sorted(out.cell_line.unique()) == ["L1", "L2"]
    l2 = out[out.cell_line == "L2"]
    assert set(l2.lane_role) == {"Control", "Treated"}
    assert l2[l2.lane_role == "Control"].lane_label.item() == "Reference"

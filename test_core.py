"""Unit tests for the densitometry and normalization math."""
import io
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from wblot import (alignment, detection, image_io, densitometry, normalization, stats,
                   project_io, roi_constraints, excel_export, label_ocr)


def test_ocr_uses_installed_package_models_when_vendor_bundle_is_absent():
    with TemporaryDirectory() as tmp:
        params = label_ocr._ocr_params(Path(tmp))

    assert params == {"Global.log_level": "warning"}


def test_ocr_prefers_complete_offline_vendor_bundle():
    with TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "rapidocr" / "models"
        model_dir.mkdir(parents=True)
        for name in (
            "PP-OCRv6_det_small.onnx",
            "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "PP-OCRv6_rec_small.onnx",
        ):
            (model_dir / name).touch()

        params = label_ocr._ocr_params(Path(tmp))

    assert params["Global.log_level"] == "warning"
    assert set(params) == {
        "Global.log_level", "Det.model_path", "Cls.model_path", "Rec.model_path",
    }


def test_alignment_estimates_and_corrects_a_skewed_blot():
    from scipy.ndimage import rotate

    height, width = 240, 360
    yy, xx = np.mgrid[:height, :width]
    original = np.full((height, width), 220.0)
    for cy in (60, 120, 180):
        for cx in (75, 180, 285):
            band = (np.abs(yy - cy) < 4) & (np.abs(xx - cx) < 35)
            original[band] = 30.0
    skewed = rotate(original, 4.0, reshape=False, order=1,
                    mode="constant", cval=220.0, prefilter=False)

    estimated = alignment.estimate_correction_angle(skewed, max_angle=8.0)

    assert abs(estimated + 4.0) <= 0.3


def test_alignment_rotation_preserves_shape_metadata_and_native_range():
    gray = np.arange(20 * 30, dtype=np.float64).reshape(20, 30)
    image = image_io.LoadedImage(gray, max_value=65535.0, bit_depth=16,
                                 name="scan.tif")

    corrected = alignment.rotate_loaded(image, 3.2)

    assert corrected.gray.shape == gray.shape
    assert corrected.max_value == 65535.0
    assert corrected.bit_depth == 16
    assert corrected.name == "scan.tif"
    assert corrected.gray.min() >= 0.0
    assert corrected.gray.max() <= 65535.0


def test_known_intden_uniform_signal():
    # signal image where every pixel = 10; a 5x4 ROI on a zero background frame.
    sig = np.zeros((20, 20))
    sig[5:10, 5:9] = 10.0  # 5 rows x 4 cols = 20 px of value 10
    m = densitometry.measure_band(sig, "T", "L1", x=5, y=5, w=4, h=5,
                                  method="local", margin=3, max_value=255)
    assert m.area == 20
    assert m.raw_intden == 200.0          # 20 px * 10
    assert abs(m.background_per_px) < 1e-9  # surrounding frame is 0
    assert m.net_intden == 200.0


def test_local_background_subtracted():
    sig = np.full((20, 20), 3.0)          # uniform background of 3
    sig[8:12, 8:12] = 13.0                # band = 10 above background, 16 px
    m = densitometry.measure_band(sig, "T", "L1", x=8, y=8, w=4, h=4,
                                  method="local", margin=3, max_value=255)
    assert abs(m.background_per_px - 3.0) < 1e-9
    assert abs(m.net_intden - 160.0) < 1e-6   # 16 px * (13-3)


def test_profile_guided_measurement_ignores_a_distant_border_peak():
    signal = np.full((24, 20), 5.0)
    signal[10:14, 7:13] = 15.0       # intended central band: 6 * 4 * 10 = 240
    signal[4, 7:13] = 45.0           # stronger, thin border near the ROI edge

    profile = densitometry.measure_band(
        signal, "T", "L", x=7, y=3, w=6, h=16,
        method="local", margin=2, max_value=255,
        measurement_mode="profile",
    )
    rectangle = densitometry.measure_band(
        signal, "T", "L", x=7, y=3, w=6, h=16,
        method="local", margin=2, max_value=255,
        measurement_mode="rectangle",
    )

    assert profile.measurement_mode == "profile"
    assert profile.profile_y0 <= 10 and profile.profile_y1 >= 14
    assert abs(profile.net_intden - 240.0) < 1e-6
    assert rectangle.net_intden > profile.net_intden


def test_normalize_to_control():
    df = pd.DataFrame({
        "target": ["Insig", "Prmt9", "GAPDH"] * 2,
        "lane": ["L1", "L1", "L1", "L2", "L2", "L2"],
        "net_intden": [200, 100, 100,   150, 300, 150],
    })
    out = normalization.normalize_to_control(df, "GAPDH")
    o = out.set_index(["target", "lane"])["normalized"]
    assert abs(o[("Insig", "L1")] - 2.0) < 1e-9    # 200/100
    assert abs(o[("Prmt9", "L1")] - 1.0) < 1e-9    # 100/100
    assert abs(o[("Insig", "L2")] - 1.0) < 1e-9    # 150/150
    assert abs(o[("Prmt9", "L2")] - 2.0) < 1e-9    # 300/150
    assert abs(o[("GAPDH", "L1")] - 1.0) < 1e-9    # control == 1


def test_prepare_intensities_without_loading_control():
    df = pd.DataFrame({
        "target": ["Target 1", "Target 1", "Target 2"],
        "lane": ["Lane 1", "Lane 1", "Lane 1"],
        "net_intden": [40.0, 15.0, 23.0],
    })
    summed = normalization.prepare_intensities(df, multiband="sum")
    values = summed.set_index("target")["net_intden"]
    assert values["Target 1"] == 55.0
    assert values["Target 2"] == 23.0

    kept = normalization.prepare_intensities(df, multiband="keep")
    assert list(kept[kept["target"] == "Target 1"]["band_index"]) == [0, 1]
    assert not normalization.has_usable_control(df, "")
    assert not normalization.has_usable_control(df, "GAPDH")


def test_nonpositive_or_missing_control_is_not_usable():
    df = pd.DataFrame({
        "target": ["AKT", "GAPDH", "GAPDH"],
        "net_intden": [80.0, 0.0, np.nan],
    })
    assert not normalization.has_usable_control(df, "GAPDH")
    df.loc[1, "net_intden"] = 12.0
    assert normalization.has_usable_control(df, "GAPDH")


def test_fold_change_reference_lane_is_one():
    df = pd.DataFrame({
        "target": ["Insig", "Insig"],
        "lane": ["L1", "L2"],
        "normalized": [2.0, 4.0],
    })
    fc = normalization.fold_change(df, reference_lane="L1")
    f = fc.set_index("lane")["fold_change"]
    assert abs(f["L1"] - 1.0) < 1e-9
    assert abs(f["L2"] - 2.0) < 1e-9


def test_divide_by_zero_control_is_nan():
    df = pd.DataFrame({
        "target": ["Insig", "GAPDH"],
        "lane": ["L1", "L1"],
        "net_intden": [200, 0],
    })
    out = normalization.normalize_to_control(df, "GAPDH")
    v = out.set_index("target")["normalized"]
    assert np.isnan(v["Insig"])


def test_aggregate_sem_sd():
    df = pd.DataFrame({
        "target": ["Insig"] * 3,
        "condition": ["KO"] * 3,
        "value": [1.0, 2.0, 3.0],
    })
    agg = stats.aggregate(df, "value").iloc[0]
    assert agg["n"] == 3
    assert abs(agg["mean"] - 2.0) < 1e-9
    assert abs(agg["sd"] - 1.0) < 1e-9              # ddof=1 -> 1.0
    assert abs(agg["sem"] - (1.0 / np.sqrt(3))) < 1e-9


def test_blank_background_is_not_saturation():
    # A dim band on a zero background: nothing here is clipped, so nothing
    # should be flagged. (Regression: zero-valued pixels were read as saturated.)
    sig = np.zeros((20, 20))
    sig[8:12, 8:12] = 40.0
    m = densitometry.measure_band(sig, "T", "L1", x=8, y=8, w=4, h=4,
                                  method="local", margin=3, max_value=255)
    assert m.saturated is False

    # A band clipped at the top of the range must still be flagged.
    sig[8:12, 8:12] = 255.0
    m2 = densitometry.measure_band(sig, "T", "L1", x=8, y=8, w=4, h=4,
                                   method="local", margin=3, max_value=255)
    assert m2.saturated is True


def test_fixed_background_is_used():
    sig = np.full((20, 20), 12.0)
    m = densitometry.measure_band(sig, "T", "L1", x=8, y=8, w=4, h=4,
                                  method="fixed", bg_value=5.0, max_value=255)
    assert abs(m.background_per_px - 5.0) < 1e-9
    assert abs(m.net_intden - 16 * 7.0) < 1e-6

    # A missing bg_value must fail loudly rather than silently measuring
    # with a different background method.
    try:
        densitometry.measure_band(sig, "T", "L1", x=8, y=8, w=4, h=4,
                                  method="fixed", max_value=255)
    except ValueError:
        pass
    else:
        raise AssertionError("method='fixed' without bg_value should raise")


def test_16bit_image_keeps_its_range():
    arr = np.zeros((8, 8), dtype=np.uint16)
    arr[2:4, 2:4] = 65535
    loaded = image_io.LoadedImage(gray=arr.astype(np.float64),
                                  max_value=65535.0, bit_depth=16)
    sig = image_io.to_signal(loaded, bands_are_dark=False)
    m = densitometry.measure_band(sig, "T", "L1", x=2, y=2, w=2, h=2,
                                  method="local", margin=2,
                                  max_value=loaded.max_value)
    assert m.saturated is True          # genuinely at the top of a 16-bit range
    assert m.raw_intden == 4 * 65535.0


def test_aggregate_keeps_group_columns():
    df = pd.DataFrame({
        "target": ["Insig"] * 3 + ["Prmt9"] * 3,
        "condition": ["WT", "KO", "KO", "WT", "KO", "KO"],
        "value": [1.0, 2.0, 4.0, 5.0, 1.0, 3.0],
    })
    agg = stats.aggregate(df, "value")
    assert list(agg.columns) == ["target", "condition", "mean", "sd", "sem", "ci95", "n"]
    assert set(agg["target"]) == {"Insig", "Prmt9"}
    ko = agg[(agg["target"] == "Insig") & (agg["condition"] == "KO")].iloc[0]
    assert ko["n"] == 2 and abs(ko["mean"] - 3.0) < 1e-9
    wt = agg[(agg["target"] == "Insig") & (agg["condition"] == "WT")].iloc[0]
    assert wt["n"] == 1 and wt["sd"] == 0.0     # single replicate has no spread


def test_ci95_is_t_based_not_1p96():
    df = pd.DataFrame({"target": ["A"] * 3, "condition": ["KO"] * 3,
                       "value": [1.0, 2.0, 3.0]})
    agg = stats.aggregate(df, "value").iloc[0]
    # t(0.975, df=2) = 4.303, so the CI must be far wider than 1.96*SEM.
    assert abs(agg["ci95"] - 4.302652729911275 * agg["sem"]) < 1e-6


def test_compare_groups_labels_match_groups():
    # The WT group is all-NaN and gets dropped; the remaining labels must still
    # line up with the data actually tested.
    df = pd.DataFrame({
        "target": ["A"] * 6,
        "condition": ["WT", "WT", "KO", "KO", "Rescue", "Rescue"],
        "value": [np.nan, np.nan, 1.0, 1.2, 3.0, 3.4],
    })
    res = stats.compare_groups(df, "value", target="A")
    assert res["labels"] == ["KO", "Rescue"]
    assert res["test"] == "Welch t-test"


def test_project_round_trip_from_bytes():
    state = {"targets": ["Insig", "GAPDH"], "lanes": ["WT", "KO"],
             "control": "GAPDH", "bands_dark": True,
             "alignment_applied": {"blot1.tif": 0.0},
             "image_provenance": {"blot1.tif": {"sha256": "abc123"}},
             "evaluation_task": {
                 "schema_version": 1,
                 "task_id": "task-opaque",
                 "task_binding_sha256": "f" * 64,
             },
             "layouts": {"blot1.tif": {"band_width": 40, "lane_x": [80, 180],
                                       "targets": {"Insig": {"y": 60, "h": 20}}}}}
    raw = project_io.project_json(state).encode("utf-8")
    back = project_io.load_project(raw)          # as uploaded bytes, not a path
    assert back["targets"] == ["Insig", "GAPDH"]
    assert back["layouts"]["blot1.tif"]["lane_x"] == [80, 180]
    assert back["alignment_applied"]["blot1.tif"] == 0.0
    assert back["image_provenance"]["blot1.tif"]["sha256"] == "abc123"
    assert back["evaluation_task"]["task_id"] == "task-opaque"
    assert "_saved_at" in back


def test_measurement_clips_original_rectangle_endpoints():
    sig = np.ones((10, 10), dtype=float)
    m = densitometry.measure_band(
        sig, "T", "L", x=-3, y=-2, w=5, h=4,
        method="fixed", bg_value=0.0, max_value=255,
    )
    # The intersection is x=0..2 and y=0..2, not a shifted 5x4 rectangle.
    assert (m.x, m.y, m.w, m.h, m.area) == (0, 0, 2, 2, 4)


def test_saturation_mask_respects_polarity():
    gray = np.array([[0.0, 100.0, 255.0]])
    loaded = image_io.LoadedImage(gray=gray, max_value=255.0, bit_depth=8)
    assert image_io.saturated_mask(loaded, bands_are_dark=True).tolist() == [[True, False, False]]
    assert image_io.saturated_mask(loaded, bands_are_dark=False).tolist() == [[False, False, True]]


def test_fold_change_keeps_multiband_species_aligned():
    df = pd.DataFrame({
        "target": ["A", "A", "A", "A"],
        "lane": ["ref", "ref", "test", "test"],
        "band_index": [0, 1, 0, 1],
        "normalized": [2.0, 10.0, 4.0, 5.0],
    })
    out = normalization.fold_change(df, "ref")
    vals = out[out["lane"] == "test"].set_index("band_index")["fold_change"]
    assert vals[0] == 2.0
    assert vals[1] == 0.5


def _synthetic_blot(*, dark: bool, doublet: bool = False):
    H, W = 320, 420
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[:H, :W]
    foreground = np.zeros((H, W), dtype=float)
    xs, ys = [70, 205, 345], [65, 165, 265]
    amounts = [35, 16, 28, 20, 38, 25, 30, 27, 32]
    for j, cy in enumerate(ys):
        for i, cx in enumerate(xs):
            foreground += amounts[j * 3 + i] * np.exp(
                -((xx - cx) ** 2 / (2 * 25**2) + (yy - cy) ** 2 / (2 * 7**2)))
    if doublet:
        foreground += 25 * np.exp(
            -((xx - 205) ** 2 / (2 * 22**2) + (yy - 142) ** 2 / (2 * 5**2)))
    background = 120 + 0.015 * xx + 0.01 * yy + rng.normal(0, 1.1, (H, W))
    return background - foreground if dark else background + foreground


def test_auto_detection_finds_all_dark_bands_and_layout():
    result = detection.detect_bands(
        _synthetic_blot(dark=True),
        ["A", "B", "C"], ["1", "2", "3"],
    )
    assert result.bands_are_dark is True
    assert len(result.bands) == 9
    assert len(result.lane_centers) == 3
    assert len(result.target_centers) == 3
    assert {(b.target, b.lane) for b in result.bands} == {
        (target, lane) for target in ["A", "B", "C"] for lane in ["1", "2", "3"]
    }


def test_auto_detection_finds_light_doublet_as_two_bands():
    result = detection.detect_bands(
        _synthetic_blot(dark=False, doublet=True),
        ["A", "B", "C"], ["1", "2", "3"],
    )
    assert result.bands_are_dark is False
    assert len(result.bands) == 10
    doublet = [b for b in result.bands if b.target == "B" and b.lane == "2"]
    assert len(doublet) == 2
    assert doublet[0].y < doublet[1].y


def test_merged_broad_peak_is_flagged_unresolved_not_invented_as_two_bands():
    gray = _synthetic_blot(dark=True)
    height, width = gray.shape
    yy, xx = np.mgrid[:height, :width]
    # Add a close shoulder to the existing B/2 band. It remains one visible
    # maximum but becomes a robust width outlier relative to the other bands.
    gray -= 24 * np.exp(
        -((xx - 205) ** 2 / (2 * 22**2) + (yy - 153) ** 2 / (2 * 5**2)))

    result = detection.detect_bands(
        gray, ["A", "B", "C"], ["1", "2", "3"], polarity="dark")
    merged = [band for band in result.bands if band.target == "B" and band.lane == "2"]

    assert len(merged) == 1
    assert merged[0].status == "unresolved overlap"
    assert merged[0].review_required is True


def test_long_thick_bands_are_not_mistaken_for_panel_borders():
    height, width = 240, 360
    yy, xx = np.mgrid[:height, :width]
    lane_centres = [60, 180, 300]
    target_centres = [60, 120, 180]
    gray = np.full((height, width), 235.0)
    # Broad neighbouring bands merge horizontally across much of the image,
    # but their biological thickness distinguishes them from a thin panel rule.
    for y in target_centres:
        for x in lane_centres:
            gray -= 70 * np.exp(
                -((xx - x) ** 2 / (2 * 55**2) + (yy - y) ** 2 / (2 * 7**2)))

    result = detection.detect_bands(
        gray, ["A", "B", "C"], ["L1", "L2", "L3"], polarity="dark",
        lane_centers_hint=lane_centres,
        target_centers_hint=target_centres,
    )

    assert len(result.bands) == 9
    assert all(band.use for band in result.bands)


def test_faint_recovery_is_active_but_blank_cell_stays_excluded():
    height, width = 80, 120
    yy, xx = np.mgrid[:height, :width]

    faint = np.full((height, width), 235.0)
    faint -= 0.5 * np.exp(
        -((xx - 60) ** 2 / (2 * 15**2) + (yy - 40) ** 2 / (2 * 3**2)))
    faint_result = detection.DetectionResult([], [60], [40], True, 1.0)
    detection.ensure_complete_grid(
        faint_result, ["T"], ["L"], faint.shape, gray=faint)
    faint_band = faint_result.bands[0]

    blank = np.full((height, width), 235.0)
    blank_result = detection.DetectionResult([], [60], [40], True, 1.0)
    detection.ensure_complete_grid(
        blank_result, ["T"], ["L"], blank.shape, gray=blank)
    blank_band = blank_result.bands[0]

    assert faint_band.source == "faint recovery"
    assert faint_band.use is True
    assert faint_band.review_required is True
    assert blank_band.source == "fallback"
    assert blank_band.use is False


def test_auto_detection_ignores_panel_borders_and_right_side_labels():
    """Regression for small publication figures with labels beside each blot."""
    H, W = 250, 208
    rng = np.random.default_rng(19)
    yy, xx = np.mgrid[:H, :W]
    image = np.full((H, W), 238.0) + rng.normal(0, 1.0, (H, W))
    row_y = [26, 60, 94, 128, 162, 196, 230]
    lane_x = [35, 91]

    for row_index, cy in enumerate(row_y):
        y0, y1 = max(1, cy - 16), min(H - 2, cy + 16)
        # Thin panel frame, like a pasted/cropped paper figure.
        image[y0:y0 + 1, 10:119] = 150
        image[y1 - 1:y1, 10:119] = 150
        image[y0:y1, 10:11] = 150
        image[y0:y1, 118:119] = 150
        for lane_index, cx in enumerate(lane_x):
            amplitude = 90 + ((row_index * 2 + lane_index) * 13) % 80
            sy, sx = 4.0 + row_index % 3, 14 + lane_index * 3
            image -= amplitude * np.exp(
                -((xx - cx) ** 2 / (2 * sx**2) + (yy - cy) ** 2 / (2 * sy**2)))

        # Label-like dark strokes outside the blot panel.  They must not become
        # a third lane or extra bands.
        image[max(0, cy - 7):min(H, cy + 8), 132:134] = 80
        image[max(0, cy - 6):max(1, cy - 5), 132:187] = 80
        image[min(H - 1, cy + 5):min(H, cy + 6), 132:175] = 80

    targets = [f"T{i}" for i in range(7)]
    result = detection.detect_bands(
        np.clip(image, 0, 255), targets, ["left", "right"])

    assert result.bands_are_dark is True
    assert len(result.bands) == 14
    assert max(b.x + b.w for b in result.bands) < 120
    assert len({(b.w, b.h) for b in result.bands}) == 1
    assert {(b.target, b.lane) for b in result.bands} == {
        (target, lane) for target in targets for lane in ["left", "right"]
    }


def test_roi_editor_server_guard_enforces_one_blot_size():
    rois = [
        {"roi_id": "a", "lane": "L1", "x": 10, "y": 10, "w": 20, "h": 10,
         "annotation_approved": True},
        {"roi_id": "b", "lane": "L1", "x": 10, "y": 40, "w": 20, "h": 10,
         "annotation_approved": True},
        {"roi_id": "c", "lane": "L2", "x": 60, "y": 10, "w": 20, "h": 10,
         "annotation_approved": True},
    ]
    shared_size = {"w": 20, "h": 10}

    # A move event is not allowed to smuggle in per-band size changes.
    malicious_move = {
        "action": "move", "selected_id": "a",
        "rois": [
            {"roi_id": "a", "x": 15, "y": 12, "w": 80, "h": 50},
            {"roi_id": "b", "x": 10, "y": 40, "w": 4, "h": 3},
            {"roi_id": "c", "x": 60, "y": 10, "w": 30, "h": 12},
        ],
    }
    assert roi_constraints.apply_editor_event(rois, shared_size, malicious_move, 120, 100)
    assert {(roi["w"], roi["h"]) for roi in rois} == {(20, 10)}
    assert not rois[0]["annotation_approved"]

    # A legitimate handle resize changes the shared lane size, never one band.
    for roi in rois:
        roi["annotation_approved"] = True
    resize = {
        "action": "resize", "selected_id": "a",
        "rois": [
            {"roi_id": "a", "x": 12, "y": 10, "w": 34, "h": 16},
            {"roi_id": "b", "x": 4, "y": 35, "w": 2, "h": 2},
            {"roi_id": "c", "x": 60, "y": 10, "w": 99, "h": 50},
        ],
    }
    assert roi_constraints.apply_editor_event(rois, shared_size, resize, 120, 100)
    assert shared_size == {"w": 34, "h": 16}
    assert {(roi["w"], roi["h"]) for roi in rois} == {(34, 16)}

    frozen_rois = [dict(roi) for roi in rois]
    frozen_size = dict(shared_size)
    assert not roi_constraints.apply_editor_event(
        rois,
        shared_size,
        resize,
        120,
        100,
        allow_resize=False,
    )
    assert rois == frozen_rois
    assert shared_size == frozen_size
    assert not any(roi["annotation_approved"] for roi in rois)


def test_excel_export_is_in_memory_and_sanitizes_invalid_values():
    frames = {
        "summary": pd.DataFrame({
            "target": ["Acc1", "bad\x01text"],
            "mean": [1.25, np.inf],
        }),
        "normalized": pd.DataFrame({
            "lane": ["WT", "KO"],
            "value": [1.0, np.nan],
        }),
        "raw_densitometry": pd.DataFrame({"signal": [42, 17]}),
    }

    payload = excel_export.dataframes_to_xlsx(frames)
    workbook = load_workbook(io.BytesIO(payload), read_only=True, data_only=True)

    assert workbook.sheetnames == list(frames)
    assert workbook["summary"]["A3"].value == "badtext"
    assert workbook["summary"]["B3"].value is None
    assert workbook["normalized"]["B3"].value is None


def test_ocr_metadata_classifies_targets_and_infers_unlabelled_lanes():
    height, width = 250, 208
    yy, xx = np.mgrid[:height, :width]
    gray = np.full((height, width), 240.0)
    row_centres = [24, 57, 91, 124, 158, 193, 227]
    for y in row_centres:
        for x in (35, 90):
            gray -= 130 * np.exp(
                -((xx - x) ** 2 / (2 * 12**2) + (yy - y) ** 2 / (2 * 4**2)))

    names = ["Acc1", "Fasn", "Hmgcr", "Scd1", "Srebf1", "Insig1", "Gapdh"]
    tokens = [
        label_ocr.TextToken(
            name, 0.98,
            ((124.0, y - 8.0), (176.0, y - 8.0),
             (176.0, y + 8.0), (124.0, y + 8.0)),
        )
        for name, y in zip(names, row_centres)
    ]

    labels = label_ocr.classify_experiment_labels(gray, tokens)

    assert list(labels.targets) == names
    assert list(labels.lanes) == ["Lane 1", "Lane 2"]
    assert labels.lane_source == "inferred"
    assert len(labels.lane_centers) == 2
    assert labels.target_centers == tuple(row_centres)
    assert label_ocr.suggested_control(labels.targets) == "Gapdh"


def test_raw_scan_infers_neutral_three_by_three_grid_and_no_control():
    height, width = 260, 330
    yy, xx = np.mgrid[:height, :width]
    gray = np.full((height, width), 238.0)
    target_centres = [48, 130, 214]
    lane_centres = [58, 166, 274]
    amplitudes = [130, 22, 92, 75, 115, 33, 100, 87, 125]
    for amplitude, (y, x) in zip(
            amplitudes,
            ((y, x) for y in target_centres for x in lane_centres)):
        gray -= amplitude * np.exp(
            -((xx - x) ** 2 / (2 * 18**2) + (yy - y) ** 2 / (2 * 5**2)))

    image = image_io.LoadedImage(gray, max_value=255.0, bit_depth=8, name="raw.tif")
    labels = label_ocr.infer_raw_scan_labels(image)

    assert labels.targets == ("Target 1", "Target 2", "Target 3")
    assert labels.lanes == ("Lane 1", "Lane 2", "Lane 3")
    assert labels.lane_source == "raw_scan"
    assert np.max(np.abs(np.asarray(labels.target_centers) - target_centres)) <= 2
    assert np.max(np.abs(np.asarray(labels.lane_centers) - lane_centres)) <= 3
    assert label_ocr.suggested_control(labels.targets) == ""


def test_unstructured_raw_scan_uses_safe_single_cell_fallback():
    rng = np.random.default_rng(19)
    gray = 230.0 + rng.normal(0.0, 2.0, size=(180, 240))
    grid = detection.infer_raw_grid(gray)
    assert grid.target_centers == (90,)
    assert grid.lane_centers == (120,)
    assert grid.warning


def test_unknown_target_names_do_not_become_an_implicit_control():
    assert label_ocr.suggested_control(["AKT", "ERK", "Target 3"]) == ""
    assert label_ocr.suggested_control(["AKT", "GAPDH 37 kDa"]) == "GAPDH 37 kDa"


def test_ocr_metadata_infers_nine_lane_bordered_panel():
    height, width = 301, 488
    yy, xx = np.mgrid[:height, :width]
    gray = np.full((height, width), 244.0)
    row_centres = [20, 60, 100, 140, 180, 220, 260]
    lane_centres = [64 + 40 * index for index in range(9)]

    # Repeated panel frames resemble a figure assembled from seven blots.
    for y in row_centres:
        y0, y1 = max(0, y - 17), min(height, y + 18)
        gray[y0:y1, 44:46] = 35
        gray[y0:y1, 404:406] = 35
        gray[max(0, y - 17):max(0, y - 15), 44:406] = 35
        gray[min(height, y + 15):min(height, y + 17), 44:406] = 35
        for lane_index, x in enumerate(lane_centres):
            amplitude = 45 + ((lane_index + row_centres.index(y)) % 4) * 25
            gray -= amplitude * np.exp(
                -((xx - x) ** 2 / (2 * 10**2) + (yy - y) ** 2 / (2 * 3.5**2)))

    names = ["Acc1", "Fasn", "Hmgcr", "Scd1", "Srebf1", "Insig1", "Gapdh"]
    tokens = [
        label_ocr.TextToken(
            name, 0.99,
            ((410.0, y - 7.0), (474.0, y - 7.0),
             (474.0, y + 7.0), (410.0, y + 7.0)),
        )
        for name, y in zip(names, row_centres)
    ]

    labels = label_ocr.classify_experiment_labels(gray, tokens)

    assert labels.targets == tuple(names)
    assert len(labels.lanes) == 9
    assert labels.blot_bounds is not None
    assert abs(labels.blot_bounds[0] - 44) <= 3
    assert abs(labels.blot_bounds[1] - 406) <= 3
    assert np.max(np.abs(np.asarray(labels.lane_centers) - lane_centres)) <= 3


def test_detection_hints_and_complete_grid_cover_every_cell_with_equal_rois():
    height, width = 150, 240
    yy, xx = np.mgrid[:height, :width]
    gray = np.full((height, width), 235.0)
    targets = ["T1", "T2", "Gapdh"]
    lanes = [f"Lane {index}" for index in range(1, 5)]
    target_centres = [30, 75, 120]
    lane_centres = [45, 95, 145, 195]

    # Deliberately omit several peaks so the completion stage has work to do.
    for row_index, y in enumerate(target_centres):
        for lane_index, x in enumerate(lane_centres):
            if (row_index + lane_index) % 3 == 0:
                continue
            gray -= 125 * np.exp(
                -((xx - x) ** 2 / (2 * 10**2) + (yy - y) ** 2 / (2 * 3**2)))

    result = detection.detect_bands(
        gray,
        targets,
        lanes,
        lane_centers_hint=lane_centres,
        target_centers_hint=target_centres,
    )
    added = detection.ensure_complete_grid(result, targets, lanes, gray.shape)

    assert result.lane_centers == lane_centres
    assert result.target_centers == target_centres
    assert added > 0
    assert {(band.target, band.lane) for band in result.bands} == {
        (target, lane) for target in targets for lane in lanes
    }
    assert len(result.bands) == len(targets) * len(lanes)
    assert len({(band.w, band.h) for band in result.bands}) == 1
    fallbacks = [band for band in result.bands if band.source == "fallback"]
    assert fallbacks
    assert all(band.use is False for band in fallbacks)
    assert all(band.review_required is True for band in fallbacks)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("All tests passed.")

"""Focused tests for OCR lane metadata and conservative ladder recognition."""

import numpy as np
import pytest

from wblot import detection, image_io, label_ocr


def _box(x: float, y: float, width: float = 52, height: float = 14):
    return (
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    )


def test_printed_ladder_is_recorded_without_removing_the_physical_lane():
    gray = np.full((240, 420), 230.0)
    tokens = [
        label_ocr.TextToken("AKT", 0.98, _box(8, 55)),
        label_ocr.TextToken("GAPDH", 0.98, _box(8, 165)),
        label_ocr.TextToken("Ladder", 0.99, _box(86, 8, 62)),
        label_ocr.TextToken("WT", 0.97, _box(202, 8)),
        label_ocr.TextToken("KO", 0.96, _box(318, 8)),
    ]

    labels = label_ocr.classify_experiment_labels(gray, tokens)

    assert labels.lanes == ("Ladder", "WT", "KO")
    assert labels.ladder_lanes == ("Ladder",)
    assert labels.lane_centers == (117, 228, 344)
    assert labels.autofill_ready is True
    assert label_ocr.can_autofill_metadata(labels) is True


@pytest.mark.parametrize(
    "value",
    [
        "Marker",
        "ladder",
        "Protein marker",
        "Protein-Ladder",
        "molecular-weight marker",
        "Molecular weight ladder",
        "MW",
        "M.W.",
        "MW marker",
    ],
)
def test_ladder_recognition_accepts_only_explicit_whole_labels(value):
    assert label_ocr.detected_ladder_lanes((value,)) == (value,)


@pytest.mark.parametrize(
    "value",
    [
        "M",
        "L",
        "marker-treated",
        "biomarker",
        "protein",
        "molecular weight",
        "MW treatment",
    ],
)
def test_ladder_recognition_rejects_ambiguous_or_embedded_terms(value):
    assert label_ocr.detected_ladder_lanes((value,)) == ()


@pytest.mark.parametrize(
    ("grid_confidence", "expected"),
    [(0.12, False), (0.42, True)],
)
def test_raw_grid_autofill_requires_confidence(monkeypatch, grid_confidence, expected):
    grid = detection.RawGridInference(
        lane_centers=(40, 120),
        target_centers=(35, 95),
        bands_are_dark=True,
        polarity_confidence=0.8,
        grid_confidence=grid_confidence,
    )
    monkeypatch.setattr(label_ocr.detection, "infer_raw_grid", lambda gray: grid)
    image = image_io.LoadedImage(
        gray=np.full((130, 170), 230.0),
        max_value=255.0,
        bit_depth=8,
    )

    labels = label_ocr.infer_raw_scan_labels(image)

    assert labels.autofill_ready is expected
    assert label_ocr.can_autofill_metadata(labels) is expected


def test_autofill_helper_rejects_incomplete_and_legacy_low_confidence_metadata():
    incomplete = label_ocr.ExperimentLabels(
        targets=("AKT",),
        lanes=(),
        target_confidence=0.95,
        lane_confidence=0.0,
        lane_source="none",
        tokens=(),
    )
    legacy_raw = label_ocr.ExperimentLabels(
        targets=("Target 1",),
        lanes=("Lane 1",),
        target_confidence=0.10,
        lane_confidence=0.10,
        lane_source="raw_scan",
        tokens=(),
    )

    assert incomplete.ladder_lanes == ()
    assert incomplete.autofill_ready is True
    assert label_ocr.can_autofill_metadata(incomplete) is False
    assert label_ocr.can_autofill_metadata(legacy_raw) is False


def test_autofill_helper_requires_at_least_one_quantifiable_lane():
    labels = label_ocr.ExperimentLabels(
        targets=("AKT",),
        lanes=("Ladder",),
        target_confidence=0.98,
        lane_confidence=0.98,
        lane_source="printed",
        tokens=(),
        ladder_lanes=("Ladder",),
    )

    assert label_ocr.can_autofill_metadata(labels) is False

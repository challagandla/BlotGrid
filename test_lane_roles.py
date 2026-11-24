"""Focused tests for explicit ladder-lane exclusion guards."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from wblot.lane_roles import (
    enforce_excluded_rois,
    filter_lane_rows,
    ordered_quantifiable_lanes,
)


def test_ordered_quantifiable_lanes_preserves_order_and_deduplicates_state():
    lanes = ["Ladder", "WT", "WT", "KO", "Ladder", "Rescue"]
    ladders = ["Ladder", "Ladder"]

    result = ordered_quantifiable_lanes(lanes, ladders)

    assert result == ["WT", "KO", "Rescue"]
    assert lanes == ["Ladder", "WT", "WT", "KO", "Ladder", "Rescue"]
    assert ladders == ["Ladder", "Ladder"]


def test_ordered_quantifiable_lanes_uses_exact_case_sensitive_identity():
    assert ordered_quantifiable_lanes(
        ["Ladder", "ladder", "Marker treated"], {"Ladder"}
    ) == ["ladder", "Marker treated"]


def test_ordered_quantifiable_lanes_accepts_generators_and_none_exclusions():
    lanes = (lane for lane in ("WT", "KO", "Rescue"))
    assert ordered_quantifiable_lanes(lanes, None) == ["WT", "KO", "Rescue"]
    assert ordered_quantifiable_lanes([], []) == []


@pytest.mark.parametrize(
    ("all_lanes", "ladder_lanes", "error"),
    [
        (None, None, TypeError),
        ("WT", None, TypeError),
        (["WT", 7], None, TypeError),
        (["WT", "  "], None, ValueError),
        (["WT"], "Ladder", TypeError),
        (["WT"], [None], TypeError),
        (["WT"], ["Ladder"], ValueError),
    ],
)
def test_ordered_quantifiable_lanes_rejects_malformed_or_stale_state(
    all_lanes, ladder_lanes, error
):
    with pytest.raises(error):
        ordered_quantifiable_lanes(all_lanes, ladder_lanes)


def test_filter_lane_rows_removes_only_explicit_ladders_and_returns_a_copy():
    frame = pd.DataFrame(
        {
            "lane": ["Ladder", "WT", "KO", "ladder", pd.NA],
            "value": [999.0, 1.0, 2.0, 500.0, 3.0],
        },
        index=[10, 11, 12, 13, 14],
    )

    result = filter_lane_rows(frame, ["Ladder", "Ladder"])

    assert result is not frame
    assert list(result.index) == [11, 12, 13, 14]
    assert result["lane"].tolist()[:3] == ["WT", "KO", "ladder"]
    assert pd.isna(result["lane"].iloc[3])
    result.loc[11, "value"] = 77.0
    assert frame.loc[11, "value"] == 1.0


def test_filter_lane_rows_preserves_none_and_copies_empty_or_unfiltered_frames():
    assert filter_lane_rows(None, ["Ladder"]) is None

    empty = pd.DataFrame(columns=["value"])
    empty_result = filter_lane_rows(empty, ["Ladder"])
    assert empty_result is not empty
    pd.testing.assert_frame_equal(empty_result, empty)

    frame = pd.DataFrame({"value": [1.0]})
    result = filter_lane_rows(frame, None)
    assert result is not frame
    pd.testing.assert_frame_equal(result, frame)


def test_filter_lane_rows_supports_an_explicit_lane_column_name():
    frame = pd.DataFrame(
        {"condition": ["WT", "Marker", "KO"], "value": [1, 100, 2]}
    )
    result = filter_lane_rows(frame, ["Marker"], lane_col="condition")
    assert result["condition"].tolist() == ["WT", "KO"]


def test_filter_lane_rows_requires_one_lane_column_only_when_filtering_data():
    with pytest.raises(ValueError, match="missing lane column"):
        filter_lane_rows(pd.DataFrame({"value": [1.0]}), ["Ladder"])

    duplicate_columns = pd.DataFrame(
        [["WT", "Ladder"]], columns=["lane", "lane"]
    )
    with pytest.raises(ValueError, match="duplicate lane column"):
        filter_lane_rows(duplicate_columns, ["Ladder"])


@pytest.mark.parametrize(
    ("frame", "ladder_lanes", "lane_col", "error"),
    [
        ([{"lane": "WT"}], [], "lane", TypeError),
        (None, "Ladder", "lane", TypeError),
        (None, [], None, TypeError),
        (None, [], "  ", ValueError),
    ],
)
def test_filter_lane_rows_rejects_malformed_arguments(
    frame, ladder_lanes, lane_col, error
):
    with pytest.raises(error):
        filter_lane_rows(frame, ladder_lanes, lane_col=lane_col)


def test_enforce_excluded_rois_updates_matches_and_preserves_unrelated_rois():
    unrelated = {
        "roi_id": "sample",
        "lane": "WT",
        "use": True,
        "annotation_approved": True,
        "review_required": False,
        "status": "accepted",
        "confidence": 0.97,
    }
    layout = {
        "annotation_complete": True,
        "negative_image": True,
        "rois": [
            {
                "roi_id": "marker-a",
                "lane": "Ladder",
                "use": True,
                "annotation_approved": True,
                "review_required": False,
                "status": "accepted",
                "confidence": 0.75,
            },
            unrelated,
            {
                "roi_id": "marker-b",
                "lane": "Ladder",
                "use": np.bool_(True),
                "annotation_approved": 0,
                "review_required": 1,
                "status": "fallback",
            },
        ],
    }
    unrelated_before = copy.deepcopy(unrelated)

    changed = enforce_excluded_rois(layout, ["Ladder", "Ladder"])

    assert changed is True
    for roi in (layout["rois"][0], layout["rois"][2]):
        assert roi["use"] is False
        assert roi["annotation_approved"] is False
        assert roi["review_required"] is True
        assert roi["status"] == "ladder omitted"
    assert layout["rois"][0]["confidence"] == 0.75
    assert unrelated == unrelated_before
    assert layout["annotation_complete"] is False
    assert layout["negative_image"] is False


def test_enforce_excluded_rois_is_idempotent():
    layout = {
        "annotation_complete": True,
        "negative_image": False,
        "rois": [{"lane": "Ladder", "use": True}],
    }
    assert enforce_excluded_rois(layout, ["Ladder"]) is True
    once = copy.deepcopy(layout)

    assert enforce_excluded_rois(layout, ["Ladder"]) is False
    assert layout == once


def test_enforce_excluded_rois_does_not_invalidate_without_semantic_change():
    layout = {
        "annotation_complete": True,
        "negative_image": True,
        "rois": [
            {
                "lane": "Ladder",
                "use": False,
                "annotation_approved": False,
                "review_required": True,
                "status": "ladder omitted",
            }
        ],
    }

    assert enforce_excluded_rois(layout, ["Ladder"]) is False
    assert layout["annotation_complete"] is True
    assert layout["negative_image"] is True


@pytest.mark.parametrize(
    "layout",
    [
        {},
        {"rois": []},
        {"rois": [{"lane": "WT", "use": True}]},
    ],
)
def test_enforce_excluded_rois_safely_handles_no_matching_rois(layout):
    before = copy.deepcopy(layout)
    assert enforce_excluded_rois(layout, ["Ladder"]) is False
    assert layout == before


def test_enforce_excluded_rois_empty_exclusions_are_a_safe_noop():
    layout = {"rois": "legacy coarse layout"}
    before = copy.deepcopy(layout)
    assert enforce_excluded_rois(layout, None) is False
    assert layout == before


@pytest.mark.parametrize(
    "bad_roi",
    [
        "not a mapping",
        {"use": True},
        {"lane": None, "use": True},
        {"lane": "  ", "use": True},
    ],
)
def test_enforce_excluded_rois_rejects_malformed_entries_atomically(bad_roi):
    layout = {
        "annotation_complete": True,
        "negative_image": True,
        "rois": [
            {
                "lane": "Ladder",
                "use": True,
                "annotation_approved": True,
                "status": "accepted",
            },
            bad_roi,
            {"lane": "WT", "use": True},
        ],
    }
    before = copy.deepcopy(layout)

    with pytest.raises(ValueError):
        enforce_excluded_rois(layout, ["Ladder"])

    assert layout == before


def test_enforce_excluded_rois_rejects_malformed_containers_without_mutation():
    with pytest.raises(TypeError, match="mutable mapping"):
        enforce_excluded_rois(None, ["Ladder"])

    layout = {"rois": ({"lane": "Ladder", "use": True},)}
    before = copy.deepcopy(layout)
    with pytest.raises(ValueError, match="must be a list"):
        enforce_excluded_rois(layout, ["Ladder"])
    assert layout == before


def test_enforce_excluded_rois_rejects_malformed_exclusions_atomically():
    layout = {"rois": [{"lane": "Ladder", "use": True}]}
    before = copy.deepcopy(layout)
    with pytest.raises(TypeError):
        enforce_excluded_rois(layout, ["Ladder", 3])
    assert layout == before

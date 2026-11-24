"""Explicit lane-role guards shared by quantification and review workflows.

Lane names are identifiers in BlotGrid.  These helpers therefore use exact,
case-sensitive matching and never infer a ladder from a label.  Detection or UI
code may propose ladder lanes, but only the explicit ``ladder_lanes`` state is
allowed to exclude data.
"""
from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any

import pandas as pd


_EXCLUDED_ROI_STATE = {
    "use": False,
    "annotation_approved": False,
    "review_required": True,
    "status": "ladder omitted",
}


def _ordered_labels(
    values: Iterable[str] | None,
    *,
    field: str,
    allow_none: bool,
) -> list[str]:
    """Validate labels and deduplicate them while retaining first-seen order."""
    if values is None:
        if allow_none:
            return []
        raise TypeError(f"{field} must be an iterable of lane-name strings, not None.")
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be an iterable of lane-name strings, not one string.")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TypeError(f"{field} must be an iterable of lane-name strings.") from exc

    ordered: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(iterator):
        if not isinstance(value, str):
            raise TypeError(f"{field}[{index}] must be a string.")
        if not value.strip():
            raise ValueError(f"{field}[{index}] must not be empty or whitespace.")
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def ordered_quantifiable_lanes(
    all_lanes: Iterable[str],
    ladder_lanes: Iterable[str] | None,
) -> list[str]:
    """Return unique non-ladder lanes in their original left-to-right order.

    Duplicate labels are collapsed at their first occurrence because the current
    BlotGrid data model uses the label itself as the lane identity.  An explicit
    ladder name that is absent from ``all_lanes`` is rejected as stale state
    rather than being silently ignored.
    """
    lanes = _ordered_labels(all_lanes, field="all_lanes", allow_none=False)
    excluded = _ordered_labels(
        ladder_lanes, field="ladder_lanes", allow_none=True
    )
    known = set(lanes)
    unknown = [lane for lane in excluded if lane not in known]
    if unknown:
        formatted = ", ".join(repr(lane) for lane in unknown)
        raise ValueError(
            f"ladder_lanes contains lane(s) absent from all_lanes: {formatted}."
        )
    excluded_set = set(excluded)
    return [lane for lane in lanes if lane not in excluded_set]


def filter_lane_rows(
    frame: pd.DataFrame | None,
    ladder_lanes: Iterable[str] | None,
    lane_col: str = "lane",
) -> pd.DataFrame | None:
    """Return a copy without explicitly excluded ladder-lane rows.

    ``None`` remains ``None`` so callers can preserve the app's not-yet-computed
    sentinel.  Empty frames and frames with no exclusions are copied without
    requiring a lane column.  A populated frame only requires ``lane_col`` when
    filtering is actually requested.
    """
    if not isinstance(lane_col, str):
        raise TypeError("lane_col must be a string.")
    if not lane_col.strip():
        raise ValueError("lane_col must not be empty or whitespace.")
    excluded = _ordered_labels(
        ladder_lanes, field="ladder_lanes", allow_none=True
    )
    if frame is None:
        return None
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame or None.")

    copied = frame.copy(deep=True)
    if copied.empty or not excluded:
        return copied
    matching_columns = sum(column == lane_col for column in copied.columns)
    if matching_columns == 0:
        raise ValueError(f"frame is missing lane column {lane_col!r}.")
    if matching_columns > 1:
        raise ValueError(f"frame contains duplicate lane column {lane_col!r}.")
    keep = ~copied[lane_col].isin(excluded)
    return copied.loc[keep].copy(deep=True)


def _has_exact_value(mapping: MutableMapping[str, Any], key: str, expected: Any) -> bool:
    value = mapping.get(key)
    if isinstance(expected, bool):
        return type(value) is bool and value is expected
    return value == expected


def enforce_excluded_rois(
    layout: MutableMapping[str, Any],
    ladder_lanes: Iterable[str] | None,
) -> bool:
    """Force ladder-lane ROIs inactive and return whether semantics changed.

    The complete ROI collection is validated before any edits, so a malformed
    entry cannot leave earlier ROIs partially updated.  When a matching ROI's
    semantic state changes, whole-image positive/negative attestation is cleared.
    Reapplying the same exclusion is idempotent.
    """
    if not isinstance(layout, MutableMapping):
        raise TypeError("layout must be a mutable mapping.")
    excluded = set(
        _ordered_labels(ladder_lanes, field="ladder_lanes", allow_none=True)
    )
    if not excluded or "rois" not in layout:
        return False

    rois = layout["rois"]
    if not isinstance(rois, list):
        raise ValueError("layout['rois'] must be a list when present.")

    validated: list[tuple[MutableMapping[str, Any], str]] = []
    for index, roi in enumerate(rois):
        if not isinstance(roi, MutableMapping):
            raise ValueError(f"layout['rois'][{index}] must be a mutable mapping.")
        lane = roi.get("lane")
        if not isinstance(lane, str):
            raise ValueError(f"layout['rois'][{index}]['lane'] must be a string.")
        if not lane.strip():
            raise ValueError(
                f"layout['rois'][{index}]['lane'] must not be empty or whitespace."
            )
        validated.append((roi, lane))

    matching = [roi for roi, lane in validated if lane in excluded]
    changed = any(
        not all(
            _has_exact_value(roi, key, expected)
            for key, expected in _EXCLUDED_ROI_STATE.items()
        )
        for roi in matching
    )
    if not changed:
        return False

    for roi in matching:
        roi.update(_EXCLUDED_ROI_STATE)
    layout["annotation_complete"] = False
    layout["negative_image"] = False
    return True

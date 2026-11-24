"""Save/load a project (ROIs, labels, settings, results) as JSON."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Project JSON contains duplicate key {key!r}.")
        payload[key] = value
    return payload


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"Project JSON contains non-finite value {value}.")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Project value of type {type(value).__name__} is not JSON serializable."
    )


def merged_image_provenance(
    previous: Mapping[str, Any] | None,
    *,
    sha256: str,
    width: int,
    height: int,
    bit_depth: int,
) -> dict[str, Any]:
    """Refresh pixel facts while retaining metadata bound to the same bytes."""
    prior = dict(previous) if isinstance(previous, Mapping) else {}
    merged = prior if prior.get("sha256") == sha256 else {}
    merged.update(
        {
            "sha256": sha256,
            "width": int(width),
            "height": int(height),
            "bit_depth": int(bit_depth),
        }
    )
    return merged


def attest_complete_annotations(layout: dict[str, Any]) -> bool:
    """Approve active boxes and close their review gate after human attestation."""
    rois = layout.get("rois")
    if not isinstance(rois, list):
        raise ValueError("A reviewed layout requires an ROI list.")
    if any(not isinstance(roi, dict) for roi in rois):
        raise ValueError("A reviewed layout contains a malformed ROI.")
    has_active_roi = False
    for roi in rois:
        active = roi.get("use", True) is True
        roi["annotation_approved"] = active
        if active:
            # Detector proposals enter the UI with review_required=true. Only
            # this whole-image human attestation is allowed to clear that gate.
            roi["review_required"] = False
            has_active_roi = True
    layout["annotation_complete"] = True
    layout["negative_image"] = not has_active_roi
    return has_active_roi


def project_json(state: dict) -> str:
    """Serialize project state to a JSON string (DataFrames become records)."""
    payload = dict(state)
    payload["_saved_at"] = datetime.now().isoformat(timespec="seconds")
    for k, v in list(payload.items()):
        if isinstance(v, pd.DataFrame):
            payload[k] = {"__dataframe__": True, "records": v.to_dict(orient="records")}
    return json.dumps(
        payload,
        indent=2,
        default=_json_default,
        allow_nan=False,
    )


def save_project(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(project_json(state))


def load_project(source) -> dict:
    """Load a project from a path, a JSON string/bytes, or a file-like object."""
    if hasattr(source, "read"):
        raw = source.read()
    elif isinstance(source, (bytes, bytearray)):
        raw = source
    elif isinstance(source, str) and source.lstrip().startswith("{"):
        raw = source
    else:
        with open(source, "r", encoding="utf-8") as f:
            raw = f.read()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    payload = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("Project JSON must contain one top-level object.")
    for k, v in list(payload.items()):
        if isinstance(v, dict) and v.get("__dataframe__"):
            payload[k] = pd.DataFrame(v["records"])
    return payload

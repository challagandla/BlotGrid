"""Streamlit wrapper for the on-image, ImageJ-style ROI editor."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit.components.v1 as components


_FRONTEND = Path(__file__).with_name("roi_editor_frontend")
_component = components.declare_component("wblot_roi_editor", path=str(_FRONTEND))


def _image_data_url(display_image: np.ndarray) -> str:
    """Encode an 8-bit display image for the self-contained canvas component."""
    arr = np.asarray(display_image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="L" if arr.ndim == 2 else None)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def edit_rois(
    display_image: np.ndarray,
    rois: list[dict],
    *,
    control_target: str,
    key: str,
    display_width: int | None = None,
    viewport_height: int | None = None,
    selected_roi_id: str | None = None,
    allow_resize: bool = True,
):
    """Render the interactive overlay and return an edit event, if any.

    display_width is the on-screen width in CSS pixels; None fits the blot to
    the panel. viewport_height caps the scrollable blot area so that a tall
    image does not push the controls beneath it off the screen.
    """
    height, width = np.asarray(display_image).shape[:2]
    safe_rois = []
    for roi in rois:
        source = str(roi.get("source", "manual"))
        status = str(roi.get("status", "manual"))
        review_required = bool(roi.get("review_required", False))
        use = bool(roi.get("use", True))
        if not use or source == "fallback":
            color = "#94a3b8"
        elif review_required or status in {"overlapping", "unresolved overlap", "low confidence"}:
            color = "#f59e0b"
        elif roi["target"] == control_target:
            color = "#ef4444"
        else:
            color = "#22c55e"
        safe_rois.append({
            "roi_id": str(roi["roi_id"]),
            "target": str(roi["target"]),
            "lane": str(roi["lane"]),
            "x": int(roi["x"]),
            "y": int(roi["y"]),
            "w": int(roi["w"]),
            "h": int(roi["h"]),
            "use": use,
            "color": color,
            "source": source,
            "status": status,
            "review_required": review_required,
            "confidence": float(roi.get("confidence", 0.0)),
        })
    return _component(
        image_url=_image_data_url(display_image),
        image_width=int(width),
        image_height=int(height),
        rois=safe_rois,
        display_width=int(display_width) if display_width else 0,
        viewport_height=int(viewport_height) if viewport_height else 0,
        selected_roi_id=str(selected_roi_id) if selected_roi_id else "",
        allow_resize=bool(allow_resize),
        default=None,
        key=key,
        tab_index=0,
    )

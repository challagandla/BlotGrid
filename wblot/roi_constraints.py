"""Server-side ROI geometry constraints for unbiased band measurement."""
from __future__ import annotations


def _clamp(value: int, low: int, high: int) -> int:
    return int(min(max(int(value), low), high))


def resize_centered(
    roi: dict,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> None:
    """Resize one ROI around its centre while keeping it inside the image."""
    width = _clamp(width, 1, image_width)
    height = _clamp(height, 1, image_height)
    cx = float(roi.get("x", 0)) + float(roi.get("w", width)) / 2.0
    cy = float(roi.get("y", 0)) + float(roi.get("h", height)) / 2.0
    roi["x"] = _clamp(round(cx - width / 2.0), 0, image_width - width)
    roi["y"] = _clamp(round(cy - height / 2.0), 0, image_height - height)
    roi["w"], roi["h"] = width, height


def enforce_shared_size(
    rois: list[dict],
    shared_size: dict,
    image_width: int,
    image_height: int,
) -> None:
    """Enforce exactly one width/height pair for the entire blot."""
    for roi in rois:
        resize_centered(
            roi, shared_size["w"], shared_size["h"], image_width, image_height)


def apply_editor_event(
    rois: list[dict],
    shared_size: dict,
    event: dict,
    image_width: int,
    image_height: int,
) -> bool:
    """Apply a canvas event without trusting its per-ROI rectangle sizes.

    A resize changes the one blot-level width and height. All incoming positions
    are accepted within image bounds, but a malicious or malformed event cannot
    assign different measurement areas to any bands.
    """
    if not isinstance(event, dict) or not isinstance(event.get("rois"), list):
        return False
    incoming = {
        str(roi.get("roi_id")): roi
        for roi in event["rois"]
        if isinstance(roi, dict) and roi.get("roi_id") is not None
    }
    selected_id = str(event.get("selected_id", ""))
    selected = next(
        (roi for roi in rois if str(roi.get("roi_id")) == selected_id), None)
    action = str(event.get("action", ""))

    if action.startswith("resize") and selected is not None:
        selected_update = incoming.get(selected_id, {})
        try:
            width = _clamp(selected_update["w"], 3, image_width)
            height = _clamp(selected_update["h"], 3, image_height)
        except (KeyError, TypeError, ValueError):
            return False
        shared_size.update(w=width, h=height)

    changed = False
    for roi in rois:
        update = incoming.get(str(roi.get("roi_id")))
        if update is None:
            continue
        try:
            x = int(round(float(update["x"])))
            y = int(round(float(update["y"])))
            incoming_w = _clamp(update.get("w", roi.get("w", 1)), 1, image_width)
            incoming_h = _clamp(update.get("h", roi.get("h", 1)), 1, image_height)
        except (KeyError, TypeError, ValueError):
            continue
        old_geometry = (roi.get("x"), roi.get("y"), roi.get("w"), roi.get("h"))
        roi.update(x=x, y=y, w=incoming_w, h=incoming_h)
        resize_centered(
            roi, shared_size["w"], shared_size["h"], image_width, image_height)
        changed = changed or old_geometry != (roi["x"], roi["y"], roi["w"], roi["h"])

    enforce_shared_size(rois, shared_size, image_width, image_height)
    return changed

"""BlotGrid — local Western blot alignment and quantification app.

Quantify band intensities from a blot image, normalize each target against a
housekeeping/loading control in the same lane, aggregate replicates, and make
publication-quality bar charts.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import os
from pathlib import Path

# Matplotlib otherwise tries to write its font cache beside the user's profile,
# which can be read-only in managed lab/workstation environments.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".cache" / "matplotlib"),
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

from wblot import (alignment, detection, image_io, densitometry, normalization, stats,
                   plotting, project_io, roi_constraints, roi_editor,
                   label_ocr)

st.set_page_config(page_title="BlotGrid", page_icon="▦", layout="wide")

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
S = st.session_state
S.setdefault("blots", {})        # name -> LoadedImage
S.setdefault("original_blots", {})  # untouched uploads used for non-cumulative rotation
S.setdefault("alignment_applied", {})  # name -> correction angle in degrees
S.setdefault("layouts", {})      # name -> layout dict
S.setdefault("replicate", {})    # name -> replicate int
S.setdefault("targets", [])
S.setdefault("control", "")
S.setdefault("lanes", [])
S.setdefault("label_scans", {})
S.setdefault("metadata_autofill_done", False)
S.setdefault("metadata_source", "")
S.setdefault("metadata_message", "Upload a labelled blot or raw scan to fill this setup automatically.")
S.setdefault("bands_dark", True)
S.setdefault("polarity_mode", "auto")
S.setdefault("detection_sensitivity", 0.65)
S.setdefault("bg_method", "local")
S.setdefault("bg_margin", 6)
S.setdefault("rb_radius", 50)
S.setdefault("bg_fixed", 0.0)
S.setdefault("multiband", "sum")
S.setdefault("measurement_mode", "profile")
S.setdefault("measurements", None)
S.setdefault("normalized", None)
S.setdefault("analysis_values", None)
S.setdefault("normalization_applied", False)
S.setdefault("summary", None)
S.setdefault("agg_raw", None)
S.setdefault("value_col", "normalized")
# Panel/preview sizing. These are view preferences, deliberately kept out of
# PROJECT_KEYS so restoring a project never rearranges someone else's screen.
S.setdefault("roi_fit_mode", "Fit panel width")
S.setdefault("roi_zoom", 100)
S.setdefault("roi_panel_height", 460)
S.setdefault("roi_side_by_side", True)
S.setdefault("roi_image_share", 62)
S.setdefault("roi_table_height", 260)
S.setdefault("thumb_width", 320)
S.setdefault("fig_w", 5.0)
S.setdefault("fig_h", 4.0)

NO_CONTROL_OPTION = "No control — intensity only"

# Settings written to / restored from a project file. Images are not embedded;
# they are re-uploaded and matched back to their layout by file name.
PROJECT_KEYS = ["targets", "control", "lanes", "bands_dark", "polarity_mode",
                "detection_sensitivity", "bg_method",
                "bg_margin", "rb_radius", "bg_fixed", "multiband",
                "measurement_mode", "layouts", "replicate"]


def clamp(v, lo, hi) -> int:
    """Keep a stored slider value inside the range this image allows."""
    return int(min(max(int(v), lo), hi))


def _invalidate_results() -> None:
    """Clear every result derived from ROI geometry or measurement settings."""
    S["measurements"] = None
    S["normalized"] = None
    S["analysis_values"] = None
    S["normalization_applied"] = False
    S["summary"] = None
    S["agg_raw"] = None
    S["value_col"] = "normalized"


def _resize_roi_centered(roi, width: int, height: int, img: image_io.LoadedImage) -> None:
    """Resize an ROI around its current centre without crossing image edges."""
    roi_constraints.resize_centered(roi, width, height, img.width, img.height)


def _ensure_shared_roi_size(layout, img: image_io.LoadedImage) -> dict:
    """Create/apply one manipulation-resistant size for every ROI on the blot."""
    stored = layout.get("roi_size")
    rois = layout.get("rois", [])
    if not isinstance(stored, dict) or "w" not in stored or "h" not in stored:
        if rois:
            width = int(np.ceil(np.quantile([int(roi.get("w", 1)) for roi in rois], 0.90)))
            height = int(np.ceil(np.quantile([int(roi.get("h", 1)) for roi in rois], 0.90)))
        else:
            width, height = max(6, int(img.width * 0.12)), max(4, int(img.height * 0.08))
        stored = {"w": clamp(width, 1, img.width), "h": clamp(height, 1, img.height)}
        layout["roi_size"] = stored
    roi_constraints.enforce_shared_size(rois, stored, img.width, img.height)
    return stored


def _ensure_roi_ids(layout) -> None:
    """Assign stable internal IDs used by the interactive on-image editor."""
    used = set()
    counter = int(layout.get("next_roi_id", 0))
    for roi in layout.get("rois", []):
        roi_id = str(roi.get("roi_id", ""))
        if not roi_id or roi_id in used:
            while f"roi_{counter}" in used:
                counter += 1
            roi_id = f"roi_{counter}"
            counter += 1
            roi["roi_id"] = roi_id
        used.add(roi_id)
    layout["next_roi_id"] = counter


def _apply_roi_editor_event(layout, event, lanes, img: image_io.LoadedImage) -> bool:
    """Apply a canvas edit while enforcing the per-lane size lock in Python."""
    _ensure_roi_ids(layout)
    shared_size = _ensure_shared_roi_size(layout, img)
    changed = roi_constraints.apply_editor_event(
        layout.get("rois", []), shared_size, event, img.width, img.height)
    layout["roi_size"] = shared_size
    _ensure_shared_roi_size(layout, img)
    return changed


def _add_manual_band_boxes(
    layout: dict,
    img: image_io.LoadedImage,
    target: str,
    lane: str,
    lanes: list[str],
    count: int = 1,
) -> list[str]:
    """Add active, locked-size ROIs near one expected target/lane position."""
    _ensure_roi_ids(layout)
    shared_size = _ensure_shared_roi_size(layout, img)
    width, height = int(shared_size["w"]), int(shared_size["h"])
    count = clamp(count, 1, 20)

    try:
        lane_index = lanes.index(lane)
    except ValueError:
        lane_index = 0
    lane_centers = list(layout.get("lane_x", []))
    if len(lane_centers) == len(lanes):
        center_x = int(lane_centers[lane_index])
    else:
        center_x = int(np.linspace(img.width * 0.18, img.width * 0.82,
                                   max(len(lanes), 1))[lane_index])

    target_entry = layout.get("targets", {}).get(target, {})
    if "y" in target_entry:
        center_y = int(target_entry["y"])
    else:
        matching_y = [
            int(roi.get("y", 0)) + int(roi.get("h", height)) // 2
            for roi in layout.get("rois", []) if roi.get("target") == target
        ]
        center_y = int(round(np.median(matching_y))) if matching_y else img.height // 2

    used_ids = {str(roi.get("roi_id")) for roi in layout.get("rois", [])}
    counter = int(layout.get("next_roi_id", 0))
    spacing = max(4, int(round(height * 0.55)))
    added_ids: list[str] = []
    for index in range(count):
        while f"roi_{counter}" in used_ids:
            counter += 1
        roi_id = f"roi_{counter}"
        used_ids.add(roi_id)
        counter += 1
        vertical_offset = int(round((index - (count - 1) / 2.0) * spacing))
        x = clamp(round(center_x - width / 2.0), 0, img.width - width)
        y = clamp(round(center_y + vertical_offset - height / 2.0),
                  0, img.height - height)
        layout.setdefault("rois", []).append({
            "roi_id": roi_id,
            "use": True,
            "target": target,
            "lane": lane,
            "x": x,
            "y": y,
            "w": width,
            "h": height,
            "confidence": 0.0,
            "source": "manual",
            "status": "manual addition",
            "review_required": False,
            "component_count": 1,
            "profile_span": 0.0,
        })
        added_ids.append(roi_id)
    layout["next_roi_id"] = counter
    return added_ids


def _clear_roi_widget_state(image_name: str) -> None:
    """Clear widgets whose values depend on a newly generated ROI layout."""
    prefixes = (f"roi_table_{image_name}", f"roi_canvas_{image_name}",
                f"roi_canvas_event_{image_name}", f"lane_roi_w_{image_name}_",
                f"lane_roi_h_{image_name}_", f"roi_width_{image_name}",
                f"roi_height_{image_name}")
    for key in [key for key in S.keys() if key.startswith(prefixes)]:
        del S[key]


def _reset_image_analysis(image_name: str) -> None:
    """Discard results whose pixels or coordinates depend on image alignment."""
    S["label_scans"].pop(image_name, None)
    S["layouts"].pop(image_name, None)
    _clear_roi_widget_state(image_name)
    _invalidate_results()


def _alignment_figure(img: image_io.LoadedImage, grid_step: int):
    """Build a display-only preview with a square alignment grid."""
    fig, ax = plt.subplots(figsize=(7.2, max(2.2, 7.2 * img.height / img.width)))
    ax.imshow(image_io.to_display(img), cmap="gray", vmin=0, vmax=255,
              interpolation="nearest")
    step = max(8, int(grid_step))
    for x in range(0, img.width, step):
        ax.axvline(x, color="#22d3ee", linewidth=0.65, alpha=0.72)
    for y in range(0, img.height, step):
        ax.axhline(y, color="#22d3ee", linewidth=0.65, alpha=0.72)
    ax.set_xlim(0, img.width - 1)
    ax.set_ylim(img.height - 1, 0)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    return fig


def default_layout(img: image_io.LoadedImage, targets, n_lanes):
    H, W = img.height, img.width
    lane_x = list(np.linspace(W * 0.18, W * 0.82, n_lanes).astype(int))
    ys = np.linspace(H * 0.2, H * 0.8, len(targets))
    return {
        "lane_x": lane_x,
        "band_width": max(12, int(W * 0.12)),
        "targets": {t: {"y": int(y), "h": max(10, int(H * 0.08))} for t, y in zip(targets, ys)},
    }


def automatic_layout(img: image_io.LoadedImage, targets, lanes):
    """Detect polarity, lane/row positions, and all individual bands."""
    mode = S.get("polarity_mode", "auto")
    scan = S.get("label_scans", {}).get(img.name)
    lane_hint = None
    target_hint = None
    if isinstance(scan, label_ocr.ExperimentLabels):
        stored_lane_centers = getattr(scan, "lane_centers", ())
        stored_target_centers = getattr(scan, "target_centers", ())
        if len(stored_lane_centers) == len(lanes):
            lane_hint = stored_lane_centers
        if len(stored_target_centers) == len(targets):
            target_hint = stored_target_centers
    result = detection.detect_bands(
        img.gray,
        targets,
        lanes,
        polarity=mode,
        sensitivity=S.get("detection_sensitivity", 0.65),
        lane_centers_hint=lane_hint,
        target_centers_hint=target_hint,
    )
    directly_detected = len(result.bands)
    fallback_count = detection.ensure_complete_grid(
        result, targets, lanes, img.gray.shape,
        gray=img.gray,
        sensitivity=S.get("detection_sensitivity", 0.65),
    )
    heights = [b.h for b in result.bands]
    widths = [b.w for b in result.bands]
    roi_rows = [dict(b.as_row(), roi_id=f"roi_{index}")
                for index, b in enumerate(result.bands)]
    roi_size = ({"w": roi_rows[0]["w"], "h": roi_rows[0]["h"]}
                if roi_rows else {"w": max(6, int(img.width * 0.12)),
                                  "h": max(4, int(img.height * 0.08))})
    target_layout = {}
    for target, cy in zip(targets, result.target_centers):
        target_heights = [b.h for b in result.bands if b.target == target]
        target_layout[target] = {
            "y": int(cy),
            "h": int(round(np.median(target_heights))) if target_heights else max(4, int(img.height * 0.08)),
        }
    return {
        "mode": "automatic",
        "rois": roi_rows,
        "roi_size": roi_size,
        # Keep the coarse-grid fields for backward-compatible project files and
        # the manual fallback.
        "lane_x": result.lane_centers,
        "band_width": int(round(np.median(widths))) if widths else max(12, int(img.width * 0.12)),
        "targets": target_layout,
        "bands_dark": result.bands_are_dark,
        "polarity_confidence": result.polarity_confidence,
        "detection_warnings": result.warnings,
        "detected_count": len(result.bands),
        "directly_detected_count": directly_detected,
        "fallback_count": fallback_count,
        "next_roi_id": len(result.bands),
    }


def rois_from_layout(layout, targets, lanes):
    """Yield (target, lane, x, y, w, h) rectangles from layout parameters."""
    if "rois" in layout:
        valid_targets, valid_lanes = set(targets), set(lanes)
        for roi in layout.get("rois", []):
            if not roi.get("use", True):
                continue
            target, lane = roi.get("target"), roi.get("lane")
            if target not in valid_targets or lane not in valid_lanes:
                continue
            try:
                x, y, w, h = (int(roi[k]) for k in ("x", "y", "w", "h"))
            except (KeyError, TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                yield target, lane, x, y, w, h
        return
    w = layout["band_width"]
    for t in targets:
        ty = layout["targets"].get(t)
        if ty is None:
            continue
        h = ty["h"]
        y = int(ty["y"] - h / 2)
        for lane, cx in zip(lanes, layout["lane_x"]):
            x = int(cx - w / 2)
            yield t, lane, x, y, w, h


def _metadata_message(labels: label_ocr.ExperimentLabels) -> str:
    target_text = ", ".join(labels.targets)
    if labels.lane_source == "printed":
        lane_text = f"printed lanes: {', '.join(labels.lanes)}"
    elif labels.lane_source == "raw_scan":
        return (
            f"Raw scan — inferred {len(labels.targets)} target rows and "
            f"{len(labels.lanes)} lanes; neutral labels were assigned."
        )
    else:
        lane_text = f"no printed lane names; inferred {len(labels.lanes)} band columns"
    return f"Read from image — targets: {target_text}; {lane_text}."


def _queue_detected_metadata(labels: label_ocr.ExperimentLabels, image_name: str) -> None:
    """Stage OCR values so keyed sidebar widgets receive them before rendering."""
    targets = list(labels.targets)
    lanes = list(labels.lanes)
    S["_pending_metadata"] = {
        "targets": targets,
        "lanes": lanes,
        "control": label_ocr.suggested_control(targets),
        "source": image_name,
        "message": _metadata_message(labels),
    }
    S["metadata_autofill_done"] = True
    # Target/lane names are part of every ROI identity. Rebuild layouts rather
    # than retaining rectangles labelled with the previous placeholders.
    S["layouts"] = {}
    _invalidate_results()


# Apply values staged by image OCR or project restore before keyed widgets are
# instantiated. Streamlit intentionally forbids changing widget state later in
# the same run.
pending_metadata = S.pop("_pending_metadata", None)
if pending_metadata:
    S["targets"] = list(pending_metadata.get("targets", []))
    S["lanes"] = list(pending_metadata.get("lanes", []))
    S["control"] = str(pending_metadata.get("control", ""))
    S["targets_input"] = "\n".join(S["targets"])
    S["lanes_input"] = "\n".join(S["lanes"])
    S["control_select"] = S["control"] or NO_CONTROL_OPTION
    S["metadata_source"] = str(pending_metadata.get("source", ""))
    S["metadata_message"] = str(pending_metadata.get("message", ""))

S.setdefault("targets_input", "\n".join(S["targets"]))
S.setdefault("lanes_input", "\n".join(S["lanes"]))
# Parse keyed text before drawing the sidebar. This also captures edits made in
# the previous Streamlit event before dependent controls are created.
S["targets"] = [line.strip() for line in S["targets_input"].splitlines() if line.strip()]
S["lanes"] = [line.strip() for line in S["lanes_input"].splitlines() if line.strip()]


# ----------------------------------------------------------------------------
# Sidebar: global experiment setup + project I/O
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Experiment setup")
    target_text = st.text_area(
        "Targets (one per line, top→bottom)", height=150, key="targets_input",
        placeholder="Automatically read after image upload")
    S["targets"] = [line.strip() for line in target_text.splitlines() if line.strip()]
    if S["targets"]:
        desired_control = (S["control"] if S["control"] in S["targets"]
                           else label_ocr.suggested_control(S["targets"]))
        control_options = [NO_CONTROL_OPTION, *S["targets"]]
        if S.get("control_select") not in control_options:
            S["control_select"] = desired_control or NO_CONTROL_OPTION
        selected_control = st.selectbox(
            "Loading control (optional)", control_options, key="control_select",
            help="Choose no control when the scan has no recognized housekeeping band. "
                 "The app will report background-corrected intensity without normalization.")
        S["control"] = "" if selected_control == NO_CONTROL_OPTION else selected_control
    else:
        S["control"] = ""
    lane_text = st.text_area(
        "Lanes / conditions (one per line, left→right)", height=90,
        key="lanes_input", placeholder="Read from labels, or inferred as Lane 1, Lane 2, …")
    S["lanes"] = [line.strip() for line in lane_text.splitlines() if line.strip()]
    st.caption(S.get("metadata_message", ""))

    st.divider()
    st.subheader("Signal & background")
    polarity_labels = {
        "Automatic (recommended)": "auto",
        "Dark bands on light background": "dark",
        "Light bands on dark background": "light",
    }
    current_polarity = S.get("polarity_mode", "auto")
    polarity_names = list(polarity_labels)
    polarity_index = next(
        (i for i, label in enumerate(polarity_names)
         if polarity_labels[label] == current_polarity), 0)
    polarity_label = st.selectbox(
        "Band polarity", polarity_names, index=polarity_index,
        help="Automatic polarity is detected separately for every uploaded blot.")
    S["polarity_mode"] = polarity_labels[polarity_label]
    if S["polarity_mode"] != "auto":
        S["bands_dark"] = S["polarity_mode"] == "dark"
    S["detection_sensitivity"] = st.slider(
        "Automatic detection sensitivity", 0.0, 1.0,
        float(S.get("detection_sensitivity", 0.65)), 0.05,
        help="Raise this to retain very faint bands; lower it if dust or noise is being selected.")
    S["bg_method"] = st.selectbox("Background subtraction",
                                  ["local", "rolling_ball", "fixed"],
                                  index=["local", "rolling_ball", "fixed"].index(S["bg_method"]))
    if S["bg_method"] == "local":
        S["bg_margin"] = st.slider("Local background margin (px)", 2, 30, S["bg_margin"])
    elif S["bg_method"] == "rolling_ball":
        S["rb_radius"] = st.slider("Rolling-ball radius (px)", 5, 200, S["rb_radius"])
    else:  # fixed
        S["bg_fixed"] = st.number_input(
            "Fixed background (signal units per pixel)",
            min_value=0.0, value=float(S["bg_fixed"]), step=1.0,
            help="Subtracted from every pixel of every ROI. In signal units "
                 "(inverted already, if 'bands are dark' is on).")
    S["multiband"] = st.selectbox("Multiple bands per target",
                                  ["sum", "mean", "keep"],
                                  index=["sum", "mean", "keep"].index(S["multiband"]))
    measurement_labels = {
        "Profile-guided peak area (recommended)": "profile",
        "Full locked rectangle (classic)": "rectangle",
    }
    current_measurement_mode = S.get("measurement_mode", "profile")
    current_measurement_label = next(
        (label for label, value in measurement_labels.items()
         if value == current_measurement_mode),
        "Profile-guided peak area (recommended)",
    )
    measurement_label = st.selectbox(
        "Band integration",
        list(measurement_labels),
        index=list(measurement_labels).index(current_measurement_label),
        help="Profile-guided integration uses the locked lane width but limits "
             "vertical integration to the dominant band peak, reducing border "
             "and neighbouring-band contamination.",
    )
    selected_measurement_mode = measurement_labels[measurement_label]
    if selected_measurement_mode != current_measurement_mode:
        S["measurement_mode"] = selected_measurement_mode
        _invalidate_results()

    st.divider()
    st.subheader("Project")
    st.download_button(
        "Save project (.json)",
        project_io.project_json({k: S.get(k) for k in PROJECT_KEYS}).encode("utf-8"),
        file_name="wblot_project.json", mime="application/json")
    proj = st.file_uploader("Load project (.json)", type=["json"], key="proj_up")
    if proj is not None and st.button("Restore project"):
        data = project_io.load_project(proj.getvalue())
        for k in PROJECT_KEYS:
            if data.get(k) is not None:
                S[k] = data[k]
        restored_targets = list(S.get("targets", []))
        restored_lanes = list(S.get("lanes", []))
        S["_pending_metadata"] = {
            "targets": restored_targets,
            "lanes": restored_lanes,
            "control": S.get("control", label_ocr.suggested_control(restored_targets)),
            "source": "saved project",
            "message": "Experiment labels restored from the saved project.",
        }
        S["metadata_autofill_done"] = True
        # ROI sliders are keyed widgets: their stored state would otherwise win
        # over the restored layout, so clear them and let the sliders rebuild.
        for wk in [k for k in S.keys()
                   if k.startswith(("lx_", "ty_", "th_", "bw_", "rep_"))]:
            del S[wk]
        st.success("Project restored. Re-upload the blot images to re-quantify.")
        st.rerun()


tabs = st.tabs(["1 · Upload", "2 · ROIs", "3 · Quantify",
                "4 · Normalize", "5 · Replicates", "6 · Plot & Export"])

# ----------------------------------------------------------------------------
# Tab 1 — Upload
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Upload blot image(s)")
    st.caption("Each image is one replicate. Add several blots to get SEM/SD across replicates.")
    st.info(
        "Privacy: images and text recognition stay on this computer. "
        "This app does not upload blots to a cloud service."
    )
    ups = st.file_uploader("Blot images (TIFF / PNG / JPG)",
                           type=["tif", "tiff", "png", "jpg", "jpeg"],
                           accept_multiple_files=True)
    if ups:
        metadata_rerun = False
        for i, up in enumerate(ups):
            name = up.name
            if name not in S["blots"]:
                uploaded_image = image_io.load_image(up.getvalue(), name=name)
                S["original_blots"][name] = uploaded_image
                S["blots"][name] = uploaded_image
                S["alignment_applied"][name] = 0.0
                _invalidate_results()
                S["replicate"].setdefault(name, len(S["replicate"]) + 1)
            elif name not in S["original_blots"]:
                # Supports sessions created before scan alignment was added.
                S["original_blots"][name] = S["blots"][name]
                S["alignment_applied"].setdefault(name, 0.0)

            if name not in S["label_scans"]:
                try:
                    with st.spinner(f"Reading labels or inferring the raw scan grid for {name}..."):
                        S["label_scans"][name] = label_ocr.read_experiment_labels_or_grid(
                            S["blots"][name])
                except ValueError as exc:
                    S["label_scans"][name] = str(exc)

            scan = S["label_scans"].get(name)
            # Streamlit preserves session objects across a source hot reload.
            # Refresh an older OCR result locally when it lacks the newer
            # spatial fields, rather than asking for another upload.
            if (isinstance(scan, label_ocr.ExperimentLabels)
                    and (not hasattr(scan, "lane_centers")
                         or not hasattr(scan, "target_centers"))):
                try:
                    with st.spinner(f"Updating local label analysis for {name}..."):
                        scan = label_ocr.read_experiment_labels_or_grid(S["blots"][name])
                        S["label_scans"][name] = scan
                except ValueError as exc:
                    scan = str(exc)
                    S["label_scans"][name] = scan
            current_lanes_are_generic = (
                S.get("lanes") == [f"Lane {index}" for index in range(1, len(S.get("lanes", [])) + 1)]
            )
            inferred_lane_upgrade = (
                isinstance(scan, label_ocr.ExperimentLabels)
                and scan.lane_source == "inferred"
                and tuple(S.get("targets", [])) == scan.targets
                and current_lanes_are_generic
                and len(scan.lanes) > len(S.get("lanes", []))
            )
            if ((not S.get("metadata_autofill_done") or inferred_lane_upgrade)
                    and isinstance(scan, label_ocr.ExperimentLabels)
                    and scan.targets and scan.lanes):
                _queue_detected_metadata(scan, name)
                metadata_rerun = True
                break

            # A restored project's layout survives image re-upload. New images
            # and layouts reset by OCR are detected with the current metadata.
            if name not in S["layouts"]:
                if S["targets"] and S["lanes"]:
                    try:
                        S["layouts"][name] = automatic_layout(
                            S["blots"][name], S["targets"], S["lanes"])
                    except ValueError as exc:
                        S["layouts"][name] = default_layout(
                            S["blots"][name], S["targets"], len(S["lanes"]))
                        S["layouts"][name]["detection_warnings"] = [str(exc)]
                else:
                    S["layouts"][name] = default_layout(
                        S["blots"][name], S["targets"], len(S["lanes"]))

        if metadata_rerun:
            st.rerun()

    if S["blots"]:
        size_col, count_col = st.columns([2, 2])
        S["thumb_width"] = size_col.slider(
            "Preview size (px)", 120, 900, int(S["thumb_width"]), 20,
            help="Previews are shown at this width whatever the blot's own pixel size.")
        per_row = count_col.slider("Previews per row", 1, 4, min(3, len(S["blots"])))
        cols = st.columns(min(per_row, len(S["blots"])))
        for i, (name, img) in enumerate(S["blots"].items()):
            with cols[i % len(cols)]:
                st.image(image_io.to_display(img), width=int(S["thumb_width"]),
                         caption=f"{name}  ·  {img.width}×{img.height}  ·  {img.bit_depth}-bit", clamp=True)
                with st.expander("Straighten scan with alignment grid", expanded=False):
                    original = S["original_blots"].get(name, img)
                    angle_key = f"align_angle_{name}"
                    pending_angle_key = f"_pending_align_angle_{name}"
                    if pending_angle_key in S:
                        S[angle_key] = float(S.pop(pending_angle_key))
                    if angle_key not in S:
                        S[angle_key] = float(S["alignment_applied"].get(name, 0.0))
                    if st.button("Estimate angle automatically", key=f"auto_align_{name}"):
                        with st.spinner("Estimating the band angle locally..."):
                            S[angle_key] = alignment.estimate_correction_angle(original.gray)
                        st.rerun()
                    angle = st.slider(
                        "Correction angle (degrees)", -15.0, 15.0,
                        step=0.1, key=angle_key,
                        help="Positive rotates counter-clockwise. Adjust until bands follow the horizontal grid lines.",
                    )
                    max_grid = max(12, min(300, min(original.width, original.height) // 2))
                    default_grid = min(max_grid, max(12, min(original.width, original.height) // 6))
                    grid_step = st.slider(
                        "Grid spacing (px)", 8, max_grid, default_grid,
                        key=f"align_grid_{name}",
                    )
                    preview = alignment.rotate_loaded(original, angle)
                    align_fig = _alignment_figure(preview, grid_step)
                    st.pyplot(align_fig, width="stretch")
                    plt.close(align_fig)
                    st.caption(
                        "The cyan grid is a visual guide only and is never written into the image. "
                        "Apply the correction before band detection and quantification."
                    )
                    apply_col, reset_col = st.columns(2)
                    if apply_col.button("Apply straightening", type="primary",
                                        key=f"apply_align_{name}"):
                        S["blots"][name] = alignment.rotate_loaded(original, angle)
                        S["alignment_applied"][name] = float(angle)
                        _reset_image_analysis(name)
                        st.rerun()
                    if reset_col.button("Reset to uploaded image", key=f"reset_align_{name}"):
                        S["blots"][name] = alignment.rotate_loaded(original, 0.0)
                        S["alignment_applied"][name] = 0.0
                        S[pending_angle_key] = 0.0
                        _reset_image_analysis(name)
                        st.rerun()
                    applied_angle = float(S["alignment_applied"].get(name, 0.0))
                    st.caption(f"Working scan correction: {applied_angle:+.1f}°")
                S["replicate"][name] = st.number_input(f"Replicate # for {name}", 1, 99,
                                                       S["replicate"].get(name, 1), key=f"rep_{name}")
                layout = S["layouts"].get(name, {})
                dark = layout.get("bands_dark", S["bands_dark"])
                sat = image_io.saturated_mask(img, bands_are_dark=dark).mean() * 100
                if sat > 0.1:
                    st.warning(f"{sat:.1f}% saturated pixels — bands may be out of linear range.")
                scan = S["label_scans"].get(name)
                if isinstance(scan, label_ocr.ExperimentLabels):
                    if scan.targets:
                        confidence = int(round(scan.target_confidence * 100))
                        if scan.lane_source == "printed":
                            lane_note = f"lanes: {', '.join(scan.lanes)}"
                        elif scan.lane_source == "raw_scan":
                            lane_note = (
                                f"raw grid: {len(scan.targets)} target rows × "
                                f"{len(scan.lanes)} lanes"
                            )
                        else:
                            lane_note = f"{len(scan.lanes)} unlabeled columns inferred"
                        if scan.lane_source == "raw_scan":
                            st.caption(
                                f"Raw scan grid ({confidence}% confidence): {lane_note}")
                        else:
                            st.caption(
                                f"Text recognition ({confidence}%): "
                                f"{', '.join(scan.targets)} · {lane_note}")
                        if st.button("Apply these labels", key=f"apply_labels_{name}"):
                            _queue_detected_metadata(scan, name)
                            st.rerun()
                        if scan.warning:
                            st.caption(scan.warning)
                    else:
                        st.warning(scan.warning)
                elif scan:
                    st.warning(str(scan))
        if st.button("Clear all blots"):
            for k in ("blots", "original_blots", "alignment_applied", "layouts",
                      "replicate", "label_scans"):
                S[k] = {}
            _invalidate_results()
            S["metadata_autofill_done"] = False
            S["metadata_source"] = ""
            S["metadata_message"] = "Upload a labelled blot or raw scan to fill this setup automatically."
            st.rerun()
    else:
        st.info("No image yet. Try `python make_sample_blot.py` to create sample_blot.tif, then upload it.")

# ----------------------------------------------------------------------------
# Tab 2 — ROIs
# ----------------------------------------------------------------------------
with tabs[1]:
    if not S["blots"]:
        st.info("Upload a blot first.")
    elif not S["targets"] or not S["lanes"]:
        st.error("Add at least one target and one lane in Experiment setup.")
    else:
        name = st.selectbox("Blot to lay out", list(S["blots"].keys()))
        img = S["blots"][name]
        layout = S["layouts"].setdefault(name, default_layout(img, S["targets"], len(S["lanes"])))
        detect_col, fallback_col, status_col = st.columns([1.1, 1.0, 2.2])
        if detect_col.button("Detect all bands", type="primary", key=f"detect_{name}"):
            try:
                layout = automatic_layout(img, S["targets"], S["lanes"])
                S["layouts"][name] = layout
                _invalidate_results()
                _clear_roi_widget_state(name)
            except ValueError as exc:
                st.error(str(exc))
        if fallback_col.button("Use even grid", key=f"grid_{name}"):
            layout = default_layout(img, S["targets"], len(S["lanes"]))
            layout["mode"] = "grid"
            if S["polarity_mode"] == "auto":
                dark, confidence = detection.infer_band_polarity(
                    img.gray, len(S["lanes"]), len(S["targets"]))
                layout["bands_dark"] = dark
                layout["polarity_confidence"] = confidence
            else:
                layout["bands_dark"] = S["polarity_mode"] == "dark"
                layout["polarity_confidence"] = 1.0
            S["layouts"][name] = layout
            _invalidate_results()
            _clear_roi_widget_state(name)

        if "rois" in layout:
            shared_roi_size = _ensure_shared_roi_size(layout, img)
            _ensure_roi_ids(layout)
            active_count = sum(bool(r.get("use", True)) for r in layout["rois"])
            review_count = sum(bool(r.get("review_required", False)) for r in layout["rois"])
            excluded_count = len(layout["rois"]) - active_count
            polarity_text = "dark on light" if layout.get("bands_dark", True) else "light on dark"
            status_text = (
                f"{active_count} active · {excluded_count} excluded · "
                f"{review_count} need review · {polarity_text}"
            )
            if review_count:
                status_col.warning(status_text)
            else:
                status_col.success(status_text)
            for warning in layout.get("detection_warnings", []):
                st.warning(warning)

            with st.expander("Display & panel size", expanded=False):
                view_cols = st.columns([1.2, 1.2, 1.2, 1.4])
                S["roi_fit_mode"] = view_cols[0].radio(
                    "Blot scale", ["Fit panel width", "Zoom %"],
                    index=0 if S["roi_fit_mode"] == "Fit panel width" else 1,
                    key=f"roi_fit_{name}",
                    help="Zoom % shows the blot at a fixed magnification and lets "
                         "you scroll it, which is what you want for pixel-accurate edges.")
                S["roi_zoom"] = view_cols[1].slider(
                    "Zoom", 25, 400, int(S["roi_zoom"]), 25, format="%d%%",
                    disabled=S["roi_fit_mode"] == "Fit panel width",
                    key=f"roi_zoom_{name}")
                S["roi_panel_height"] = view_cols[2].slider(
                    "Blot panel height (px)", 200, 1200, int(S["roi_panel_height"]), 20,
                    key=f"roi_ph_{name}",
                    help="You can also drag the bottom-right corner of the blot panel.")
                S["roi_side_by_side"] = view_cols[3].toggle(
                    "Side-by-side panels", bool(S["roi_side_by_side"]),
                    key=f"roi_sbs_{name}")
                if S["roi_side_by_side"]:
                    S["roi_image_share"] = view_cols[3].slider(
                        "Blot panel width share", 30, 80, int(S["roi_image_share"]), 2,
                        format="%d%%", key=f"roi_share_{name}")
                S["roi_table_height"] = view_cols[3].slider(
                    "Table height (px)", 150, 800, int(S["roi_table_height"]), 10,
                    key=f"roi_th_{name}")

            display_width = (None if S["roi_fit_mode"] == "Fit panel width"
                             else int(img.width * S["roi_zoom"] / 100))
            if S["roi_side_by_side"]:
                share = int(S["roi_image_share"])
                image_panel, control_panel = st.columns([share, 100 - share], gap="medium")
            else:
                image_panel, control_panel = st.container(), st.container()

            with image_panel:
                st.markdown("**Adjust rectangles directly on the blot**")
                selected_roi_id = S.pop(f"_select_roi_{name}", None)
                editor_event = roi_editor.edit_rois(
                    image_io.to_display(img),
                    layout["rois"],
                    control_target=S["control"],
                    key=f"roi_canvas_{name}",
                    display_width=display_width,
                    viewport_height=int(S["roi_panel_height"]),
                    selected_roi_id=selected_roi_id,
                )
                st.caption(
                    "Green = accepted · red = accepted loading control · "
                    "orange = review/overlap · dashed grey = excluded fallback."
                )
            if isinstance(editor_event, dict) and editor_event.get("event_id"):
                event_key = f"roi_canvas_event_{name}"
                event_id = str(editor_event["event_id"])
                if S.get(event_key) != event_id:
                    S[event_key] = event_id
                    if _apply_roi_editor_event(layout, editor_event, S["lanes"], img):
                        _invalidate_results()
                    st.rerun()

            stale_targets = {r.get("target") for r in layout["rois"]} - set(S["targets"])
            stale_lanes = {r.get("lane") for r in layout["rois"]} - set(S["lanes"])
            if stale_targets or stale_lanes:
                with image_panel:
                    st.warning("Target or lane labels changed after detection. Run Detect all bands again.")

            with control_panel:
                with st.expander("Add missed bands manually", expanded=False):
                    st.caption(
                        "Choose the target and lane, add one or more extra boxes, "
                        "then drag each box directly over a missed band on the blot."
                    )
                    add_columns = st.columns([1.2, 1.2, 0.8])
                    manual_target = add_columns[0].selectbox(
                        "Target", S["targets"], key=f"manual_target_{name}")
                    manual_lane = add_columns[1].selectbox(
                        "Lane", S["lanes"], key=f"manual_lane_{name}")
                    manual_count = add_columns[2].number_input(
                        "Boxes", 1, 20, 1, 1, key=f"manual_count_{name}")
                    if st.button(
                            "Add manual band box",
                            type="primary",
                            key=f"add_manual_roi_{name}",
                            width="stretch"):
                        added_ids = _add_manual_band_boxes(
                            layout, img, manual_target, manual_lane,
                            S["lanes"], int(manual_count))
                        _invalidate_results()
                        _clear_roi_widget_state(name)
                        if added_ids:
                            S[f"_select_roi_{name}"] = added_ids[-1]
                        st.rerun()
                    st.caption(
                        "New boxes use the same locked width and height as every "
                        "other ROI. Repeat this for any number of missed bands; "
                        "delete unwanted rows from the table below."
                    )

                with st.expander("Locked rectangle size", expanded=True):
                    st.caption("These dimensions are applied to every measurement ROI. Dragging any on-image handle updates the same values automatically.")
                    with st.form(f"roi_size_form_{name}"):
                        size_columns = st.columns(2)
                        width = size_columns[0].number_input(
                            "Width (px)", 1, img.width, int(shared_roi_size["w"]), 1,
                            key=f"roi_width_{name}")
                        height = size_columns[1].number_input(
                            "Height (px)", 1, img.height, int(shared_roi_size["h"]), 1,
                            key=f"roi_height_{name}")
                        apply_roi_size = st.form_submit_button(
                            "Apply size to every band", type="primary")
                    if apply_roi_size:
                        layout["roi_size"] = {"w": int(width), "h": int(height)}
                        _ensure_shared_roi_size(layout, img)
                        _invalidate_results()
                        S.pop(f"roi_table_{name}", None)
                        st.rerun()

                roi_columns = [
                    "roi_id", "use", "status", "review_required", "source",
                    "target", "lane", "x", "y", "w", "h", "confidence",
                    "component_count", "profile_span",
                ]
                roi_df = pd.DataFrame(layout["rois"])
                roi_defaults = {
                    "use": True,
                    "status": "manual",
                    "review_required": False,
                    "source": "manual",
                    "confidence": 0.0,
                    "component_count": 1,
                    "profile_span": 0.0,
                }
                for col in roi_columns:
                    if col not in roi_df:
                        roi_df[col] = roi_defaults.get(col)
                roi_df = roi_df[roi_columns]
                edited = st.data_editor(
                    roi_df,
                    key=f"roi_table_{name}",
                    num_rows="dynamic",
                    hide_index=True,
                    width="stretch",
                    height=int(S["roi_table_height"]),
                    column_config={
                        "roi_id": None,
                        "use": st.column_config.CheckboxColumn("Use", default=True),
                        "status": st.column_config.TextColumn("Status", disabled=True),
                        "review_required": st.column_config.CheckboxColumn(
                            "Review", disabled=True),
                        "source": st.column_config.TextColumn("Source", disabled=True),
                        "target": st.column_config.SelectboxColumn(
                            "Target", options=S["targets"], required=True),
                        "lane": st.column_config.SelectboxColumn(
                            "Lane", options=S["lanes"], required=True),
                        "x": st.column_config.NumberColumn("X", min_value=0, max_value=img.width - 1, step=1),
                        "y": st.column_config.NumberColumn("Y", min_value=0, max_value=img.height - 1, step=1),
                        "w": st.column_config.NumberColumn(
                            "Width", min_value=1, max_value=img.width, step=1, disabled=True),
                        "h": st.column_config.NumberColumn(
                            "Height", min_value=1, max_value=img.height, step=1, disabled=True),
                        "confidence": st.column_config.NumberColumn(
                            "Confidence", min_value=0.0, max_value=1.0, format="%.2f", disabled=True),
                        "component_count": st.column_config.NumberColumn(
                            "Components", min_value=0, format="%d", disabled=True),
                        "profile_span": st.column_config.NumberColumn(
                            "Peak span", min_value=0.0, format="%.1f", disabled=True),
                    },
                )
                st.caption(
                    "Fallback ROIs are excluded by default. Enable Use only after "
                    "checking the blot. Width/height remain globally locked."
                )

                profile_rois = list(layout["rois"])
                if profile_rois:
                    with st.expander(
                        "Band review & intensity profile",
                        expanded=bool(review_count),
                    ):
                        profile_ids = [str(roi["roi_id"]) for roi in profile_rois]
                        profile_by_id = {str(roi["roi_id"]): roi for roi in profile_rois}
                        profile_labels = {
                            str(roi["roi_id"]):
                                f"{roi.get('target')} · {roi.get('lane')} · "
                                f"{roi.get('status', 'manual')} · "
                                f"confidence {float(roi.get('confidence', 0.0)):.2f}"
                            for roi in profile_rois
                        }
                        first_review = next(
                            (index for index, roi in enumerate(profile_rois)
                             if roi.get("review_required")), 0)
                        selected_profile_id = st.selectbox(
                            "ROI to inspect",
                            profile_ids,
                            index=first_review,
                            format_func=lambda roi_id: profile_labels[roi_id],
                            key=f"profile_roi_{name}",
                        )
                        profile_roi = profile_by_id[selected_profile_id]
                        px = clamp(profile_roi.get("x", 0), 0, img.width - 1)
                        py = clamp(profile_roi.get("y", 0), 0, img.height - 1)
                        pw = clamp(profile_roi.get("w", 1), 1, img.width - px)
                        ph = clamp(profile_roi.get("h", 1), 1, img.height - py)
                        profile_signal = image_io.to_signal(
                            img, layout.get("bands_dark", S["bands_dark"]))
                        profile_pixels = profile_signal[py:py + ph, px:px + pw]
                        preview_background = float(np.percentile(profile_pixels, 20.0))
                        local_y0, local_y1, intensity_profile = (
                            densitometry.profile_peak_interval(
                                profile_pixels, preview_background))
                        profile_y = np.arange(py, py + len(intensity_profile))
                        profile_figure, profile_axis = plt.subplots(figsize=(4.5, 2.2))
                        profile_axis.plot(profile_y, intensity_profile, color="#2563eb", lw=1.8)
                        profile_axis.axvspan(
                            py + local_y0, py + local_y1,
                            color="#22c55e", alpha=0.20,
                            label="profile-guided integration",
                        )
                        profile_axis.set_xlabel("Image Y position (px)")
                        profile_axis.set_ylabel("Mean signal")
                        profile_axis.legend(frameon=False, fontsize=8)
                        profile_axis.spines[["top", "right"]].set_visible(False)
                        profile_figure.tight_layout()
                        st.pyplot(profile_figure)
                        plt.close(profile_figure)
                        st.caption(
                            "The green interval is the vertical area used by "
                            "profile-guided integration; the full locked width "
                            "is retained for consistent lane sampling."
                        )

            old_signature = [
                tuple(roi.get(key) for key in ("roi_id", "use", "target", "lane", "x", "y", "w", "h"))
                for roi in layout["rois"]
            ]
            cleaned_rois = []
            for record in edited.to_dict(orient="records"):
                if record.get("target") not in S["targets"] or record.get("lane") not in S["lanes"]:
                    continue
                try:
                    x, y = (int(record[k]) for k in ("x", "y"))
                except (KeyError, TypeError, ValueError):
                    continue
                x, y = clamp(x, 0, img.width - 1), clamp(y, 0, img.height - 1)
                w, h = int(shared_roi_size["w"]), int(shared_roi_size["h"])
                confidence = record.get("confidence", 0.0)
                confidence = 0.0 if pd.isna(confidence) else float(confidence)
                use = record.get("use", True)
                use = True if pd.isna(use) else bool(use)
                review_required = record.get("review_required", False)
                review_required = False if pd.isna(review_required) else bool(review_required)
                component_count = record.get("component_count", 1)
                component_count = 1 if pd.isna(component_count) else max(0, int(component_count))
                profile_span = record.get("profile_span", 0.0)
                profile_span = 0.0 if pd.isna(profile_span) else max(0.0, float(profile_span))
                try:
                    previous_w, previous_h = int(record.get("w", w)), int(record.get("h", h))
                except (TypeError, ValueError):
                    previous_w, previous_h = w, h
                cleaned_roi = {
                    "roi_id": str(record.get("roi_id", "")),
                    "use": use, "target": record["target"], "lane": record["lane"],
                    "x": x, "y": y, "w": previous_w, "h": previous_h,
                    "confidence": confidence,
                    "source": str(record.get("source") or "manual"),
                    "status": str(record.get("status") or "manual"),
                    "review_required": review_required,
                    "component_count": component_count,
                    "profile_span": profile_span,
                }
                _resize_roi_centered(cleaned_roi, w, h, img)
                cleaned_rois.append(cleaned_roi)
            new_signature = [
                tuple(roi.get(key) for key in ("roi_id", "use", "target", "lane", "x", "y", "w", "h"))
                for roi in cleaned_rois
            ]
            if new_signature != old_signature:
                _invalidate_results()
            layout["rois"] = cleaned_rois
            _ensure_roi_ids(layout)
        else:
            status_col.info("Even-grid fallback: align the shared boxes manually.")
            # ensure the legacy coarse layout has all current targets/lanes
            if len(layout["lane_x"]) != len(S["lanes"]):
                layout["lane_x"] = list(np.linspace(img.width * 0.18, img.width * 0.82, len(S["lanes"])).astype(int))
            for t in S["targets"]:
                layout["targets"].setdefault(t, {"y": img.height // 2, "h": max(10, int(img.height * 0.08))})

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Band width & lane centres**")
                bw_max = max(7, int(img.width * 0.4))
                layout["band_width"] = st.slider(
                    "Band width (px)", 6, bw_max,
                    clamp(layout["band_width"], 6, bw_max), key=f"bw_{name}")
                for j, lane in enumerate(S["lanes"]):
                    layout["lane_x"][j] = st.slider(
                        f"Lane centre X · {lane}", 0, img.width,
                        clamp(layout["lane_x"][j], 0, img.width), key=f"lx_{name}_{j}")
            with c2:
                st.markdown("**Row Y-position & height per target**")
                h_max = max(5, int(img.height * 0.3))
                for t in S["targets"]:
                    tt = layout["targets"][t]
                    tt["y"] = st.slider(f"Y centre · {t}", 0, img.height,
                                        clamp(tt["y"], 0, img.height), key=f"ty_{name}_{t}")
                    tt["h"] = st.slider(f"Height · {t}", 4, h_max,
                                        clamp(tt["h"], 4, h_max), key=f"th_{name}_{t}")

        if "rois" not in layout:
            # Static preview for the coarse even-grid fallback. Automatic ROIs
            # use the interactive on-image editor above.
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(image_io.to_display(img), cmap="gray")
            for t, lane, x, y, w, h in rois_from_layout(layout, S["targets"], S["lanes"]):
                color = "red" if t == S["control"] else "yellow"
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, lw=1.2))
                ax.text(x, y - 2, f"{t[:6]}·{lane[:4]}", color=color, fontsize=6)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Red = loading control; yellow = targets. Every visible band should have its own box before quantification.")

# ----------------------------------------------------------------------------
# Tab 3 — Quantify
# ----------------------------------------------------------------------------
with tabs[2]:
    if not S["blots"]:
        st.info("Upload a blot first.")
    elif st.button("Run densitometry on all blots", type="primary"):
        rows = []
        for name, img in S["blots"].items():
            layout = S["layouts"][name]
            bands_dark = layout.get("bands_dark", S["bands_dark"])
            sig = image_io.to_signal(img, bands_dark)
            bg_image = densitometry.rolling_ball_background(sig, S["rb_radius"]) if S["bg_method"] == "rolling_ball" else None
            for t, lane, x, y, w, h in rois_from_layout(layout, S["targets"], S["lanes"]):
                m = densitometry.measure_band(
                    sig, t, lane, x, y, w, h,
                    method=S["bg_method"], margin=S["bg_margin"],
                    bg_image=bg_image, bg_value=S["bg_fixed"],
                    max_value=img.max_value,
                    measurement_mode=S.get("measurement_mode", "profile"),
                )
                row = m.as_row()
                row["blot"] = name
                row["replicate"] = S["replicate"].get(name, 1)
                rows.append(row)
        S["measurements"] = pd.DataFrame(rows)
        if not rows:
            st.warning("No active band boxes were available. Detect bands or add a box in the ROIs tab.")

    if S["measurements"] is not None and not S["measurements"].empty:
        df = S["measurements"]
        st.dataframe(df[["blot", "replicate", "target", "lane", "measurement_mode",
                         "profile_y0", "profile_y1", "raw_intden",
                         "background_per_px", "net_intden", "saturated"]],
                     width="stretch", height=320)
        nsat = int(df["saturated"].sum())
        if nsat:
            st.warning(f"{nsat} band(s) contain saturated pixels — interpret those with caution.")
        if (df["net_intden"] <= 0).any():
            st.warning("Some bands have net intensity ≤ 0 (background ≥ signal). Check ROI / background settings.")

# ----------------------------------------------------------------------------
# Tab 4 — Normalize
# ----------------------------------------------------------------------------
with tabs[3]:
    if S["measurements"] is None or S["measurements"].empty:
        st.info("Run Quantify first.")
    elif not normalization.has_usable_control(S["measurements"], S["control"]):
        S["normalized"] = None
        S["analysis_values"] = None
        S["normalization_applied"] = False
        parts = []
        try:
            for name, group in S["measurements"].groupby("blot"):
                intensity = normalization.prepare_intensities(
                    group, value_col="net_intden", multiband=S["multiband"])
                intensity["blot"] = name
                intensity["replicate"] = S["replicate"].get(name, 1)
                parts.append(intensity)
        except ValueError as exc:
            st.error(f"{exc} Re-run Quantify after changing the experiment setup.")
            parts = []

        if parts:
            intensity_all = pd.concat(parts, ignore_index=True)
            S["normalized"] = None
            S["analysis_values"] = intensity_all
            S["normalization_applied"] = False
            S["value_col"] = "net_intden"
            show_cols = ["blot", "replicate", "target", "lane", "net_intden"]
            if "band_index" in intensity_all.columns:
                show_cols.insert(4, "band_index")
            if S["control"]:
                st.info(
                    f"No measured {S['control']} control band was found. "
                    "Normalization was skipped and background-corrected band "
                    "intensities are shown instead."
                )
            else:
                st.info(
                    "No loading control is selected. Normalization was skipped and "
                    "background-corrected band intensities are shown instead."
                )
            st.dataframe(intensity_all[show_cols], width="stretch", height=340)
            st.caption("Intensity = background-corrected net integrated density (arbitrary units).")
    else:
        S["normalized"] = None
        S["analysis_values"] = None
        S["normalization_applied"] = False
        do_fc = st.checkbox("Also express as fold-change vs a reference lane")
        ref_lane = st.selectbox("Reference lane", S["lanes"]) if do_fc else None
        parts = []
        try:
            for name, g in S["measurements"].groupby("blot"):
                norm = normalization.normalize_to_control(
                    g, S["control"], value_col="net_intden", multiband=S["multiband"])
                if do_fc:
                    norm = normalization.fold_change(norm, ref_lane, value_col="normalized")
                norm["blot"] = name
                norm["replicate"] = S["replicate"].get(name, 1)
                parts.append(norm)
        except ValueError as exc:
            st.error(f"{exc} Re-run Quantify after changing targets or the control.")
            parts = []

        if parts:
            norm_all = pd.concat(parts, ignore_index=True)
            S["normalized"] = norm_all
            S["analysis_values"] = norm_all
            S["normalization_applied"] = True
            S["value_col"] = "fold_change" if do_fc else "normalized"
            show_cols = ["blot", "replicate", "target", "lane", "normalized"] + (["fold_change"] if do_fc else [])
            st.dataframe(norm_all[show_cols], width="stretch", height=340)
            st.caption(f"normalized[target, lane] = net_intden(target,lane) / net_intden({S['control']},lane)")
            if norm_all["normalized"].isna().any():
                st.warning("Some lanes have no usable loading control (control = 0 or "
                           "missing); those values are NaN and are dropped from the stats.")

# ----------------------------------------------------------------------------
# Tab 5 — Replicates
# ----------------------------------------------------------------------------
with tabs[4]:
    if S.get("analysis_values") is None:
        st.info("Run Quantify first, then prepare the results in Normalize.")
    else:
        nd = S["analysis_values"].copy()
        nd["condition"] = nd["lane"]  # lane label IS the shared condition across replicates
        vcol = S.get(
            "value_col",
            "normalized" if S.get("normalization_applied") else "net_intden",
        )
        if vcol not in nd.columns:
            vcol = "normalized" if S.get("normalization_applied") else "net_intden"
        summary = stats.aggregate(nd, vcol, group_cols=("target", "condition"))
        S["summary"] = summary
        S["agg_raw"] = nd
        S["value_col"] = vcol
        st.dataframe(summary, width="stretch")
        st.caption("SEM = SD / √N. N = number of replicate blots per condition. "
                   "ci95 is the t-based 95% CI half-width.")

        st.markdown("**Optional significance test**")
        stat_opts = [t for t in S["targets"] if t != S["control"]] or S["targets"]
        tgt = st.selectbox("Target", stat_opts, key="stat_tgt")
        if st.button("Compare conditions"):
            res = stats.compare_groups(nd, vcol, target=tgt)
            st.write(res)
            if res.get("p_value") is not None:
                st.info(f"{res['test']}: p = {res['p_value']:.4g}  →  {stats.p_to_stars(res['p_value'])}")

# ----------------------------------------------------------------------------
# Tab 6 — Plot & Export
# ----------------------------------------------------------------------------
with tabs[5]:
    if S.get("summary") is None:
        st.info("Run Replicates first.")
    else:
        summary = S["summary"]
        vcol = S.get(
            "value_col",
            "normalized" if S.get("normalization_applied") else "net_intden",
        )
        target_opts = [t for t in S["targets"] if t != S["control"]] or S["targets"]
        tgt = st.selectbox("Target to plot", target_opts)
        sub = summary[summary["target"] == tgt].copy()
        raw = S["agg_raw"]
        raw_pts = raw[raw["target"] == tgt][["condition", vcol]].rename(columns={vcol: "value"})
        if sub.empty or sub["mean"].isna().all():
            suffix = " and the loading control" if S.get("normalization_applied") else ""
            st.warning(f"No usable values for {tgt} — check the ROIs{suffix}.")
            st.stop()

        panel_share = st.slider("Settings panel width share", 20, 70, 45, 5,
                                format="%d%%", key="plot_share",
                                help="Drag toward the left to give the chart preview more room.")
        left, right = st.columns([panel_share, 100 - panel_share], gap="medium")
        with left:
            o = plotting.PlotOptions()
            o.error_type = st.selectbox("Error bars", ["sem", "sd", "ci95", "none"])
            o.show_points = st.checkbox("Show data points", True)
            o.palette = st.selectbox("Palette", list(plotting.PALETTES.keys()))
            o.point_color = st.color_picker("Point color", "#333333")
            o.font_family = st.selectbox("Font", ["DejaVu Sans", "Arial", "Helvetica", "Times New Roman"])
            o.title = st.text_input("Title", tgt)
            default_ylabel = (f"Relative expression (norm. {S['control']})"
                              if S.get("normalization_applied")
                              else "Net integrated density (a.u.)")
            o.ylabel = st.text_input("Y-axis label", default_ylabel)
            o.xlabel = st.text_input("X-axis label", "")
            c1, c2 = st.columns(2)
            o.title_size = c1.number_input("Title size", 6, 30, 14)
            o.label_size = c2.number_input("Label size", 6, 24, 12)
            o.tick_size = c1.number_input("Tick size", 6, 20, 10)
            o.dpi = c2.selectbox("Export DPI", [150, 300, 600], index=1)
            # Figure size in inches is what fixes the exported figure's physical
            # dimensions; the preview below is just this figure scaled to fit.
            S["fig_w"] = c1.number_input("Figure width (in)", 2.0, 20.0,
                                         float(S["fig_w"]), 0.5)
            S["fig_h"] = c2.number_input("Figure height (in)", 2.0, 20.0,
                                         float(S["fig_h"]), 0.5)
            o.fig_w, o.fig_h = float(S["fig_w"]), float(S["fig_h"])
            o.despine = st.checkbox("Remove top/right spines", True)
            o.gridlines = st.checkbox("Y gridlines", False)
            o.log_y = st.checkbox("Log Y axis", False)
            o.show_legend = st.checkbox("Show legend", False)
            o.show_n = st.checkbox("Annotate N", True)
            o.caption = st.text_input("Figure caption", "")
            manual_y = st.checkbox("Manual Y-limit")
            if manual_y:
                ymax = st.number_input("Y max", value=float(np.nanmax(sub["mean"]) * 1.4))
                o.ylim = (0, ymax)

        fig = plotting.bar_chart(sub, o, raw_points=raw_pts, order=S["lanes"],
                                 value_col="mean")
        with right:
            st.pyplot(fig)

        st.markdown("**Export**")
        ec = st.columns(4)
        for i, fmt in enumerate(["png", "tiff", "svg", "pdf"]):
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=o.dpi, bbox_inches="tight")
            ec[i].download_button(f"{fmt.upper()}", buf.getvalue(),
                                  file_name=f"{tgt}_barchart.{fmt}",
                                  mime="application/octet-stream", key=f"dl_{fmt}")
        plt.close(fig)

        st.markdown("**Data export**")
        dc = st.columns(2)
        dc[0].download_button("Summary CSV", summary.to_csv(index=False).encode(),
                              file_name="summary.csv", mime="text/csv")
        # Keep workbook creation genuinely in memory. openpyxl serializes worksheets
        # through the system TEMP directory, which can fail when C: has no free space.
        from wblot.excel_export import dataframes_to_xlsx

        export_tables = {"summary": summary}
        if S["normalized"] is not None:
            export_tables["normalized"] = S["normalized"]
        else:
            export_tables["intensity_values"] = S["analysis_values"]
        if S["measurements"] is not None:
            export_tables["raw_densitometry"] = S["measurements"]
        xbuf = io.BytesIO(dataframes_to_xlsx(export_tables))
        dc[1].download_button("All data XLSX", xbuf.getvalue(),
                              file_name="western_blot_data.xlsx",
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

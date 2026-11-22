"""Quantify every band of a composite (multi-panel) Western blot figure.

Usage:
    python scripts/quantify_composite.py IMAGE --layout LAYOUT.json --out OUTDIR [--runs 3]

The layout JSON names the figure's panel rows and columns:

{
  "row_blocks": [ {"name": "A", "targets": ["Target 1", ..., "Loading control"]},
                  {"name": "B", "targets": [...]} ],
  "col_blocks": [ {"condition": "OE", "cell_lines": ["Line 1", ...]},
                  {"condition": "KD", "cell_lines": [...]} ],
  "control_target": "Loading control",
  "lanes": ["Control", "{variant}-{condition}"]
}

Each run re-places every ROI independently (seeded +/-1 px shifts, a different
ROI width and frame inset), mimicking three separate manual ImageJ passes, so
the spread across runs measures how sensitive each value is to ROI placement.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wblot.composite import (Redraw, TileSettings, assign_grid,  # noqa: E402
                             find_framed_panels, quantify_panel)

# Band detection runs once with fixed settings.  Each repeat measurement moves
# every box by a seeded random -1/0/+1 px in x and y (independently per box)
# and changes the tile's shared box size by -1/0/+1 px in width and height.
# Detection settings are not varied between runs: that would change the
# measured region, not repeat the measurement.
DETECTION = TileSettings()
REDRAW_MAX_PX = 1


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(float)
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def label_panels(panels, layout):
    rows_per_block = [len(b["targets"]) for b in layout["row_blocks"]]
    cols_per_block = [len(b["cell_lines"]) for b in layout["col_blocks"]]
    n_rows = max(p.row for p in panels) + 1
    n_cols = max(p.col for p in panels) + 1
    if n_rows != sum(rows_per_block) or n_cols != sum(cols_per_block):
        raise SystemExit(
            f"Found a {n_rows} x {n_cols} panel grid but the layout describes "
            f"{sum(rows_per_block)} x {sum(cols_per_block)}.")
    row_info, col_info = {}, {}
    r = 0
    for block in layout["row_blocks"]:
        # Optional per-block "cell_lines" (one per figure column) for figures
        # whose cell line changes with the row block, e.g. a 2 x 2 grid of
        # cell lines each with its own target + loading-control tiles.
        override = block.get("cell_lines")
        if override is not None and len(override) != sum(cols_per_block):
            raise SystemExit(f"Row block {block['name']!r}: 'cell_lines' needs "
                             f"{sum(cols_per_block)} entries (one per column).")
        for t in block["targets"]:
            row_info[r] = (block["name"], t, override)
            r += 1
    c = 0
    for block in layout["col_blocks"]:
        for cl in block["cell_lines"]:
            col_info[c] = (block["condition"], cl)
            c += 1
    return row_info, col_info


def run_once(gray, panels, layout, row_info, col_info, run):
    rng = np.random.default_rng(1000 + run)
    rows = []
    for p in panels:
        variant, target, override = row_info[p.row]
        condition, cell_line = col_info[p.col]
        if override is not None:
            cell_line = override[p.col]
        # Two lanes: Control + treated.  More lanes (e.g. a reference line and
        # several cell lines in one tile): every lane is listed in the layout.
        lanes = [lane.format(variant=variant, condition=condition) for lane in layout["lanes"]]
        redraw = Redraw.random(rng, n_lanes=len(lanes), max_px=REDRAW_MAX_PX)
        for m in quantify_panel(gray, p, target, lanes, settings=DETECTION, redraw=redraw):
            row = m.as_row()
            row.update(run=run, variant=variant, condition=condition,
                       cell_line=cell_line,
                       lane_role="Control" if m.lane_index == 0 else "Treated",
                       lane_label=m.lane)
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise_vs_reference(raw: pd.DataFrame) -> pd.DataFrame:
    """Multi-lane tiles: one Control-vs-treated comparison per non-reference lane.

    The first lane is the reference (Control).  For every other lane the
    reference rows are repeated with ``cell_line`` = that lane's label, so the
    normalisation, t-tests, QC and plotting steps see ordinary two-lane
    comparisons ("<lane> vs <reference>").  Two-lane data is returned as is.
    """
    if raw["lane_index"].max() <= 1:
        return raw
    ref = raw[raw.lane_index == 0]
    parts = []
    for label in raw.loc[raw.lane_index > 0, "lane_label"].unique():
        trt = raw[raw.lane_label == label].copy()
        ctl = ref.copy()
        for part, role in ((ctl, "Control"), (trt, "Treated")):
            part["cell_line"] = label
            part["lane_role"] = role
            part["comparison"] = f"{label} vs {ref.lane_label.iloc[0]}"
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def normalise(raw: pd.DataFrame, control: str, value: str) -> pd.DataFrame:
    keys = ["run", "variant", "condition", "cell_line", "lane_role"]
    ctrl = raw[raw.target == control].set_index(keys)[value].rename("control_value")
    tgt = raw[raw.target != control].join(ctrl, on=keys)
    tgt["ratio_to_control"] = tgt[value] / tgt["control_value"].replace(0, np.nan)
    ref = tgt[tgt.lane_role == "Control"].set_index(
        ["run", "variant", "condition", "cell_line", "target"])["ratio_to_control"].rename("control_ratio")
    tgt = tgt.join(ref, on=["run", "variant", "condition", "cell_line", "target"])
    tgt["fold_change"] = tgt["ratio_to_control"] / tgt["control_ratio"]
    return tgt


def qc_flags(raw: pd.DataFrame, control: str) -> pd.DataFrame:
    """One QC verdict per (variant, condition, cell line, target) fold change.

    Looks at all four bands that enter the fold change (target control lane,
    target treated lane and both loading-control lanes) across all runs.
    """
    keys = ["variant", "condition", "cell_line"]
    rows = []
    for (v, c, cl, t), g in raw[raw.target != control].groupby(keys + ["target"]):
        h3 = raw[(raw.variant == v) & (raw.condition == c) & (raw.cell_line == cl)
                 & (raw.target == control)]
        issues = []
        ctrl = g[g.lane_role == "Control"]
        trt = g[g.lane_role == "Treated"]
        if (ctrl.source != "detected").any():
            issues.append("control-lane band faint/absent: fold change unstable (denominator ~0)")
        elif ctrl.peak_area.mean() < 0.15 * trt.peak_area.mean():
            issues.append("control-lane band <15% of treated: fold change unstable")
        if (trt.source != "detected").any():
            issues.append("treated band faint (value near background)")
        sat_t = g.saturated_fraction.max()
        sat_h = h3.saturated_fraction.max()
        if sat_t > 0.05:
            issues.append(f"target band saturated ({sat_t:.0%} near-black pixels): difference underestimated")
        if sat_h > 0.05:
            issues.append(f"{control} band saturated ({sat_h:.0%} near-black pixels)")
        both = pd.concat([g, h3])
        if both.notes.fillna("").str.contains("box covers").any():
            issues.append("a box covers <95% of its band (bands too close to widen it)")
        if g.notes.fillna("").str.contains("stacked bands").any():
            issues.append("vertical doublet in a box: both stacked bands measured together")
        if (both.roi_h > both.roi_w).any():
            issues.append("irregular tile (box taller than wide: smear/artifact)")
        cv = g.groupby("lane_role").peak_area.agg(lambda s: s.std() / s.mean() if s.mean() else 0).max()
        if cv > 0.15:
            issues.append(f"ROI-placement sensitive (lane CV {cv:.0%})")
        # Saturation shrinks the measured difference but keeps its direction,
        # so it is CHECK.  Only a missing/faint control band (fold change
        # divides by ~0) makes the number itself unusable.
        severe = (ctrl.source != "detected").any()
        verdict = "UNRELIABLE" if severe else ("CHECK" if issues else "OK")
        rows.append({"variant": v, "condition": c, "cell_line": cl, "target": t,
                     "QC": verdict, "QC notes": "; ".join(issues)})
    return pd.DataFrame(rows).set_index(keys + ["target"])


def overlay(gray, raw_run1, panels, path: Path):
    scale = 2
    img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).convert("RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    for p in panels:
        d.rectangle([p.x * scale, p.y * scale, (p.x + p.w) * scale, (p.y + p.h) * scale],
                    outline=(0, 120, 255), width=1)
    for _, r in raw_run1.iterrows():
        colour = {"detected": (0, 170, 0), "faint": (200, 0, 200)}.get(r.source, (255, 140, 0))
        if r.saturated_fraction > 0.02:
            colour = (220, 0, 0)
        x0, y0 = r.roi_x * scale, r.roi_y * scale
        d.rectangle([x0, y0, x0 + r.roi_w * scale, y0 + r.roi_h * scale], outline=colour, width=2)
    img.save(path)
    return img


SOURCE_COLOURS = {"detected": (0, 160, 0), "faint": (200, 0, 200), "absent": (255, 140, 0)}


def tile_sheet(gray, raw_run1, panels, row_info, col_info, path: Path, scale: int = 4):
    """Every tile enlarged with its two boxes, band extents and the split line.

    Solid box = measured ROI (shared size), thin grey bracket = detected band
    extent, blue dashed line = split between the two bands.
    """
    font = ImageFont.load_default()
    tw = max(p.w for p in panels) * scale
    th = max(p.h for p in panels) * scale
    label_h = 14
    n_rows = max(p.row for p in panels) + 1
    n_cols = max(p.col for p in panels) + 1
    sheet = Image.new("RGB", (n_cols * (tw + 8) + 8, n_rows * (th + label_h + 8) + 8), "white")
    for p in panels:
        crop = Image.fromarray(np.clip(gray[p.y:p.y + p.h, p.x:p.x + p.w], 0, 255).astype(np.uint8))
        tile = crop.convert("RGB").resize((p.w * scale, p.h * scale), Image.NEAREST)
        d = ImageDraw.Draw(tile)
        lanes = raw_run1[(raw_run1.row == p.row) & (raw_run1.col == p.col)]
        for _, r in lanes.iterrows():
            sx = (r.split_x - p.x) * scale
            for yy in range(0, p.h * scale, 6):
                d.line([sx, yy, sx, yy + 3], fill=(0, 90, 255), width=1)
            if r.source != "absent":
                bx0, bx1 = (r.band_x0 - p.x) * scale, (r.band_x1 - p.x) * scale
                by0, by1 = (r.band_y0 - p.y) * scale, (r.band_y1 - p.y) * scale
                d.line([bx0, by1 + 2, bx1, by1 + 2], fill=(120, 120, 120), width=1)
                d.line([bx0, by0, bx0, by1 + 2], fill=(120, 120, 120), width=1)
                d.line([bx1, by0, bx1, by1 + 2], fill=(120, 120, 120), width=1)
            x0, y0 = (r.roi_x - p.x) * scale, (r.roi_y - p.y) * scale
            d.rectangle([x0, y0, x0 + r.roi_w * scale - 1, y0 + r.roi_h * scale - 1],
                        outline=SOURCE_COLOURS.get(r.source, (255, 0, 0)), width=2)
        variant, target, override = row_info[p.row]
        condition, cell_line = col_info[p.col]
        if override is not None:
            cell_line = override[p.col]
        ox = 8 + p.col * (tw + 8)
        oy = 8 + p.row * (th + label_h + 8)
        ImageDraw.Draw(sheet).text((ox, oy), f"{variant}-{condition} {cell_line} {target}",
                                   fill=(0, 0, 0), font=font)
        sheet.paste(tile, (ox, oy + label_h))
    sheet.save(path)
    return sheet


def autosize(ws):
    for column in ws.columns:
        width = max(len(str(c.value)) if c.value is not None else 0 for c in column)
        ws.column_dimensions[column[0].column_letter].width = min(max(10, width + 2), 60)


def write_excel(path: Path, sheets: dict[str, pd.DataFrame], overlay_png: Path, notes: list[str],
                tiles_png: Path | None = None):
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font, PatternFill
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"Notes": notes}).to_excel(xw, sheet_name="README", index=False)
        for name, frame in sheets.items():
            frame.to_excel(xw, sheet_name=name, index=False)
        wb = xw.book
        for ws in wb.worksheets:
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="DDEBF7")
            ws.freeze_panes = "A2"
            autosize(ws)
        wb["README"].column_dimensions["A"].width = 140
        ws = wb.create_sheet("QC overlay")
        ws["A1"] = ("Green = detected band box, magenta = faint band, orange = no band (mirror "
                    "position), red = >2% saturated pixels, blue = tile frame. Run 1 boxes shown.")
        pic = XLImage(str(overlay_png))
        pic.width, pic.height = pic.width // 2, pic.height // 2
        ws.add_image(pic, "A3")
        if tiles_png is not None:
            ws = wb.create_sheet("QC tiles (zoomed)")
            ws["A1"] = ("Every tile at 4x. Solid box = measured ROI (same size for both bands), grey "
                        "bracket = detected band extent, blue dashed line = split between bands. "
                        "Green detected, magenta faint, orange absent. Run 1.")
            pic = XLImage(str(tiles_png))
            pic.width, pic.height = int(pic.width * 0.6), int(pic.height * 0.6)
            ws.add_image(pic, "A3")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args(argv)
    layout = json.loads(args.layout.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    gray = load_gray(args.image)
    panels = assign_grid(find_framed_panels(gray))
    row_info, col_info = label_panels(panels, layout)
    print(f"{len(panels)} panels -> {len(row_info)} rows x {len(col_info)} columns")

    raw = pd.concat([run_once(gray, panels, layout, row_info, col_info, r)
                     for r in range(1, args.runs + 1)], ignore_index=True)
    raw = pairwise_vs_reference(raw)
    control = layout["control_target"]
    norm = normalise(raw, control, "peak_area")
    norm_rect = normalise(raw, control, "net_intden")

    keys = ["variant", "condition", "cell_line", "target", "lane_role", "lane_label"]
    wide = norm.pivot_table(index=keys, columns="run", values="fold_change").reset_index()
    wide.columns = [c if not isinstance(c, (int, np.integer)) else f"Run {c}" for c in wide.columns]
    run_cols = [f"Run {r}" for r in range(1, args.runs + 1)]
    wide["Mean"] = wide[run_cols].mean(axis=1)
    wide["SD"] = wide[run_cols].std(axis=1, ddof=1)
    wide["CV %"] = 100 * wide["SD"] / wide["Mean"]
    summary = wide[wide.lane_role == "Treated"].drop(columns=["lane_role"]).copy()
    summary = summary.join(qc_flags(raw, control), on=["variant", "condition", "cell_line", "target"])
    target_order = {t: i for i, t in enumerate(layout["row_blocks"][0]["targets"])}
    lines_seen: list[str] = []
    for block in layout["row_blocks"]:
        for c in block.get("cell_lines") or []:
            if c not in lines_seen:
                lines_seen.append(c)
    for block in layout["col_blocks"]:
        for c in block["cell_lines"]:
            if c not in lines_seen and not any(b.get("cell_lines") for b in layout["row_blocks"]):
                lines_seen.append(c)
    if len(layout["lanes"]) > 2:  # multi-lane: comparisons are named by lane
        lines_seen = list(layout["lanes"][1:])
    line_order = {c: i for i, c in enumerate(lines_seen)}
    summary = summary.sort_values(
        ["variant", "condition", "cell_line", "target"],
        key=lambda s: s.map(target_order) if s.name == "target"
        else s.map(line_order) if s.name == "cell_line"
        else s.map({"OE": 0, "KD": 1}) if s.name == "condition" else s).round(3)

    # Compact table (cell lines as columns) ready for GraphPad/plotting.
    table = summary.pivot_table(index=["variant", "condition", "target"],
                                columns="cell_line", values=["Mean", "SD"])
    table = table.reindex(columns=[(stat, c) for c in line_order for stat in ("Mean", "SD")])
    table.columns = [f"{c} {stat}" for stat, c in table.columns]
    table = table.reset_index()
    table = table.sort_values(
        ["variant", "condition", "target"],
        key=lambda s: s.map(target_order) if s.name == "target"
        else s.map({"OE": 0, "KD": 1}) if s.name == "condition" else s).round(3)

    # Agreement check between the two measurement methods.
    both = norm[["run", "variant", "condition", "cell_line", "target", "lane_role", "fold_change"]].merge(
        norm_rect[["run", "variant", "condition", "cell_line", "target", "lane_role", "fold_change"]],
        on=["run", "variant", "condition", "cell_line", "target", "lane_role"],
        suffixes=("_peak_area", "_rect_local_bg"))
    both = both[both.lane_role == "Treated"].drop(columns="lane_role").round(3)

    raw_cols = ["run", "variant", "condition", "cell_line", "target", "lane_label", "lane_role",
                "peak_area", "net_intden", "background_per_px", "roi_x", "roi_y", "roi_w",
                "roi_h", "band_x0", "band_x1", "band_y0", "band_y1", "split_x",
                "baseline_y0", "baseline_y1", "saturated_fraction", "source",
                "status", "notes"]
    per_run = norm[["run", "variant", "condition", "cell_line", "target", "lane_label",
                    "lane_role", "peak_area", "control_value", "ratio_to_control",
                    "fold_change"]].round(4)

    overlay_png = args.out / "qc_overlay_run1.png"
    overlay(gray, raw[raw.run == 1], panels, overlay_png)
    tiles_png = args.out / "qc_tiles_run1.png"
    tile_sheet(gray, raw[raw.run == 1], panels, row_info, col_info, tiles_png)
    notes = [
        f"Source image: {args.image}",
        f"{len(panels)} framed panels detected; {len(raw) // args.runs} lane ROIs per run; {args.runs} runs.",
        "Primary value: ImageJ Gel-Analyzer-style peak area (lane profile, straight baseline between "
        "the valleys either side of the band, area above baseline x ROI width).",
        f"Normalisation: target / {control} of the same cell line, same variant block, same lane "
        "(Control or treated). Fold change: treated ratio / control ratio (Control = 1).",
        "Box model: each tile holds two bands. The split between them is the column valley nearest "
        "the mirror-symmetry midpoint (works with a small gap or no gap). Each band's extent is "
        "measured above a local straight baseline; dumbbell bands (dark ends, pale middle) are ONE "
        "band; a band with one dark end and a faded tail is followed along the tail until it meets "
        "background level (two-level edges). Blotchy background is removed with a vertical rolling "
        "window several band-thicknesses tall, and tails stop at the background's own structure, "
        "measured in band-free rows of the same tile. Both bands get ONE box size, set by the longer and thicker band; the other box may "
        "include empty background but never reaches into the neighbouring band. An absent band's "
        "box sits at the mirror position of the present band.",
        "Repeat measurements: band geometry (split, band extents, shared box size) is detected once "
        "with fixed settings. In each run every box moves by a seeded random -1/0/+1 px in x and y, "
        "and the tile's shared box size changes by "
        "-1/0/+1 px; the ImageJ baseline window stays anchored to the detected band. Run-to-run "
        "SD/SEM is measurement (box-placement) precision, NOT biological replication.",
        "Cross-check: sheet 'Method check' compares the peak-area fold change with BlotGrid's "
        "rectangle integrated density (background = median of the strips just above and below the "
        "box, so the neighbouring band never enters the background).",
        "CAUTION: this image is an 8-bit figure export with per-panel cropping/contrast. Near-black "
        "pixels (<=8) are treated as saturated; values from saturated bands underestimate the true difference. "
        "Re-quantify from the raw imager files before publication if they are available.",
        "Faint bands (typically strong knockdowns) keep a box centred on their own faint band, or at "
        "the mirror position if there is no band; their values are near background by design.",
        "QC column: OK = no issues; CHECK = saturation, doublet, faint treated band or ROI-placement "
        "sensitivity (see QC notes); UNRELIABLE = control-lane band faint/absent, so the fold change "
        "divides by ~0 and its size is meaningless (direction only). Inspect the 'QC overlay' sheet.",
        "Saturation (black pixels in the figure) compresses differences: saturated fold changes are "
        "conservative (true OE increase likely larger, true KD decrease likely deeper).",
    ]
    xlsx = args.out / "western_blot_quantification.xlsx"
    write_excel(xlsx, {
        "Fold change table": table,
        "Summary (fold change)": summary,
        "Per-run normalised": per_run,
        "Raw measurements": raw[raw_cols],
        "Method check": both,
    }, overlay_png, notes, tiles_png)
    print(f"wrote {xlsx}")
    return raw, norm, summary


if __name__ == "__main__":
    main()

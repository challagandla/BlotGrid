# BlotGrid

Local, privacy-first Western blot band detection and densitometry with raw-scan
recognition, visual scan straightening, editable ROIs, and optional control
normalization.

A local Streamlit app to quantify Western blot bands from an image, optionally
normalize each target against a housekeeping/loading control **in the same lane**,
aggregate replicates, and export publication-quality bar charts.

Public Western images from HPA v25.1, Mendeley Data, two BioImage Archive
studies, and Zenodo can be prepared as fail-closed review projects with
`scripts/prepare_review_sources.py`. The downloads are not labels: follow the
generated `data/public_western/MANUAL_LABELING.md`, and do not train until every
accepted box and whole image has been reviewed and attested. Source and license
scope is documented in [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

[Beginner tutorial](docs/BEGINNER_TUTORIAL.md) ·
[Pretrained model workflow](docs/PRETRAINED_MODELS.md) ·
[Holdout evaluation schema](docs/HOLDOUT_EVALUATION.md) ·
[Third-party models](THIRD_PARTY_MODELS.md) ·
[Third-party data](THIRD_PARTY_DATA.md) ·
[MIT license](LICENSE)

![BlotGrid workflow](docs/blotgrid-workflow.svg)

## Ubuntu install & run

```bash
sudo apt update
sudo apt install -y python3 python3-venv
./scripts/setup_ubuntu.sh
./run_blotgrid.sh
```

The launcher uses an isolated `.venv` and keeps caches inside the project. The
app opens at `http://localhost:8501`; no command line is needed after launch.
To create the optional demo image first, run `.venv/bin/python
make_sample_blot.py`.

The original code is Python/Streamlit and does not depend on Windows APIs.
Ubuntu-specific setup, a local OCR-model fallback, and version-compatible ONNX
Runtime dependencies are included here.

### Optional pretrained detection

BlotGrid can run the verified Apache-2.0 GelGenie Universal ONNX model and
optionally use SAM 2.1 to recenter one selected ROI:

```bash
./scripts/setup_ubuntu.sh --with-ml
.venv/bin/python scripts/download_pretrained_models.py \
  --model gelgenie-universal

# Optional SAM-assisted ROI refinement:
./scripts/setup_ubuntu.sh --with-sam2 --ml-runtime cu118
.venv/bin/python scripts/download_pretrained_models.py \
  --model sam2.1-hiera-small
```

The installer chooses a tested CUDA 11.8 build when `nvidia-smi` is available
and otherwise installs the CPU build. See the
[pretrained model workflow](docs/PRETRAINED_MODELS.md) for GelGenie/SAM setup
and application use.

AI changes only the proposed ROI locations. BlotGrid still measures
background-corrected intensity from the original native-depth pixels with its
auditable classical densitometry.

### Sealed detector comparison

The optional evaluator compares the classical detector, verified GelGenie
model, and one local format-2 detector checkpoint carrying publication metadata
schema 2 on the same untouched holdout images. Its generated sidecar, embedded
metadata, model state, and container are verified before use. The band cohort
must be reviewed and approved by a person in BlotGrid, with
human-approved artifact regions added to the holdout COCO record before it is
sealed; the evaluator will not create or infer ground truth.

Prepare randomized, method-blinded review tasks:

```bash
.venv/bin/python scripts/evaluate_detector.py prepare \
  --annotations data/holdout/annotations.json \
  --images data/holdout/images \
  --seal data/holdout/seal.json \
  --development-ledger data/development_ledger.json \
  --plan data/holdout/evaluation_plan.json \
  --checkpoint models/detector.pt \
  --gelgenie-model models/gelgenie_universal.onnx \
  --output runs/holdout-001
```

Give the reviewer `runs/holdout-001/reviewer_packet/` plus read-only access to
the exact sealed `--images` directory through a separate channel. Do not give
them the annotations, seal, or `evaluator_private/`. The packet contains the
randomized task order and blinded project JSON files, but neither method
identities nor ground truth. For each task, the reviewer restores its project
in BlotGrid, re-uploads the sealed images (project JSON never embeds pixels),
inspects the whole image, corrects and approves every active ROI, and saves the
reviewed project under the task's original `<task-id>.json` filename in a
separate directory. Evaluator ROI dimensions are frozen: reviewers may move,
add, or disable fixed-size boxes, but the resize handles and size controls are
locked. Then score the three reviewed projects:

```bash
.venv/bin/python scripts/evaluate_detector.py score \
  --run runs/holdout-001 \
  --reviewed-projects data/holdout/reviewed-projects \
  --review-attestation data/holdout/review-attestation.json
```

The seal, complete development/model-selection ledger, checkpoint ancestry,
review bindings, and every frozen input are hash-checked again at scoring.
Outputs report clear/faint recall, border/text/smear/blank false positives,
one-to-one IoU and centre errors, doublet recovery, equal-review edits,
native-pixel intensity/fold-change errors, and scanner/exposure/target/
background/acquisition-style strata. Runs stay under ignored `runs/` or
`data/` directories and contain only JSON/CSV artifacts.

## Workflow (tabs, left to right)

1. **Upload** — drop in one or more blot images (TIFF/PNG/JPG). Open
   **Straighten scan with alignment grid** to estimate skew automatically, tune
   the correction angle against a cyan square grid, and apply the corrected
   working image before detection. The guide grid is never written into the
   image. Local OCR reads
   printed target labels in top-to-bottom order and printed lane/condition labels
   in left-to-right order, then fills **Experiment setup** automatically. When a
   figure has target names but no printed lane names, the app infers the number of
   band columns and uses neutral names such as `Lane 1`, `Lane 2`; it never invents
   experimental conditions. A raw scan with no readable labels is inferred directly
   from its band grid and receives editable `Target 1`, `Target 2`, ... and `Lane 1`,
   `Lane 2`, ... names. Existing recognized names are retained when only part of the
   metadata is missing. Low-confidence raw grids stay as reviewable suggestions
   instead of replacing the setup silently. Printed `Ladder`, `Marker`, and `MW`
   lane labels are selected in **Ladder / marker lanes** automatically; the physical
   lane remains in left-to-right geometry but is excluded from analysis. Each image
   is one replicate. Add several to get real SEM/SD.
2. **ROIs** — the app automatically infers dark/light polarity, locates the lane
   and target-row centres, and draws an individual box around every detected band.
   Multiple bands in one target/lane (for example, a doublet) remain separate.
   Long panel borders and scanner streaks are masked before peak detection, and
   every candidate is checked for a locally connected, horizontal band component.
   Broad merged peaks are marked as unresolved rather than being split without
   evidence. A relaxed, cell-local recovery pass keeps probable faint bands
   active in orange for review; only cells with no usable signal become dashed
   grey fallbacks that are excluded until you review and enable them.
   Every box on a blot uses the same rectangle size for consistent measurement.
   The blot itself is an ImageJ-style editor: click a rectangle, drag inside it to
   move, or drag any of its eight handles to resize. Resizing is blot-locked, so it
   updates every rectangle and cannot selectively change one band's
   measurement area. Green boxes are accepted, red boxes are accepted loading
   controls, orange boxes need review, and dashed grey boxes are excluded.
   Designated ladder/marker lanes receive no quantitative ROIs.
   If automatic detection misses bands, **Add missed bands manually** lets you
   choose the target and lane and create one or several extra rectangles at once;
   drag each new rectangle over its band before quantifying.
   Arrow keys nudge the active ROI by one pixel; Alt + arrow keys
   resize all rectangles by one pixel. Numeric Width/Height controls and the
   X/Y table remain available for exact adjustments. An intensity-profile panel
   shows the vertical interval used for measurement. An evenly spaced manual grid
   is available as a fallback. Reviewed boxes can be explicitly approved as
   training annotations; changing pixels, geometry, labels, or ROI inclusion
   clears whole-image approval.
3. **Quantify** — runs densitometry (integrated density with background subtraction);
   flags saturated bands and non-positive signals. Profile-guided integration is
   the default: it retains the locked lane width but integrates only the dominant
   band's vertical peak interval. Classic full-rectangle integration remains
   available in the sidebar. Ladder lanes are omitted before pixel measurement and
   are filtered again before normalization, statistics, plots, and data export.
4. **Normalize** — when a recognized or manually selected control is available,
   `normalized(target, lane) = target(lane) / control(lane)`. If no usable control
   is found, normalization is skipped and the app carries the background-corrected
   net integrated intensity through replicates, plots, and exports instead.
   Optional fold-change vs a reference lane (reference = 1.0) is available after
   control normalization.
5. **Replicates** — pools blots by shared lane/condition labels → mean ± SD/SEM, N.
   `ci95` is the t-based 95% CI half-width, so it stays honest at the N=3 that
   most blots have. Optional t-test / ANOVA.
6. **Plot & Export** — bar chart per target with SEM/SD/CI error bars, data points,
   colorblind-safe palettes, fonts, legends, captions. Figure width/height are set
   in inches, which is what fixes the exported figure's real dimensions. Export
   PNG/TIFF/SVG/PDF at 150–600 DPI plus CSV/XLSX of the underlying data.

## Sizing the panels

Blots differ enormously in pixel size, so nothing is measured at a fixed scale:

- **Upload** — preview width and previews-per-row are adjustable, so a 4000 px
  scan and a 300 px crop both show at a size you choose.
- **ROIs → Display & panel size** — *Fit panel width* scales the blot to the
  panel, while *Zoom %* pins it to a fixed magnification (25–400%) and scrolls,
  which is what you want for pixel-accurate edges. Above 1:1 the image switches
  to unsmoothed pixels so band edges stay crisp instead of blurring. The blot
  panel's height is set by a slider or by dragging the panel's bottom-right
  corner, and the editor reports its true height back, so shrinking the panel
  really does reclaim the space. The blot and the controls can sit side by side
  with an adjustable width split, or stacked; the ROI table's height is separate.
- **Plot & Export** — a width-share slider trades space between the settings
  column and the chart preview.

The status line under the blot always states the image's pixel size and the
current zoom, so on-screen size is never confused with measured size. None of
these are stored in the project file: they are per-screen preferences, and
restoring someone else's project never rearranges your layout.

## The normalization model

A blot is a grid: **columns = lanes/samples**, **rows = probed targets**, with one
row marked as the loading control (GAPDH, β-actin, H3, H4, etc.). Each target band
is divided by the control band in the *same lane*. Example: with lanes WT/KO/Rescue
and rows Insig, Prmt9, GAPDH, `Insig_norm(WT) = Insig(WT)/GAPDH(WT)`, and so on.
The control is optional: unknown target names are never silently treated as a
control, and an experiment without one reports net integrated density in arbitrary
units rather than a misleading normalized value.

## Densitometry notes

- Handles 8-, 16- and 32-bit images; native range is preserved for measurement.
- Automatic polarity handles dark film bands and light chemiluminescent/
  fluorescent bands independently for each blot; an explicit override is available.
- Automatic detection uses background-corrected, per-lane intensity profiles so a
   strong band does not set one global threshold that hides faint bands. Detection
  sensitivity is adjustable, and all detected rectangles remain editable. Long
  panel borders, scanner streaks, and protein labels printed beside a blot are
  excluded automatically. Low-confidence, overlapping, and broad unresolved peaks
  are visibly flagged for review.
- Background methods: local median frame, rolling-ball, or a fixed per-pixel
  value you type in the sidebar (in signal units, i.e. after inversion).
- Reported value is **net integrated density** (background-subtracted).
- A band is flagged **saturated** when its pixels sit at the top of the native
  range, in whichever polarity the blot was acquired. Saturation breaks
  linearity — re-expose rather than quantify. Blank (zero-signal) background is
  not saturation and is not flagged.

## Saving a project

The sidebar's **Save project (.json)** button writes labels, ladder-lane exclusions, ROI layouts,
replicate numbers and all measurement settings. **Load project** + **Restore
project** brings them back; re-upload the images and re-run Quantify. Images are
not embedded in the JSON. Saved SHA-256 identities must match before a restored
layout is applied; a same-named different image is rejected.

## Composite figures (many framed tiles)

Published figures are often a grid of small, separately framed blot
tiles (each tile = one target in one cell line, two lanes: Control and
treated), with headers and labels around them. `wblot/composite.py` quantifies
such a figure directly, without the Streamlit app.

1. Describe the grid in a layout JSON (see
   [`examples/composite_layout_example.json`](examples/composite_layout_example.json)):
   row blocks (variant + targets, top to bottom), column blocks (condition +
   cell lines, left to right), the loading-control target and the lane labels.
   When the cell line changes with the row block (e.g. a 2 x 2 grid of cell
   lines, each with a target tile above its loading-control tile), give the row
   block its own `cell_lines`, one per column (see
   [`examples/composite_layout_per_block_cell_lines.json`](examples/composite_layout_per_block_cell_lines.json)).
   A tile may also hold more than two lanes (e.g. a reference line and several
   cell lines): list every lane in `lanes`; the first is the reference and each
   other lane becomes a "<lane> vs <reference>" comparison (see
   [`examples/composite_layout_multilane_reference.json`](examples/composite_layout_multilane_reference.json)).
   Plot such data with `layout=shared` (one reference bar, stacked brackets).
2. Quantify (three independent box placements, Excel + QC images):

   ```bash
   .venv/bin/python scripts/quantify_composite.py FIGURE.tif \
     --layout examples/composite_layout_example.json --out results/
   ```

3. Export tidy data and draw A4 bar charts with R/ggplot2 (first variant in the
   top half, second in the bottom; black data points, mean ± SEM, GraphPad-style
   significance brackets with Welch t-test stars, Times New Roman,
   t-test table on page 2):

   ```bash
   .venv/bin/python scripts/export_plot_data.py results/western_blot_quantification.xlsx \
     figures/plot_data.csv
   Rscript scripts/plot_composite_bars.R figures/plot_data.csv figures/
   ```

   Options (`key=value` after the two paths) set colours per variant, title,
   loading-control label, y-axis title, figure size and file stem, e.g. a single
   120 x 100 mm panel:

   ```bash
   Rscript scripts/plot_composite_bars.R figures/plot_data.csv figures/ \
     "palette.A=#D4D4D4,#410257" "title=Target overexpression" \
     control=GAPDH width_mm=120 height_mm=100 name=target_bar_chart
   ```

   In RStudio, set `plot_args` to the same strings and source the script; the
   figure is drawn in the Plots pane and kept as `fig` (add `save=false` to
   leave the exported files untouched):

   ```r
   plot_args <- c("figures/plot_data.csv", "figures", "save=false")
   source("scripts/plot_composite_bars.R", encoding = "UTF-8")
   ```

   Legend and panel labels come from the layout's lane labels. Only bars whose
   control band is missing (QC UNRELIABLE) or extreme outliers (> 8x) are cut
   at the axis limit.

   R needs `ggplot2`, `dplyr`, `tidyr`, `patchwork`, `openxlsx`, `gridExtra` and `ragg`
   (Times New Roman must be installed). Without `palette.<variant>` options,
   Control bars are grey and treated bars use colour-blind-safe Okabe-Ito colours.

How each tile is measured:

- **Tiles and grid:** closed dark frames are found and clustered into rows and
  columns; the frame lines are excluded.
- **Background:** a per-column vertical rolling window several band
  thicknesses tall removes shading, streaks and blotchy background while
  keeping thin horizontal bands.
- **Split between the two bands:** the column-profile valley that is a real dip
  (lower than the bands on both sides) and nearest the mirror-symmetry
  midpoint, so touching bands split at their junction, a faint knockdown band
  next to a strong control keeps its own lane, and a dumbbell band's pale
  middle is never chosen.
- **Band extent:** dark core(s) merged into one span (dumbbell = one band),
  then grown along faded tails until the signal reaches the tile's own
  background level (measured in band-free rows); vertical edges are measured
  above a local straight baseline (gradient-safe).
- **One box size per tile:** set by the longer and thicker band; the other
  box may cover empty background but never reaches into the neighbouring
  band, and the two boxes never overlap. A missing band's box sits at the
  mirror position of the present band.
- **Measurement:** ImageJ Gel-Analyzer-style peak area (box profile, straight
  baseline between the valleys either side of the band) plus a rectangle
  integrated density whose background comes from the strips above and below
  the box. Normalisation is target / loading control of the same lane; fold
  change is treated / Control.
- **QC:** saturation (near-black pixels), boxes that could not cover their
  band, vertical doublets, irregular tiles, faint/absent control bands and
  run-to-run spread are reported per fold change as OK / CHECK / UNRELIABLE,
  with a zoomed per-tile QC sheet.

Repeat measurements: band geometry is detected once with fixed settings. In
each run every box moves by a seeded random -1/0/+1 px and the tile's shared
box size changes by -1/0/+1 px; the ImageJ baseline window stays anchored to
the detected band. Detection settings are not varied between runs, because
that changes the measured region rather than repeating the measurement. The
repeats are **technical** replicates of one image: their P values reflect
measurement precision, not biological replication.

## Layout

```
app.py                 Streamlit UI (6 tabs)
wblot/
  image_io.py          load images, signal polarity, saturation
  detection.py         automatic polarity, lanes, target rows and band rectangles
  label_ocr.py         local OCR for target/lane labels and lane-count inference
  densitometry.py      background subtraction, band measurement, lane profile
  normalization.py     per-lane control normalization, fold change
  stats.py             replicate aggregation, t-test / ANOVA
  plotting.py          publication bar chart builder
  project_io.py        save/load project JSON
  ml_detector.py       optional TorchVision ROI proposals and checkpoint provenance
  ml_training.py       reviewed-annotation export, safe splits, training/evaluation
  holdout_evaluation.py sealed holdout validation, matching, and paired errors
  gelgenie.py          verified pretrained ONNX segmentation-to-ROI adapter
  sam_refiner.py       optional fixed-size ROI recentering with SAM 2.1
  composite.py         multi-tile figures: tiles, band split/extent, shared boxes
scripts/
  setup_ubuntu.sh      creates the Ubuntu environment (optional CPU/CUDA ML)
  download_pretrained_models.py
  prepare_public_training_data.py
  smoke_pretrained_models.py
  export_project_annotations.py
  train_detector.py
  check_checkpoint_release.py
  evaluate_detector.py
  quantify_composite.py  quantify a composite figure -> Excel + QC images
  export_plot_data.py    tidy per-run data for plotting
  plot_composite_bars.R  A4 ggplot2 bar charts, Welch t-tests, stars
examples/
  composite_layout_example.json
  composite_layout_per_block_cell_lines.json
  composite_layout_multilane_reference.json
make_sample_blot.py    generates a synthetic 3x3 demo blot
test_core.py           core unit tests
test_ml_support.py     AI/export safety tests (Torch is imported only when needed)
test_app_evaluator_policy.py evaluator-review UI lock tests
test_composite.py      composite tiles: split, dumbbell/comet bands, shared boxes
test_checkpoint_release.py checkpoint/sidecar release-binding tests
test_holdout_evaluation.py sealed comparison and native-pixel metric tests
test_holdout_security.py evaluator privacy and aggregation security tests
```

## Tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q \
  test_core.py test_ml_support.py test_sam_refiner.py \
  test_public_training_data.py test_checkpoint_release.py \
  test_holdout_evaluation.py test_holdout_security.py \
  test_app_evaluator_policy.py test_composite.py
```

Covers OCR label classification and unlabeled-lane inference, automatic dark/light
band detection (including a faint band and a doublet),
known-pixel integrated density, background subtraction (local and fixed), per-lane
normalization, fold-change, divide-by-zero guard, SEM/SD/CI aggregation, saturation
flagging in both polarities, 16-bit range handling, ROI edge clipping, and project
save/load round-tripping.

## License

BlotGrid is released under the [MIT License](LICENSE).

## Caveats

Densitometry accuracy depends on exposure being in the linear range and clean ROI
placement. This tool assists quantification; it does not replace good blot practice
(loading controls, biological replicates, avoiding saturation). Statistical tests
are convenience helpers — confirm the design fits your experiment.

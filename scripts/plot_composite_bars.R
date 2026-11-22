#!/usr/bin/env Rscript
# Publication bar charts for a composite Western blot quantification.
#
# Usage:
#   Rscript scripts/plot_composite_bars.R plot_data.csv OUT_DIR [key=value ...]
#
# Options (all optional):
#   palette.<variant>=<control colour>,<treated colour>   e.g. palette.A=#D4D4D4,#410257
#                           (default: grey Control, colour-blind-safe treated colours)
#   title=<template>        panel-block title; {variant} is replaced (default {variant})
#   control=<name>          loading control shown in labels (default "loading control")
#   methods_note=<text>     extra line for the Methods sheet of ttest_results.xlsx
#   ylab=<text>             y-axis title (default "Relative level (normalised to <control>; Control = 1)")
#   width_mm=, height_mm=   figure size (default A4 portrait 210 x 297)
#   name=<file stem>        output file stem (default bar_charts_A4)
#   save=true|false         write the PDF/PNG/TIFF/Excel outputs (default true)
#   layout=paired|shared    paired: Control + treated bar per cell line (default);
#                           shared: one reference bar then one bar per
#                           line, with stacked brackets from the reference to each
#
# RStudio: set `plot_args` (same strings as on the command line) and source this
# file; the figure is drawn in the Plots pane and kept as `fig` for tweaking:
#   plot_args <- c("figures/plot_data.csv", "figures", "save=false")
#   source("scripts/plot_composite_bars.R", encoding = "UTF-8")
#
# plot_data.csv comes from scripts/export_plot_data.py (one row per run x
# variant x condition x cell line x target x lane, with ratio_to_control and QC).
#
# Output (A4 portrait, first variant in the top half, second in the bottom):
#   bar_charts_A4.pdf   page 1 = figure, page 2 = t-test table
#   bar_charts_A4.png / .tiff (600 dpi)
#   ttest_results.csv / .xlsx
#
# Statistics: two-sided Welch two-sample t-test, Control vs treated, on the
# loading-control-normalised intensities of the quantification runs; bars =
# mean +/- SEM.  Runs are repeated measurements of one blot image (technical
# replicates): they show measurement precision, not biological variation.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(openxlsx)
})

args <- if (exists("plot_args", envir = globalenv())) get("plot_args", envir = globalenv()) else
  commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript plot_composite_bars.R plot_data.csv OUT_DIR [key=value ...]")
in_csv <- args[1]
out_dir <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

opts <- list(title = "{variant}", control = "loading control", ylab = NA_character_,
             width_mm = "210", height_mm = "297", name = "bar_charts_A4", save = "true",
             layout = "paired", methods_note = NA_character_)
extra_palettes <- list()
for (a in args[-(1:2)]) {
  key <- sub("=.*$", "", a)
  val <- sub("^[^=]*=", "", a)
  if (startsWith(key, "palette.")) {
    cols <- strsplit(val, ",", fixed = TRUE)[[1]]
    extra_palettes[[sub("^palette\\.", "", key)]] <- c(Control = cols[1], Treated = cols[2])
  } else {
    opts[[key]] <- val
  }
}
control_name <- opts$control
fig_w <- as.numeric(opts$width_mm)
fig_h <- as.numeric(opts$height_mm)
save_files <- tolower(opts$save) %in% c("true", "t", "yes", "1")

# Times New Roman on screen: register it for the default Windows graphics
# device (RStudio Plots pane); file outputs use ragg/cairo, which find it by name.
if (.Platform$OS.type == "windows" &&
    !"Times New Roman" %in% names(grDevices::windowsFonts())) {
  grDevices::windowsFonts(`Times New Roman` = grDevices::windowsFont("Times New Roman"))
}

# Per-variant colours (Control / treated).  Variants without a palette.<name>
# option get grey Control and the next colour-blind-safe (Okabe-Ito) colour.
variant_palettes <- extra_palettes
default_treated <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
dagger <- "†"

d <- read.csv(in_csv, check.names = FALSE, stringsAsFactors = FALSE)
names(d)[names(d) == "QC notes"] <- "QC_notes"
names(d)[names(d) == "ratio_to_H3"] <- "ratio_to_control"   # older exports
unused <- setdiff(unique(d$variant), names(variant_palettes))
for (k in seq_along(unused)) {
  variant_palettes[[unused[k]]] <- c(Control = "grey80",
                                     Treated = default_treated[(k - 1) %% length(default_treated) + 1])
}
d <- d %>%
  mutate(
    variant   = factor(variant, levels = unique(variant)),
    condition = factor(condition, levels = unique(condition)),
    cell_line = factor(cell_line, levels = unique(cell_line)),
    target    = factor(target, levels = unique(target)),
    lane_role = factor(lane_role, levels = c("Control", "Treated"))
  ) %>%
  group_by(variant, condition, cell_line, target) %>%
  # Express every value relative to the mean Control of its comparison.
  mutate(rel = ratio_to_control / mean(ratio_to_control[lane_role == "Control"])) %>%
  ungroup()

# ---------------------------------------------------------------- t-tests
stars_for <- function(p) {
  case_when(is.na(p) ~ "n/a", p < 0.001 ~ "***", p < 0.01 ~ "**",
            p < 0.05 ~ "*", TRUE ~ "ns")
}

tt <- d %>%
  group_by(variant, condition, cell_line, target) %>%
  group_modify(function(g, key) {
    a <- g$ratio_to_control[g$lane_role == "Control"]
    b <- g$ratio_to_control[g$lane_role == "Treated"]
    r <- tryCatch(t.test(b, a, var.equal = FALSE), error = function(e) NULL)
    rc <- g$rel[g$lane_role == "Control"]
    rt <- g$rel[g$lane_role == "Treated"]
    tibble(
      comparison   = paste0(unique(g$lane_label[g$lane_role == "Treated"]), " vs Control"),
      n_control    = length(a),
      n_treated    = length(b),
      mean_control = mean(rc), sd_control = sd(rc),
      mean_treated = mean(rt), sd_treated = sd(rt),
      fold_change  = mean(rt) / mean(rc),
      t            = if (is.null(r)) NA_real_ else unname(r$statistic),
      df           = if (is.null(r)) NA_real_ else unname(r$parameter),
      p_value      = if (is.null(r)) NA_real_ else r$p.value,
      QC           = g$QC[1],
      QC_notes     = g$QC_notes[1]
    )
  }) %>%
  ungroup() %>%
  mutate(significance = stars_for(p_value))

table_out <- tt %>%
  transmute(Variant = variant, Condition = condition, `Cell line` = cell_line,
            Target = target, Comparison = comparison,
            `n (Control)` = n_control, `n (treated)` = n_treated,
            `Control mean` = round(mean_control, 3),
            `Control SEM` = round(sd_control / sqrt(n_control), 3),
            `Treated mean` = round(mean_treated, 3),
            `Treated SEM` = round(sd_treated / sqrt(n_treated), 3),
            `Fold change` = round(fold_change, 3),
            t = round(t, 3), df = round(df, 2),
            `P value` = signif(p_value, 3), Significance = significance,
            QC = QC, `QC notes` = QC_notes)
if (save_files) {
write.csv(table_out, file.path(out_dir, "ttest_results.csv"), row.names = FALSE)

wb <- createWorkbook()
addWorksheet(wb, "t-tests")
writeData(wb, "t-tests", table_out, headerStyle = createStyle(textDecoration = "bold", fgFill = "#DDEBF7"))
freezePane(wb, "t-tests", firstRow = TRUE)
setColWidths(wb, "t-tests", cols = seq_len(ncol(table_out)), widths = "auto")
addWorksheet(wb, "Methods")
writeData(wb, "Methods", data.frame(Notes = c(
  paste0("Two-sided Welch two-sample t-test (unequal variances), treated vs Control, on ", control_name, "-normalised band intensities."),
  "Values shown are relative to the mean Control of each comparison (Control = 1).",
  "n = re-measurements of ONE blot image (bands detected once; each box redrawn by a random -1/0/+1 px in position and shared size): technical replicates. P values describe measurement precision, not biological reproducibility; use independent blots (biological replicates) for inference.",
  "Significance: * P < 0.05, ** P < 0.01, *** P < 0.001, ns = not significant.",
  if (!is.na(opts$methods_note)) opts$methods_note else NULL,
  paste0(dagger, " = QC UNRELIABLE: control-lane band absent, so the fold change is not interpretable (direction only).")
)))
setColWidths(wb, "Methods", cols = 1, widths = 150)
saveWorkbook(wb, file.path(out_dir, "ttest_results.xlsx"), overwrite = TRUE)
}

# ---------------------------------------------------------------- plot data
sem <- function(x) sd(x) / sqrt(length(x))
summ <- d %>%
  group_by(variant, condition, cell_line, target, lane_role) %>%
  summarise(mean = mean(rel), sem = sem(rel), n = n(), .groups = "drop")

# Highest drawn element of each bar: bar + SEM, or its highest data point.
lane_top <- d %>%
  group_by(variant, condition, cell_line, target, lane_role) %>%
  summarise(point_max = max(rel), .groups = "drop") %>%
  left_join(summ, by = c("variant", "condition", "cell_line", "target", "lane_role")) %>%
  mutate(top = pmax(point_max, mean + sem)) %>%
  select(variant, condition, cell_line, target, lane_role, top)

group_top <- lane_top %>%
  group_by(variant, condition, cell_line, target) %>%
  summarise(top = max(top), .groups = "drop")

# A bar whose fold change divides by a ~0 control (QC UNRELIABLE) or that is
# an extreme outlier (> 8x the next bar) would flatten every other bar in its
# panel.  Only such bars are cut at an axis cap and labelled with their true
# mean; a genuine large effect (e.g. 5x) is drawn to scale.
caps <- group_top %>%
  left_join(tt %>% select(variant, condition, cell_line, target, QC),
            by = c("variant", "condition", "cell_line", "target")) %>%
  group_by(variant, condition, target) %>%
  mutate(second = if (n() > 1) sort(top, decreasing = TRUE)[2] else top,
         capped = (top > 3 * second & QC == "UNRELIABLE") | top > 8 * second,
         cap = ifelse(capped, 1.6 * second, NA_real_)) %>%
  ungroup() %>%
  select(-QC)
panel_cap <- caps %>% group_by(variant, condition, target) %>%
  summarise(cap = suppressWarnings(max(cap, na.rm = TRUE)), .groups = "drop") %>%
  mutate(cap = ifelse(is.finite(cap), cap, NA_real_))

summ <- summ %>% left_join(panel_cap, by = c("variant", "condition", "target")) %>%
  mutate(over = !is.na(cap) & mean > cap,
         mean_plot = ifelse(over, cap, mean),
         ymin = ifelse(over, NA, mean - sem),
         ymax = ifelse(over, NA, mean + sem))
# Dots are placed explicitly on their own bar (runs spread left / centre /
# right).  A dodge position would re-centre the remaining dots whenever the
# other bar's dots are hidden above an axis cap.
dot_dodge <- 0.78 / 4
pts <- d %>% left_join(panel_cap, by = c("variant", "condition", "target")) %>%
  filter(is.na(cap) | rel <= cap) %>%
  group_by(variant, condition, cell_line, target, lane_role) %>%
  mutate(spread = (rank(run, ties.method = "first") - (n() + 1) / 2) * 0.055) %>%
  ungroup() %>%
  mutate(x_dot = as.numeric(cell_line) +
           ifelse(lane_role == "Control", -dot_dodge, dot_dodge) + spread)

# GraphPad-style significance brackets: a horizontal line above the pair with
# a leg down to each bar, and the stars (or "ns") on top of the line.
dodge_w <- 0.78
half_pair <- dodge_w / 4          # bar centres sit at x -/+ dodge_w / 4
brackets <- lane_top %>%
  left_join(panel_cap, by = c("variant", "condition", "target")) %>%
  mutate(shown = ifelse(!is.na(cap) & top > cap, cap, top)) %>%
  group_by(variant, condition, target) %>%
  mutate(panel_max = max(shown)) %>%
  ungroup() %>%
  select(variant, condition, cell_line, target, lane_role, shown, panel_max) %>%
  pivot_wider(names_from = lane_role, values_from = shown) %>%
  left_join(tt %>% select(variant, condition, cell_line, target, significance, QC, fold_change),
            by = c("variant", "condition", "cell_line", "target")) %>%
  left_join(caps %>% select(variant, condition, cell_line, target, capped, cap),
            by = c("variant", "condition", "cell_line", "target")) %>%
  mutate(x = as.numeric(cell_line),
         x0 = x - half_pair, x1 = x + half_pair,
         y_line = pmax(Control, Treated) + 0.10 * panel_max,
         y_leg0 = Control + 0.035 * panel_max,
         y_leg1 = Treated + 0.035 * panel_max,
         label = paste0(significance, ifelse(QC == "UNRELIABLE", dagger, "")),
         is_ns = significance %in% c("ns", "n/a"),
         y_label = y_line + ifelse(is_ns, 0.015, 0.0) * panel_max,
         y_blank = y_line + 0.16 * panel_max,
         note = ifelse(capped, paste0("↑", formatC(fold_change, format = "f", digits = 1), "×"),
                       NA_character_))

# Panel and legend labels come from the data's own lane labels.
treated_lab <- d %>% filter(lane_role == "Treated") %>% distinct(variant, condition, lane_label)
control_lab <- d %>% filter(lane_role == "Control") %>% distinct(variant, lane_label)
strip_lab <- function(v, cond, tgt) {
  lab <- treated_lab$lane_label[match(paste(v, cond), paste(treated_lab$variant, treated_lab$condition))]
  paste0(lab, "  |  ", tgt)
}
font <- "Times New Roman"

plot_variant <- function(v) {
  pal <- variant_palettes[[as.character(v)]]
  if (is.null(pal)) pal <- c(Control = "grey80", Treated = "grey30")
  ctrl_name <- paste(unique(control_lab$lane_label[control_lab$variant == v]), collapse = " / ")
  trt_name <- paste(unique(treated_lab$lane_label[treated_lab$variant == v]), collapse = " / ")
  ss <- summ %>% filter(variant == v) %>% mutate(panel = strip_lab(v, condition, target))
  pp <- pts %>% filter(variant == v) %>% mutate(panel = strip_lab(v, condition, target))
  bb <- brackets %>% filter(variant == v) %>% mutate(panel = strip_lab(v, condition, target))
  levels_panel <- unique(strip_lab(v, rep(levels(d$condition), each = nlevels(d$target)),
                                   rep(levels(d$target), times = nlevels(d$condition))))
  ss$panel <- factor(ss$panel, levels = levels_panel)
  pp$panel <- factor(pp$panel, levels = levels_panel)
  bb$panel <- factor(bb$panel, levels = levels_panel)
  dodge <- position_dodge(width = dodge_w)
  stars <- bb %>% filter(!is_ns)
  ns <- bb %>% filter(is_ns)
  notes <- bb %>% filter(!is.na(note))
  single <- nlevels(droplevels(ss$panel)) == 1

  ggplot(ss, aes(x = cell_line, y = mean_plot, fill = lane_role)) +
    geom_col(position = dodge, width = 0.72, colour = "black", linewidth = 0.3) +
    geom_errorbar(aes(ymin = ymin, ymax = ymax), position = dodge, width = 0.3,
                  linewidth = 0.3, na.rm = TRUE) +
    geom_point(data = pp, aes(x = x_dot, y = rel),
               shape = 21, size = 0.85, stroke = 0.3, fill = "black", colour = "black",
               inherit.aes = FALSE) +
    # Bracket: legs from just above each bar up to a common line.
    geom_segment(data = bb, aes(x = x0, xend = x0, y = y_leg0, yend = y_line),
                 inherit.aes = FALSE, linewidth = 0.3) +
    geom_segment(data = bb, aes(x = x1, xend = x1, y = y_leg1, yend = y_line),
                 inherit.aes = FALSE, linewidth = 0.3) +
    geom_segment(data = bb, aes(x = x0, xend = x1, y = y_line, yend = y_line),
                 inherit.aes = FALSE, linewidth = 0.3) +
    geom_text(data = stars, aes(x = x, y = y_label, label = label), inherit.aes = FALSE,
              family = font, size = 3.1, vjust = 0.25) +
    geom_text(data = ns, aes(x = x, y = y_label, label = label), inherit.aes = FALSE,
              family = font, size = 2.2, vjust = 0) +
    geom_text(data = notes, aes(x = x1, y = cap * 0.97, label = note), inherit.aes = FALSE,
              family = font, size = 2.0, angle = 90, hjust = 1, colour = "white",
              fontface = "bold") +
    geom_blank(data = bb, aes(x = x, y = y_blank), inherit.aes = FALSE) +
    facet_wrap(~ panel, nrow = min(2, nlevels(d$condition)), scales = "free_y") +
    scale_fill_manual(values = pal,
                      labels = c(Control = ctrl_name, Treated = trt_name),
                      name = NULL) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.02))) +
    labs(title = gsub("{variant}", v, opts$title, fixed = TRUE), x = NULL,
         y = if (is.na(opts$ylab)) paste0("Relative level (normalised to ", control_name,
                                          "; Control = 1)") else opts$ylab) +
    theme_classic(base_size = 8, base_family = font) +
    theme(
      plot.title = element_text(face = "bold", size = 12, colour = "black"),
      # A single-panel figure needs no strip: the title already names it.
      strip.background = if (single) element_blank() else
        element_rect(fill = scales::alpha(pal[["Treated"]], 0.15), colour = NA),
      strip.text = if (single) element_blank() else element_text(face = "bold", size = 7.5),
      axis.text.x = element_text(angle = 40, hjust = 1, colour = "black", size = 7.5),
      axis.text.y = element_text(colour = "black", size = 7.5),
      axis.line = element_line(linewidth = 0.35),
      axis.ticks = element_line(linewidth = 0.35),
      legend.position = "top", legend.justification = "left",
      legend.key.size = unit(3.2, "mm"),
      legend.text = element_text(size = 8),
      legend.margin = margin(0, 0, 0, 0),
      panel.spacing.y = unit(3, "mm")
    )
}

# Shared-reference layout: one reference bar, then one bar per compared line,
# with GraphPad-style stacked brackets from the reference to each line.
plot_shared <- function(v) {
  pal <- variant_palettes[[as.character(v)]]
  if (is.null(pal)) pal <- c(Control = "grey80", Treated = "grey30")
  panels <- summ %>% filter(variant == v) %>% distinct(condition, target)
  plots <- lapply(seq_len(nrow(panels)), function(i) {
    cnd <- panels$condition[i]
    tgt <- panels$target[i]
    s_v <- summ %>% filter(variant == v, condition == cnd, target == tgt)
    lines <- levels(droplevels(s_v$cell_line))
    ref_cl <- lines[1]                       # reference rows are identical in every comparison
    ref_lab <- unique(d$lane_label[d$variant == v & d$condition == cnd &
                                   d$lane_role == "Control"])[1]
    groups <- c(ref_lab, lines)
    bars <- bind_rows(
      s_v %>% filter(lane_role == "Control", cell_line == ref_cl) %>% mutate(group = ref_lab),
      s_v %>% filter(lane_role == "Treated") %>% mutate(group = as.character(cell_line))
    ) %>% mutate(group = factor(group, levels = groups))
    p_v <- d %>%
      filter(variant == v, condition == cnd, target == tgt,
             (lane_role == "Control" & cell_line == ref_cl) | lane_role == "Treated") %>%
      mutate(group = factor(ifelse(lane_role == "Control", ref_lab, as.character(cell_line)),
                            levels = groups)) %>%
      group_by(group) %>%
      mutate(spread = (rank(run, ties.method = "first") - (n() + 1) / 2) * 0.09) %>%
      ungroup() %>%
      mutate(x = as.numeric(group) + spread)
    tops <- p_v %>% group_by(group) %>% summarise(pt = max(rel), .groups = "drop") %>%
      left_join(bars %>% transmute(group, bt = mean + sem), by = "group") %>%
      mutate(top = pmax(pt, bt), xi = as.numeric(group))
    ymax <- max(tops$top)
    step <- 0.09 * ymax; gap <- 0.06 * ymax; tick <- 0.025 * ymax
    br <- tt %>% filter(variant == v, condition == cnd, target == tgt) %>%
      mutate(group = factor(as.character(cell_line), levels = groups),
             x1 = as.numeric(group)) %>%
      arrange(x1)
    br$y <- NA_real_
    prev <- -Inf
    for (k in seq_len(nrow(br))) {
      span_top <- max(tops$top[tops$xi <= br$x1[k]])
      br$y[k] <- max(span_top + gap, prev + step)
      prev <- br$y[k]
    }
    br <- br %>% mutate(label = paste0(significance, ifelse(QC == "UNRELIABLE", dagger, "")),
                        is_ns = significance %in% c("ns", "n/a"), xm = (1 + x1) / 2)
    fills <- setNames(c(pal[["Control"]], rep(pal[["Treated"]], length(lines))), groups)

    ggplot(bars, aes(x = group, y = mean, fill = group)) +
      geom_col(width = 0.68, colour = "black", linewidth = 0.3) +
      geom_errorbar(aes(ymin = mean - sem, ymax = mean + sem), width = 0.28, linewidth = 0.3) +
      geom_point(data = p_v, aes(x = x, y = rel), inherit.aes = FALSE,
                 shape = 21, size = 0.85, stroke = 0.3, fill = "black", colour = "black") +
      geom_segment(data = br, aes(x = 1, xend = x1, y = y, yend = y),
                   inherit.aes = FALSE, linewidth = 0.3) +
      geom_segment(data = br, aes(x = 1, xend = 1, y = y - tick, yend = y),
                   inherit.aes = FALSE, linewidth = 0.3) +
      geom_segment(data = br, aes(x = x1, xend = x1, y = y - tick, yend = y),
                   inherit.aes = FALSE, linewidth = 0.3) +
      geom_text(data = filter(br, !is_ns), aes(x = xm, y = y, label = label),
                inherit.aes = FALSE, family = font, size = 3.1, vjust = 0.25) +
      geom_text(data = filter(br, is_ns), aes(x = xm, y = y + 0.01 * ymax, label = label),
                inherit.aes = FALSE, family = font, size = 2.2, vjust = 0) +
      scale_fill_manual(values = fills, guide = "none") +
      scale_y_continuous(limits = c(0, max(br$y) + step), expand = expansion(mult = c(0, 0.02))) +
      labs(title = if (i == 1) gsub("{variant}", v, opts$title, fixed = TRUE) else NULL,
           subtitle = if (nrow(panels) > 1) tgt else NULL, x = NULL,
           y = if (is.na(opts$ylab)) paste0("Relative level (normalised to ", control_name,
                                            "; ", ref_lab, " = 1)") else opts$ylab) +
      theme_classic(base_size = 8, base_family = font) +
      theme(
        plot.title = element_text(face = "bold", size = 12, colour = "black"),
        axis.text.x = element_text(angle = 40, hjust = 1, colour = "black", size = 7.5),
        axis.text.y = element_text(colour = "black", size = 7.5),
        axis.line = element_line(linewidth = 0.35),
        axis.ticks = element_line(linewidth = 0.35)
      )
  })
  wrap_plots(plots, nrow = 1)
}

variants <- levels(d$variant)
# Caption wording: name the reference lane when there is a single one.
ctrl_labels <- unique(d$lane_label[d$lane_role == "Control"])
ref_word <- if (length(ctrl_labels) == 1) ctrl_labels else "Control"
cmp_word <- if (identical(opts$layout, "shared")) paste0("each line vs ", ref_word) else
  paste0("treated vs ", ref_word)
caption <- paste0(
  "Bars: mean ± SEM; black dots: three re-measurements of one blot image (bands detected once, each box redrawn ±1 px; technical replicates, n = 3).\n",
  "Values normalised to ", control_name, " and expressed relative to the mean ", ref_word, " (= 1). ",
  "Two-sided Welch t-test, ", cmp_word, ": * P < 0.05, ** P < 0.01, *** P < 0.001, ns not significant.\n",
  "P values reflect measurement precision of the re-measurements, not biological replication.\n",
  if (any(caps$capped)) "\u2191 bar cut at the axis limit; number = true fold change.   " else "",
  if (any(tt$QC == "UNRELIABLE")) paste0(dagger, " control band absent: fold change not interpretable.   ") else "",
  "Many bands are saturated in the source figure, so fold changes are conservative."
)
# Narrow figures: wrap each caption line to the figure width (Times 7 pt is
# about 1.12 mm per character).  A4-wide figures keep the hand-set lines.
if (fig_w < 200) {
  wrap_chars <- max(40, floor((fig_w - 12) / 1.12))
  caption <- paste(unlist(lapply(strsplit(caption, "\n", fixed = TRUE)[[1]],
                                 function(line) strwrap(line, width = wrap_chars))), collapse = "\n")
}
plot_fun <- if (identical(opts$layout, "shared")) plot_shared else plot_variant
fig <- wrap_plots(lapply(variants, plot_fun), ncol = 1) +
  plot_annotation(caption = caption,
                  theme = theme(plot.caption = element_text(size = 7, hjust = 0, lineheight = 1.1,
                                                            family = font)))

a4_w <- 210; a4_h <- 297
stem <- opts$name
if (save_files) {
ggsave(file.path(out_dir, paste0(stem, ".png")), fig, width = fig_w, height = fig_h, units = "mm",
       dpi = 600, bg = "white", device = ragg::agg_png)
ggsave(file.path(out_dir, paste0(stem, ".tiff")), fig, width = fig_w, height = fig_h, units = "mm",
       dpi = 600, bg = "white", device = ragg::agg_tiff, compression = "lzw")

# t-test table page (PDF page 2, also saved as its own A4 PNG).
draw_table <- function() {
  tab <- table_out %>%
    transmute(Variant, Condition, `Cell line`, Target, `Fold change`,
              t, df, `P value` = formatC(`P value`, format = "g", digits = 3),
              Sig. = Significance, QC)
  grid::grid.newpage()
  theme_tab <- gridExtra::ttheme_minimal(
    base_size = 7.4, padding = unit(c(3.2, 1.35), "mm"),
    base_family = font,
    colhead = list(fg_params = list(fontface = "bold", fontfamily = font)),
    core = list(bg_params = list(fill = c("white", "grey95"))))
  g <- gridExtra::tableGrob(tab, rows = NULL, theme = theme_tab)
  title <- grid::textGrob(
    "Welch two-sample t-tests (treated vs Control; n = 3 technical replicates each)",
    gp = grid::gpar(fontsize = 10, fontface = "bold", fontfamily = font))
  note <- grid::textGrob(paste0(
    "* P < 0.05, ** P < 0.01, *** P < 0.001, ns not significant. Fold change = mean treated / mean Control (", control_name, "-normalised)."),
    gp = grid::gpar(fontsize = 7, fontfamily = font))
  gridExtra::grid.arrange(title, g, note, ncol = 1, heights = c(0.04, 0.93, 0.03))
}
has_grid_extra <- requireNamespace("gridExtra", quietly = TRUE)

# The table page is A4: appended to an A4 figure PDF, otherwise its own PDF.
cairo_pdf(file.path(out_dir, paste0(stem, ".pdf")), width = fig_w / 25.4, height = fig_h / 25.4,
          onefile = TRUE, family = font)
print(fig)
same_page_size <- fig_w == a4_w && fig_h == a4_h
if (has_grid_extra && same_page_size) draw_table()
invisible(dev.off())
if (has_grid_extra) {
  if (!same_page_size) {
    cairo_pdf(file.path(out_dir, paste0(stem, "_ttest_table.pdf")), width = a4_w / 25.4,
              height = a4_h / 25.4, family = font)
    draw_table()
    invisible(dev.off())
  }
  ragg::agg_png(file.path(out_dir, if (same_page_size) "ttest_table_A4.png"
                          else paste0(stem, "_ttest_table.png")),
                width = a4_w, height = a4_h, units = "mm", res = 300)
  draw_table()
  invisible(dev.off())
}

cat("wrote", file.path(out_dir, paste0(stem, ".pdf")), "\n")
}
print(table(tt$significance))

# Interactive session (RStudio): draw the figure in the Plots pane.
if (interactive()) print(fig)

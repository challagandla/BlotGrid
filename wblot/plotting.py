"""Publication-quality bar chart / histogram builder (matplotlib)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless / Streamlit
import matplotlib.pyplot as plt


# Colorblind-safe Okabe-Ito palette.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

PALETTES = {
    "Okabe-Ito (colorblind-safe)": OKABE_ITO,
    "Grayscale": ["#222222", "#666666", "#999999", "#bbbbbb", "#dddddd"],
    "Viridis": None,   # resolved from matplotlib colormap at plot time
    "Tab10": None,
}


@dataclass
class PlotOptions:
    error_type: str = "sem"          # 'sem' | 'sd' | 'ci95' | 'none'
    show_points: bool = True
    point_size: float = 28.0
    point_alpha: float = 0.85
    point_color: str = "#333333"
    jitter: float = 0.08
    palette: str = "Okabe-Ito (colorblind-safe)"
    custom_colors: list = field(default_factory=list)  # overrides palette if set
    bar_alpha: float = 0.95
    bar_edge: str = "#000000"
    bar_edge_width: float = 0.8
    errbar_capsize: float = 4.0
    errbar_width: float = 1.2

    title: str = ""
    xlabel: str = ""
    ylabel: str = "Relative expression"
    show_legend: bool = False
    legend_loc: str = "best"
    caption: str = ""

    font_family: str = "DejaVu Sans"
    title_size: float = 14.0
    label_size: float = 12.0
    tick_size: float = 10.0
    legend_size: float = 10.0

    ylim: tuple | None = None
    log_y: bool = False
    gridlines: bool = False
    despine: bool = True
    show_n: bool = True

    fig_w: float = 5.0
    fig_h: float = 4.0
    dpi: int = 300

    # significance bracket: (label_a, label_b, stars)
    sig_brackets: list = field(default_factory=list)


def _resolve_colors(labels, opts: PlotOptions):
    if opts.custom_colors:
        cols = list(opts.custom_colors)
    else:
        pal = PALETTES.get(opts.palette)
        if pal is None:
            cmap_name = "viridis" if "Viridis" in opts.palette else "tab10"
            cmap = plt.get_cmap(cmap_name)
            cols = [cmap(i / max(1, len(labels) - 1)) for i in range(len(labels))]
        else:
            cols = list(pal)
    # cycle if not enough colors
    return [cols[i % len(cols)] for i in range(len(labels))]


def bar_chart(
    summary: pd.DataFrame,
    opts: PlotOptions,
    raw_points: pd.DataFrame | None = None,
    order: list | None = None,
    label_col: str = "condition",
    value_col: str = "mean",
    point_value_col: str = "value",
):
    """Build a single-target bar chart figure.

    `summary` has one row per condition with columns: label_col, value_col,
    'sem'/'sd'/'ci95', 'n'. `raw_points` (optional) is tidy per-replicate data
    with columns [label_col, point_value_col] for the dot overlay.
    """
    plt.rcParams["font.family"] = opts.font_family

    labels = [str(l) for l in (order or list(summary[label_col]))]
    # reindex (not .loc) so a condition with no data yet plots as an empty slot
    # instead of raising KeyError.
    summary = summary.copy()
    summary[label_col] = summary[label_col].astype(str)
    summary = summary.set_index(label_col).reindex(labels).reset_index()
    if "n" in summary:
        summary["n"] = summary["n"].fillna(0)
    x = np.arange(len(labels))
    means = summary[value_col].to_numpy(float)

    if opts.error_type == "none":
        err = None
    else:
        err = summary.get(opts.error_type)
        err = err.to_numpy(float) if err is not None else None

    colors = _resolve_colors(labels, opts)

    fig, ax = plt.subplots(figsize=(opts.fig_w, opts.fig_h), dpi=opts.dpi)
    bars = ax.bar(
        x, means, color=colors, alpha=opts.bar_alpha,
        edgecolor=opts.bar_edge, linewidth=opts.bar_edge_width, zorder=2,
    )
    if err is not None:
        ax.errorbar(
            x, means, yerr=err, fmt="none", ecolor="black",
            capsize=opts.errbar_capsize, elinewidth=opts.errbar_width,
            capthick=opts.errbar_width, zorder=3,
        )

    if opts.show_points and raw_points is not None and len(raw_points):
        rng = np.random.default_rng(0)
        pt_labels = raw_points[label_col].astype(str)
        for i, lab in enumerate(labels):
            pts = raw_points.loc[pt_labels == lab, point_value_col].dropna()
            if len(pts) == 0:
                continue
            jit = rng.uniform(-opts.jitter, opts.jitter, size=len(pts))
            ax.scatter(
                np.full(len(pts), x[i]) + jit, pts.to_numpy(float),
                s=opts.point_size, color=opts.point_color,
                alpha=opts.point_alpha, edgecolors="white", linewidths=0.4, zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=opts.tick_size)
    ax.tick_params(axis="y", labelsize=opts.tick_size)
    ax.set_ylabel(opts.ylabel, fontsize=opts.label_size)
    if opts.xlabel:
        ax.set_xlabel(opts.xlabel, fontsize=opts.label_size)
    if opts.title:
        ax.set_title(opts.title, fontsize=opts.title_size)

    if opts.log_y:
        ax.set_yscale("log")
    if opts.ylim:
        ax.set_ylim(*opts.ylim)
    if opts.gridlines:
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    if opts.despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if opts.show_n and "n" in summary:
        # Anchor to the axes, not to a fraction of the data range: y*0.02 lands
        # off-plot on a log axis or one whose limits do not start at zero.
        trans = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        for i, nval in enumerate(summary["n"]):
            ax.text(x[i], 0.015, f"n={int(nval)}", transform=trans, ha="center",
                    va="bottom", fontsize=max(7, opts.tick_size - 2), color="#555")

    for a, b, stars in opts.sig_brackets:
        try:
            ia, ib = labels.index(a), labels.index(b)
        except ValueError:
            continue
        y = np.nanmax(means + (err if err is not None else 0)) * 1.08
        ax.plot([ia, ia, ib, ib], [y, y * 1.02, y * 1.02, y], color="black", lw=1.0)
        ax.text((ia + ib) / 2, y * 1.03, stars, ha="center", va="bottom",
                fontsize=opts.label_size)

    if opts.show_legend:
        ax.legend(bars, labels, fontsize=opts.legend_size, loc=opts.legend_loc,
                  frameon=False)

    if opts.caption:
        fig.text(0.01, -0.02, opts.caption, ha="left", va="top",
                 fontsize=max(7, opts.tick_size - 1), wrap=True)

    fig.tight_layout()
    return fig

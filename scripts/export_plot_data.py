"""Export tidy per-run data from a composite quantification workbook for plotting.

Usage:
    python scripts/export_plot_data.py WORKBOOK.xlsx OUT.csv [--rename OLD=NEW ...]

One row per run x variant x condition x cell line x target x lane, with the
loading-control-normalised value (``ratio_to_control``) and the QC verdict of that fold
change.  ``scripts/plot_composite_bars.R`` reads this file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["variant", "condition", "cell_line", "target"]


def export(workbook: Path, out: Path, rename: dict[str, str] | None = None) -> pd.DataFrame:
    per_run = pd.read_excel(workbook, sheet_name="Per-run normalised")
    per_run = per_run.rename(columns={"ratio_to_H3": "ratio_to_control"})  # older workbooks
    summary = pd.read_excel(workbook, sheet_name="Summary (fold change)")
    qc = summary[KEYS + ["QC", "QC notes"]]
    data = per_run.merge(qc, on=KEYS, how="left")
    data = data[["run", *KEYS, "lane_role", "lane_label", "ratio_to_control", "QC", "QC notes"]]
    if rename:
        for col in ("condition", "lane_label"):
            for old, new in rename.items():
                data[col] = data[col].astype(str).str.replace(old, new, regex=False)
    data.to_csv(out, index=False)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("workbook", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rename", nargs="*", default=[],
                    help="label substitutions OLD=NEW, applied to condition and lane labels")
    args = ap.parse_args(argv)
    rename = dict(item.split("=", 1) for item in args.rename)
    data = export(args.workbook, args.out, rename)
    print(f"wrote {args.out} ({len(data)} rows)")


if __name__ == "__main__":
    main()

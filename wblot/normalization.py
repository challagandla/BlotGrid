"""Per-lane normalization against a loading control, plus fold-change scaling.

Core rule (per blot):
    normalized[target, lane] = net_intden[target, lane] / net_intden[control, lane]

Optionally rescale so a chosen reference lane == 1.0 (fold change).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def has_usable_control(
    df: pd.DataFrame,
    control_target: str,
    *,
    value_col: str = "net_intden",
    target_col: str = "target",
) -> bool:
    """Return whether at least one positive measured control band is present."""
    if not control_target or target_col not in df.columns or value_col not in df.columns:
        return False
    values = pd.to_numeric(
        df.loc[df[target_col] == control_target, value_col], errors="coerce")
    return bool((values > 0).any())


def prepare_intensities(
    df: pd.DataFrame,
    value_col: str = "net_intden",
    target_col: str = "target",
    lane_col: str = "lane",
    multiband: str = "sum",
) -> pd.DataFrame:
    """Return per-target/lane intensity values without control normalization."""
    required = [target_col, lane_col, value_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"prepare_intensities() missing column(s): {missing}")
    if multiband not in {"sum", "mean", "keep"}:
        raise ValueError("multiband must be 'sum', 'mean', or 'keep'.")
    work = df.copy()
    if multiband in {"sum", "mean"}:
        return (
            work.groupby([target_col, lane_col], as_index=False)[value_col]
            .agg("sum" if multiband == "sum" else "mean")
        )
    result = work[[target_col, lane_col, value_col]].copy()
    result["band_index"] = result.groupby([target_col, lane_col]).cumcount()
    return result


def normalize_to_control(
    df: pd.DataFrame,
    control_target: str,
    value_col: str = "net_intden",
    target_col: str = "target",
    lane_col: str = "lane",
    multiband: str = "sum",
) -> pd.DataFrame:
    """Divide each target band by the loading control in the same lane.

    df must contain columns [target_col, lane_col, value_col]. If a target has
    multiple bands in a lane, `multiband` controls collapsing:
      'sum'  -> add the bands (default)
      'mean' -> average them
      'keep' -> keep separate rows (adds a band_index)

    Returns a tidy DataFrame with a 'normalized' column. Control rows are kept
    (their normalized value is 1.0 by definition).
    """
    required = [target_col, lane_col, value_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"normalize_to_control() missing column(s): {missing}")
    if multiband not in {"sum", "mean", "keep"}:
        raise ValueError("multiband must be 'sum', 'mean', or 'keep'.")
    if control_target not in set(df[target_col]):
        raise ValueError(f"Loading control '{control_target}' not found among targets.")

    work = df.copy()

    if multiband in ("sum", "mean"):
        agg = "sum" if multiband == "sum" else "mean"
        collapsed = (
            work.groupby([target_col, lane_col], as_index=False)[value_col].agg(agg)
        )
    else:  # keep separate bands
        collapsed = work.copy()
        collapsed["band_index"] = collapsed.groupby([target_col, lane_col]).cumcount()

    # Control value per lane (collapse control bands by sum regardless).
    ctrl = (
        work[work[target_col] == control_target]
        .groupby(lane_col, as_index=False)[value_col].sum()
        .rename(columns={value_col: "_control_value"})
    )

    merged = collapsed.merge(ctrl, on=lane_col, how="left")

    def _safe_div(row):
        c = row["_control_value"]
        if c is None or pd.isna(c) or float(c) <= 0.0:
            return np.nan
        return row[value_col] / float(c)

    merged["normalized"] = merged.apply(_safe_div, axis=1)
    merged = merged.drop(columns=["_control_value"])
    return merged


def fold_change(
    df: pd.DataFrame,
    reference_lane,
    target_col: str = "target",
    lane_col: str = "lane",
    value_col: str = "normalized",
    out_col: str = "fold_change",
) -> pd.DataFrame:
    """Rescale normalized values so the reference lane == 1.0 per target."""
    work = df.copy()
    required = [target_col, lane_col, value_col]
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise ValueError(f"fold_change() missing column(s): {missing}")

    # With multiband='keep', each species must be scaled to the corresponding
    # species in the reference lane.  Indexing only by target silently used the
    # final reference band for every species.
    key_cols = [target_col]
    if "band_index" in work.columns:
        key_cols.append("band_index")
    ref_rows = work[work[lane_col] == reference_lane]
    ref = {
        tuple(row[col] for col in key_cols): row[value_col]
        for _, row in ref_rows.iterrows()
    }

    def _fc(row):
        key = tuple(row[col] for col in key_cols)
        base = ref.get(key)
        if base is None or pd.isna(base) or float(base) == 0.0:
            return np.nan
        return row[value_col] / base

    work[out_col] = work.apply(_fc, axis=1)
    return work

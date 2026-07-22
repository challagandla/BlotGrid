"""Replicate aggregation and optional significance testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps


def aggregate(
    df: pd.DataFrame,
    value_col: str,
    group_cols=("target", "condition"),
) -> pd.DataFrame:
    """Aggregate replicate values into mean, SD, SEM and N per group.

    SEM = SD / sqrt(N), SD with ddof=1. With N=1 there is no spread to report,
    so SD/SEM/CI are 0.0. ci95 is the half-width of the t-based 95% confidence
    interval, which is much wider than 1.96*SEM at the N=3 typical of blots.
    """
    group_cols = list(group_cols)
    missing = [c for c in group_cols + [value_col] if c not in df.columns]
    if missing:
        raise ValueError(f"aggregate() missing column(s): {missing}")

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = g[value_col].dropna().to_numpy(dtype=float)
        n = len(vals)
        mean = float(np.mean(vals)) if n else np.nan
        sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        sem = sd / np.sqrt(n) if n > 1 else 0.0
        ci95 = float(_sps.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
        rows.append(dict(zip(group_cols, keys),
                         mean=mean, sd=sd, sem=sem, ci95=ci95, n=n))

    out = pd.DataFrame(rows, columns=group_cols + ["mean", "sd", "sem", "ci95", "n"])
    out["n"] = out["n"].astype(int)
    return out


def compare_groups(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "condition",
    target: str | None = None,
    target_col: str = "target",
    paired: bool = False,
) -> dict:
    """Run a t-test (2 groups) or one-way ANOVA (>2 groups).

    Returns a dict describing the test and p-value(s). This is a convenience
    helper, not a substitute for a proper statistical plan.
    """
    work = df if target is None else df[df[target_col] == target]
    labels, groups = [], []
    for key, g in work.groupby(group_col, sort=False):
        vals = g[value_col].dropna().to_numpy(float)
        if len(vals):          # drop empty groups and their labels together
            labels.append(str(key))
            groups.append(vals)
    if len(groups) < 2:
        return {"test": None, "reason": "need >=2 groups with data"}

    if len(groups) == 2:
        if paired and len(groups[0]) == len(groups[1]):
            t, p = _sps.ttest_rel(groups[0], groups[1])
            name = "paired t-test"
        else:
            t, p = _sps.ttest_ind(groups[0], groups[1], equal_var=False)
            name = "Welch t-test"
        return {"test": name, "statistic": float(t), "p_value": float(p), "labels": labels}

    f, p = _sps.f_oneway(*groups)
    return {"test": "one-way ANOVA", "statistic": float(f), "p_value": float(p), "labels": labels}


def p_to_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

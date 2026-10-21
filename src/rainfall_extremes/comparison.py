"""
Recent change and Bologna-Padova comparison (Tasks 5-6).

1. Is the most recent window different from the previous, non-overlapping
   one? Two DISJOINT 30-year windows share no years, so their bootstrap
   replicates are independent and the bootstrap distribution of the
   difference (last - previous) is simply the difference of the replicates.
   Each window is resampled exactly as in Task 4 (same seeds), so the bands
   are the ones of Fig. 17. Reported for the GEV, the main MEVD (EW = 1 yr),
   the pooled MEVD (EW = whole window) and the drivers mean N and mean C.
   ``share_le_0`` is the bootstrap share of differences <= 0, a one-sided
   indication for an increase (not a formal p-value: the percentile
   bootstrap is too narrow for the GEV, Task 4).

2. Are temporal changes shared by the two stations? Year-by-year rank
   correlations of the annual quantities over the common valid years, and
   correlations of the window series. Window series overlap by 29 years,
   so their correlation is reported without a p-value (the effective number
   of independent values is only about (number of windows) / 30).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .bootstrap import bootstrap_window, resample_indices, window_samples
from .estimation_window import (counts_from_indices, pooled_pwm_weighted,
                                pooled_window_data)
from . import mev
from .windows import build_windows


def _window(annual: pd.DataFrame, end: int):
    for w in build_windows(annual, "calendar"):
        if w.end == end:
            return w
    raise ValueError(f"No retained window ending in {end}.")


def _replicates(station, annual, yearly, exc, win, B, seed):
    """Point estimates and B replicates of every quantity for one window."""
    ws = window_samples(station, annual, yearly, [win])[0]
    bw = bootstrap_window(ws, B=B, seed=seed)
    idx = resample_indices(ws.n, B, station, win.end, seed)
    F = 1.0 - 1.0 / config.RETURN_PERIOD
    xs, lab, N = pooled_window_data(win.years, exc)
    fit = mev.weibull_fit_pwm(xs)
    n = N.size
    Cp, wp = pooled_pwm_weighted(xs, lab, counts_from_indices(idx, n))
    pooled_reps = mev.mev_quantile_batch(F, N[idx], np.repeat(Cp[:, None], n, 1),
                                         np.repeat(wp[:, None], n, 1))
    pooled_point = mev.mev_quantile(F, N, np.full(n, fit.C), np.full(n, fit.w))
    return {
        "GEV": (bw.rl_gev_point, bw.rl_gev),
        "MEVD (EW = 1 yr)": (bw.rl_mev_point, bw.rl_mev),
        "MEVD (EW = window)": (pooled_point, pooled_reps),
        "mean N [days]": (ws.N.mean(), ws.N[idx].mean(axis=1)),
        "mean C [mm]": (ws.C.mean(), ws.C[idx].mean(axis=1)),
        "mean w [-]": (ws.w.mean(), ws.w[idx].mean(axis=1)),
    }


def change_between_windows(station: str, annual: pd.DataFrame,
                           yearly: pd.DataFrame, exc: dict,
                           first_end: int, second_end: int,
                           B: int = config.N_BOOTSTRAP,
                           seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Second window minus first window, with bootstrap 90% intervals."""
    w1, w2 = _window(annual, first_end), _window(annual, second_end)
    if set(w1.years) & set(w2.years):
        raise ValueError("The two windows overlap: replicates not independent.")
    r1 = _replicates(station, annual, yearly, exc, w1, B, seed)
    r2 = _replicates(station, annual, yearly, exc, w2, B, seed)
    lo, hi = config.CI_PERCENTILES
    rows = []
    for q in r1:
        (p1, b1), (p2, b2) = r1[q], r2[q]
        d = b2 - b1
        rows.append({"station": station, "quantity": q,
                     "first": f"{w1.start}-{w1.end}",
                     "second": f"{w2.start}-{w2.end}",
                     "first_value": p1, "second_value": p2,
                     "change": p2 - p1, "rel_change": (p2 - p1) / p1,
                     f"change_p{lo:g}": np.percentile(d, lo),
                     f"change_p{hi:g}": np.percentile(d, hi),
                     "share_le_0": float(np.mean(d <= 0))})
    return pd.DataFrame(rows)


def yearly_correlations(yearly_a: pd.DataFrame, yearly_b: pd.DataFrame,
                        annual_a: pd.DataFrame, annual_b: pd.DataFrame
                        ) -> pd.DataFrame:
    """Spearman correlation of annual quantities over common valid years."""
    def table(y, a):
        v = a.loc[a["valid"], ["year", "total_mm", "annual_max_mm"]]
        return v.merge(y[["year", "N", "C", "w"]], on="year")
    ta, tb = table(yearly_a, annual_a), table(yearly_b, annual_b)
    m = ta.merge(tb, on="year", suffixes=("_a", "_b"))
    rows = []
    for col, label in (("total_mm", "annual total"), ("N", "wet days N"),
                       ("C", "Weibull scale C"), ("w", "Weibull shape w"),
                       ("annual_max_mm", "annual maximum")):
        r, p = spearmanr(m[f"{col}_a"], m[f"{col}_b"])
        rows.append({"quantity": label, "spearman_r": r, "p_value": p,
                     "n_years": len(m), "first_year": int(m["year"].min()),
                     "last_year": int(m["year"].max())})
    return pd.DataFrame(rows)


def window_correlations(series_a: pd.DataFrame, series_b: pd.DataFrame,
                        columns: dict[str, str]) -> pd.DataFrame:
    """Spearman correlation of window series (indexed by final year)."""
    common = series_a.index.intersection(series_b.index)
    rows = []
    for col, label in columns.items():
        r, _ = spearmanr(series_a.loc[common, col], series_b.loc[common, col])
        rows.append({"quantity": label, "spearman_r": r,
                     "n_windows": len(common),
                     "approx_independent_values": len(common) / 30,
                     "first_end": int(common.min()),
                     "last_end": int(common.max())})
    return pd.DataFrame(rows)

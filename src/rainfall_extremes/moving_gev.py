"""
Moving-window GEV analysis (Task 2).

For every retained window the annual maxima of its VALID years are fitted
with a GEV, and return levels are computed. Results are returned as a tidy
table with one row per (scenario, window mode, estimator, window), so later
tasks (bootstrap, MEVD comparison) can join on the same keys.
"""
from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from . import config, gev
from .windows import Window, build_windows

RETURN_PERIODS: tuple[float, ...] = (100.0, config.RETURN_PERIOD)


def window_maxima(annual: pd.DataFrame, window: Window) -> np.ndarray:
    """Annual maxima of the valid years of one window (block maxima sample)."""
    sel = annual["year"].isin(window.years) & annual["valid"]
    return annual.loc[sel, "annual_max_mm"].to_numpy(dtype=float)


def gev_over_windows(annual: pd.DataFrame, windows: list[Window],
                     methods: Iterable[str] = ("ml", "lmom"),
                     return_periods: Iterable[float] = RETURN_PERIODS
                     ) -> pd.DataFrame:
    """Fit the GEV in every window with every estimator."""
    rows = []
    for w, method in product(windows, methods):
        x = window_maxima(annual, w)
        p = gev.fit(x, method)
        row = {
            "window_start": w.start, "window_end": w.end,
            "window_mode": w.mode, "n_valid": w.n_valid,
            "method": method, "xi": p.xi, "sigma": p.sigma, "mu": p.mu,
            "nll": p.nll, "converged": p.converged,
            "sample_max_mm": float(x.max()), "sample_mean_mm": float(x.mean()),
        }
        for T in return_periods:
            row[f"rl_{int(T)}"] = float(gev.return_level(T, p))
        rows.append(row)
    return pd.DataFrame(rows)


def run_station(annual_long: pd.DataFrame,
                scenarios: Iterable[str] = ("asis", "qc"),
                modes: Iterable[str] = ("calendar", "consecutive"),
                methods: Iterable[str] = ("ml", "lmom")) -> pd.DataFrame:
    """All combinations of scenario x window mode x estimator for a station."""
    station = str(annual_long["station"].iat[0])
    parts = []
    for scen, mode in product(scenarios, modes):
        annual = annual_long.loc[annual_long["scenario"] == scen]
        wins = build_windows(annual, mode)
        res = gev_over_windows(annual, wins, methods)
        res.insert(0, "scenario", scen)
        res.insert(0, "station", station)
        parts.append(res)
    return pd.concat(parts, ignore_index=True)


def select(results: pd.DataFrame, scenario: str = "asis",
           mode: str = "calendar", method: str = "ml") -> pd.DataFrame:
    """Rows of one configuration, sorted by window end year."""
    sel = ((results["scenario"] == scenario) & (results["window_mode"] == mode)
           & (results["method"] == method))
    return results.loc[sel].sort_values("window_end").reset_index(drop=True)


def summarise(results: pd.DataFrame, T: float = config.RETURN_PERIOD
              ) -> pd.DataFrame:
    """Headline numbers per station for the report."""
    col = f"rl_{int(T)}"
    rows = []
    for station, res in results.groupby("station"):
        main = select(res)
        lm = select(res, method="lmom")
        qc = select(res, scenario="qc")
        cons = select(res, mode="consecutive")
        imin, imax = main[col].idxmin(), main[col].idxmax()
        d_lm = main.merge(lm, on="window_end", suffixes=("", "_lm"))
        d_qc = main.merge(qc, on="window_end", suffixes=("", "_qc"))
        rows.append({
            "station": station,
            "n_windows": len(main),
            "first_window_end": int(main["window_end"].min()),
            "last_window_end": int(main["window_end"].max()),
            f"{col}_min": main[col].min(),
            "window_of_min": f"{main.at[imin, 'window_start']}-{main.at[imin, 'window_end']}",
            f"{col}_median": main[col].median(),
            f"{col}_max": main[col].max(),
            "window_of_max": f"{main.at[imax, 'window_start']}-{main.at[imax, 'window_end']}",
            f"{col}_last_window": main[col].iat[-1],
            "xi_min": main["xi"].min(), "xi_max": main["xi"].max(),
            "share_xi_positive": float((main["xi"] > 0).mean()),
            "n_not_converged": int((~main["converged"]).sum()),
            "median_abs_diff_ml_lmom_mm": float((d_lm[col] - d_lm[f"{col}_lm"]).abs().median()),
            "max_abs_diff_ml_lmom_mm": float((d_lm[col] - d_lm[f"{col}_lm"]).abs().max()),
            "n_windows_qc": len(qc),
            "max_abs_diff_asis_qc_mm": float((d_qc[col] - d_qc[f"{col}_qc"]).abs().max())
            if len(d_qc) else np.nan,
            "n_windows_consecutive": len(cons),
        })
    return pd.DataFrame(rows)

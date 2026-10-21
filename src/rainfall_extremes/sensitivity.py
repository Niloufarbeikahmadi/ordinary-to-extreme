"""
Sensitivity of the 200-year estimates to the analysis choices (Task 5).

Every analysis choice is changed ONE AT A TIME, with everything else held
at the main configuration (record as provided, 30 calendar-year windows,
GEV by maximum likelihood, MEVD with yearly PWM Weibull fits to excesses
over 1 mm). The change of the 200-year level is computed window by window
(same final year) and summarised by its median and its largest absolute
value.

Choices tested here (Tasks 2-4 add estimator, QC scenario, window type,
raw vs excess formulation and the Weibull estimator):

* Window length: 20, 40 and 50 calendar years (a window is kept if at least
  two thirds of its years are valid, the 20-of-30 rule scaled).
* MEVD estimation window (``estimation_window.py``): 3 and 5 years, the
  whole window ("pooled"), and SMEV.
* Tail-focused Weibull: one Weibull per window fitted by maximum likelihood
  with LEFT censoring of the lowest 75% of the wet-day excesses, so the fit
  is driven by the largest ordinary events (checks the assumption that one
  Weibull describes both the bulk and the tail).
* Recording resolution ("coarsening" experiment): every daily value is
  rounded to the nearest 1 mm, as in the coarse early readings of Bologna.
  With P > 1 mm a 1-mm reading counts only from 2 mm, so the effective
  wet-day threshold becomes 1.5 mm.
* Unit of the earliest Padova readings (1725-1764): the values are exact
  multiples of 1/1000 English inch, while Marani & Zanetti (2015) report
  French (Paris) inches before 1864. If Paris inches were converted with
  the English factor, the true depths are 27.07 / 25.4 = 1.066 times larger.
* Valid-year threshold (5% or 20% of days missing instead of 10%): checked
  directly on the missing-data fractions, see ``valid_year_rule_effect``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config, gev_batch, mev
from .estimation_window import (excess_by_year, pooled_window_data,
                                window_parameters)
from .windows import Window, build_windows

RL = f"rl_{int(config.RETURN_PERIOD)}"
#: Ratio of the Paris inch (27.07 mm) to the English inch (25.4 mm).
PARIS_TO_ENGLISH_INCH: float = 27.07 / 25.4
ENGLISH_INCH_LAST_YEAR: int = 1764
TAIL_CENSOR_QUANTILE: float = 0.75


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def gev_rl_windows(annual: pd.DataFrame, windows: list[Window],
                   T: float = config.RETURN_PERIOD) -> pd.Series:
    """GEV (ML) T-year level per window, indexed by final year.

    Uses the batched fitter of Task 4 one window at a time (it coincides
    with ``gev.fit_ml`` for every regular fit and is much faster).
    """
    amax = annual.set_index("year")["annual_max_mm"]
    out = {}
    for w in windows:
        fit = gev_batch.fit_ml(amax.loc[list(w.years)].to_numpy(float)[None, :])
        out[w.end] = float(fit.return_level(T)[0])
    return pd.Series(out, name=RL)


def mev_rl_windows(yearly: pd.DataFrame, windows: list[Window],
                   T: float = config.RETURN_PERIOD) -> pd.Series:
    """MEVD T-year level per window from yearly (N, C, w)."""
    yp = yearly.set_index("year")
    out = {}
    for w in windows:
        p = yp.loc[list(w.years)]
        out[w.end] = mev.mev_return_level(T, p["N"], p["C"], p["w"])
    return pd.Series(out, name=RL)


def annual_from_daily(daily: pd.DataFrame, annual_ref: pd.DataFrame,
                      value_col: str = "precip_mm") -> pd.DataFrame:
    """Annual maxima recomputed from modified daily values (validity unchanged)."""
    mx = daily.groupby("year")[value_col].max()
    out = annual_ref.copy()
    out["annual_max_mm"] = out["year"].map(mx).astype(float)
    return out


def tail_fit_windows(exc: dict[int, np.ndarray], windows: list[Window],
                     q: float = TAIL_CENSOR_QUANTILE,
                     T: float = config.RETURN_PERIOD) -> pd.DataFrame:
    """Pooled Weibull fitted to the upper tail only (left-censored ML)."""
    rows = []
    for w in windows:
        xs, _, N = pooled_window_data(w.years, exc)
        pwm = mev.weibull_fit_pwm(xs)
        tail = mev.weibull_fit_ml_censored(xs, float(np.quantile(xs, q)))
        n = N.size
        rows.append({"window_end": w.end, "w_pwm": pwm.w, "w_tail": tail.w,
                     "C_pwm": pwm.C, "C_tail": tail.C,
                     RL: mev.mev_return_level(T, N, np.full(n, tail.C),
                                              np.full(n, tail.w))})
    return pd.DataFrame(rows).set_index("window_end")


def valid_year_rule_effect(annual: pd.DataFrame,
                           fractions=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Number of valid years under different missing-data thresholds."""
    fm = annual["frac_missing"]
    return pd.DataFrame({
        "max_missing_fraction": fractions,
        "n_valid_years": [int((fm < f).sum()) for f in fractions],
        "n_partly_missing_years": int(((fm > 0) & (fm < 1)).sum()),
    })


# ----------------------------------------------------------------------
# All one-at-a-time variants of one station
# ----------------------------------------------------------------------
def run_station(station: str, daily: pd.DataFrame, annual: pd.DataFrame,
                yearly: pd.DataFrame, ew_table: pd.DataFrame,
                gev_tasks: pd.DataFrame, mev_tasks: pd.DataFrame
                ) -> tuple[pd.DataFrame, dict]:
    """Variant series (long table) for one station.

    ``ew_table`` comes from ``estimation_window.ew_over_windows``;
    ``gev_tasks`` / ``mev_tasks`` are the Task 2 / Task 3 result tables.
    Returns the long table (window_end, model, variant, rl) and extras.
    """
    wins = build_windows(annual, "calendar")
    rows = []

    def add(model, variant, series):
        for end, v in series.items():
            rows.append({"station": station, "model": model, "variant": variant,
                         "window_end": int(end), RL: float(v)})

    # Main configuration (from the earlier tasks).
    g = gev_tasks[(gev_tasks["scenario"] == "asis")
                  & (gev_tasks["window_mode"] == "calendar")]
    m = mev_tasks[(mev_tasks["scenario"] == "asis")
                  & (mev_tasks["window_mode"] == "calendar")]
    add("GEV", "main", g[g["method"] == "ml"].set_index("window_end")[RL])
    sel = (m["formulation"] == "excess") & (m["method"] == "pwm")
    add("MEV", "main", m[sel].set_index("window_end")[RL])

    # Earlier tasks' variants, re-labelled.
    add("GEV", "estimator: L-moments",
        g[g["method"] == "lmom"].set_index("window_end")[RL])
    gq = gev_tasks[(gev_tasks["scenario"] == "qc")
                   & (gev_tasks["window_mode"] == "calendar")
                   & (gev_tasks["method"] == "ml")]
    add("GEV", "data: QC-screened", gq.set_index("window_end")[RL])
    gc = gev_tasks[(gev_tasks["scenario"] == "asis")
                   & (gev_tasks["window_mode"] == "consecutive")
                   & (gev_tasks["method"] == "ml")]
    add("GEV", "windows: 30 available years", gc.set_index("window_end")[RL])
    for label, mask in (
            ("formulation: raw values",
             (m["formulation"] == "raw") & (m["method"] == "pwm")),
            ("Weibull estimator: ML",
             (m["formulation"] == "excess") & (m["method"] == "ml"))):
        add("MEV", label, m[mask].set_index("window_end")[RL])
    mq = mev_tasks[(mev_tasks["scenario"] == "qc")
                   & (mev_tasks["window_mode"] == "calendar")]
    add("MEV", "data: QC-screened", mq.set_index("window_end")[RL])
    mc = mev_tasks[(mev_tasks["scenario"] == "asis")
                   & (mev_tasks["window_mode"] == "consecutive")]
    add("MEV", "windows: 30 available years", mc.set_index("window_end")[RL])

    # Estimation window and SMEV.
    for ew, label in (("3", "estimation window: 3 years"),
                      ("5", "estimation window: 5 years"),
                      ("all", "estimation window: whole window"),
                      ("smev", "SMEV (mean N)")):
        add("MEV", label, ew_table[ew_table["ew"] == ew]
            .set_index("window_end")[RL])

    # Tail-focused Weibull.
    exc = excess_by_year(daily)
    tail = tail_fit_windows(exc, wins)
    add("MEV", "tail-focused Weibull (censored 75%)", tail[RL])

    # Window length.
    for L in (20, 40, 50):
        wl = build_windows(annual, "calendar", length=L,
                           min_valid=math.ceil(2 * L / 3))
        add("GEV", f"window length: {L} years", gev_rl_windows(annual, wl))
        add("MEV", f"window length: {L} years", mev_rl_windows(yearly, wl))

    # Recording resolution: round every value to the nearest 1 mm.
    coarse = daily.copy()
    coarse["precip_mm"] = np.floor(coarse["precip_mm"] + 0.5)
    ann_c = annual_from_daily(coarse, annual)
    add("GEV", "resolution: rounded to 1 mm", gev_rl_windows(ann_c, wins))
    add("MEV", "resolution: rounded to 1 mm",
        mev_rl_windows(mev.yearly_parameters(coarse), wins))

    # Unit of the earliest Padova readings.
    if station == "padova":
        scaled = daily.copy()
        early = scaled["year"] <= ENGLISH_INCH_LAST_YEAR
        scaled.loc[early, "precip_mm"] *= PARIS_TO_ENGLISH_INCH
        ann_s = annual_from_daily(scaled, annual)
        aff = [w for w in wins if w.start <= ENGLISH_INCH_LAST_YEAR]
        add("GEV", "unit 1725-1764: Paris inch", gev_rl_windows(ann_s, aff))
        add("MEV", "unit 1725-1764: Paris inch",
            mev_rl_windows(mev.yearly_parameters(scaled), aff))

    return pd.DataFrame(rows), {"tail": tail,
                                "valid_rule": valid_year_rule_effect(annual)}


def one_at_a_time(long: pd.DataFrame) -> pd.DataFrame:
    """Change of the 200-year level vs the main configuration, per variant."""
    out = []
    for (st, model), g in long.groupby(["station", "model"], sort=False):
        main = g[g["variant"] == "main"].set_index("window_end")[RL]
        for var, gv in g[g["variant"] != "main"].groupby("variant", sort=False):
            v = gv.set_index("window_end")[RL]
            common = main.index.intersection(v.index)
            d = v.loc[common] - main.loc[common]
            rel = d / main.loc[common]
            out.append({"station": st, "model": model, "variant": var,
                        "n_windows": len(common),
                        "median_change_mm": d.median(),
                        "p05_change_mm": d.quantile(0.05),
                        "p95_change_mm": d.quantile(0.95),
                        "median_abs_change_mm": d.abs().median(),
                        "max_abs_change_mm": d.abs().max(),
                        "median_rel_change": rel.median()})
    return pd.DataFrame(out)

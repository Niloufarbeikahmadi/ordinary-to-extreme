"""
Common non-parametric bootstrap of the GEV and MEVD return levels (Task 4).

What is resampled
-----------------
Following the task, the unit of resampling is the YEAR. Each valid year j of
a window is represented by

* its annual maximum M_j, for the GEV, and
* its ordinary-event parameters (N_j, C_j, w_j), for the MEVD,

and one bootstrap repetition draws n years WITH replacement from the n valid
years of the window. The SAME drawn years feed both models ("the same
synthetic samples for the MEVD and GEV calculations within each window").
The GEV is refitted to the drawn maxima by maximum likelihood; the MEVD is
recomputed from the drawn (N, C, w) triplets. After B = 500 repetitions the
pointwise 90% interval is given by the 5th and 95th percentiles of the 500
estimates, separately for each model and window.

Why resample whole years
------------------------
Within a year, wet days are correlated (storms last several days, seasons
differ), and N, C and w of the same year are mutually dependent. Drawing
whole years keeps all of this intact; only the assumption that DIFFERENT
years are independent is needed, which is the same assumption both models
already make.

Relation to mevpy
-----------------
mevpy's ``mev_CI`` (default ``std_how='boot'``) uses a two-stage scheme: it
draws years and then also redraws the daily values within each drawn year,
refitting the Weibull every time, and it reports mean +/- 1.96 sd. The task
asks for the simpler year-level scheme with percentile intervals, which is
implemented here. (mevpy also contains a commented-out ``'par'`` option
close to this scheme.)

Because the two models see identical resamples, the bootstrap distribution
of the DIFFERENCE GEV - MEVD is also available for every window: a paired
comparison that tells where the two methods disagree by more than sampling
noise.

Reproducibility
---------------
Each window gets its own random generator, seeded with
(RANDOM_SEED, station code, final year of the window). Results for a window
therefore do not depend on which other windows are run, or in which order.
The station code is a CRC-32 checksum of the station name, which, unlike
Python's ``hash()``, is identical in every session.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, gev, gev_batch, mev
from .windows import Window


@dataclass(frozen=True)
class WindowSample:
    """The valid years of one window, in the form both models need."""
    station: str
    start: int
    end: int
    years: np.ndarray       # (n,) valid years
    amax: np.ndarray        # (n,) annual maxima [mm]
    N: np.ndarray           # (n,) wet days (P > 1 mm)
    C: np.ndarray           # (n,) Weibull scale of excesses [mm]
    w: np.ndarray           # (n,) Weibull shape [-]

    @property
    def n(self) -> int:
        return self.years.size


@dataclass
class WindowBootstrap:
    """Point estimates and B bootstrap replicates for one window."""
    sample: WindowSample
    gev_point: gev.GEVParams
    rl_gev_point: float
    rl_mev_point: float
    rl_gev: np.ndarray                  # (B,)
    rl_mev: np.ndarray                  # (B,)
    xi: np.ndarray                      # (B,) GEV shape of each replicate
    gev_bound: np.ndarray               # (B,) degenerate GEV fit (xi on a bound)
    extras: dict = field(default_factory=dict)


def station_code(station: str) -> int:
    """Session-independent integer code of a station name (CRC-32)."""
    return zlib.crc32(station.encode("utf-8"))


def window_samples(station: str, annual: pd.DataFrame, yearly: pd.DataFrame,
                   windows: list[Window]) -> list[WindowSample]:
    """Join annual maxima and yearly (N, C, w) for the valid years of each window.

    ``yearly`` must hold ONE configuration (e.g. asis / excess / PWM). A
    valid year without Weibull parameters would break the pairing of the two
    models, so it raises an error instead of being dropped silently.
    """
    amax = annual.set_index("year")["annual_max_mm"]
    yp = yearly.set_index("year")
    out = []
    for win in windows:
        yrs = np.array(win.years, dtype=int)
        p = yp.loc[yrs, ["N", "C", "w"]]
        if p.isna().any().any():
            raise ValueError(f"{station} {win.start}-{win.end}: a valid year "
                             f"has no Weibull fit; GEV and MEVD would not "
                             f"share the same years.")
        out.append(WindowSample(station, win.start, win.end, yrs,
                                amax.loc[yrs].to_numpy(float),
                                p["N"].to_numpy(float), p["C"].to_numpy(float),
                                p["w"].to_numpy(float)))
    return out


def resample_indices(n: int, B: int, station: str, window_end: int,
                     seed: int = config.RANDOM_SEED) -> np.ndarray:
    """(B, n) indices of years drawn with replacement, reproducible per window."""
    rng = np.random.default_rng([seed, station_code(station), int(window_end)])
    return rng.integers(0, n, size=(B, n))


def bootstrap_window(ws: WindowSample, B: int = config.N_BOOTSTRAP,
                     T: float = config.RETURN_PERIOD,
                     seed: int = config.RANDOM_SEED,
                     formulation: str = "excess") -> WindowBootstrap:
    """Common bootstrap of one window: same resampled years for both models."""
    F = 1.0 - 1.0 / T
    # Point estimates on the original sample (exactly as in Tasks 2 and 3).
    p0 = gev.fit_ml(ws.amax)
    rl_gev_point = float(gev.return_level(T, p0))
    rl_mev_point = mev.mev_quantile(F, ws.N, ws.C, ws.w,
                                    formulation=formulation)
    # One index matrix for both models.
    idx = resample_indices(ws.n, B, ws.station, ws.end, seed)
    fit = gev_batch.fit_ml(ws.amax[idx], start=p0.as_tuple())
    rl_gev = fit.return_level(T)
    rl_mev = mev.mev_quantile_batch(F, ws.N[idx], ws.C[idx], ws.w[idx],
                                    formulation=formulation)
    return WindowBootstrap(ws, p0, rl_gev_point, rl_mev_point, rl_gev, rl_mev,
                           fit.xi, fit.at_boundary,
                           extras={"gev_converged": fit.converged,
                                   "n_distinct_years": np.array(
                                       [np.unique(r).size for r in idx])})


def summarise_window(bw: WindowBootstrap,
                     pct: tuple[float, float] = config.CI_PERCENTILES) -> dict:
    """One table row: point estimates, percentile bands and diagnostics."""
    lo, hi = pct
    ws = bw.sample
    row = {"station": ws.station, "window_start": ws.start,
           "window_end": ws.end, "n_valid": ws.n}
    for m, point, reps in (("gev", bw.rl_gev_point, bw.rl_gev),
                           ("mev", bw.rl_mev_point, bw.rl_mev)):
        q_lo, q_med, q_hi = np.percentile(reps, [lo, 50.0, hi])
        row.update({
            f"{m}_rl": point, f"{m}_p{lo:g}": q_lo, f"{m}_p50": q_med,
            f"{m}_p{hi:g}": q_hi,
            # width of the 90% band relative to the point estimate
            f"{m}_rel_width": (q_hi - q_lo) / point,
            # bootstrap median vs point estimate: a skewness / bias indicator
            f"{m}_median_shift": q_med - point,
        })
    d = bw.rl_gev - bw.rl_mev                     # paired difference
    d_lo, d_med, d_hi = np.percentile(d, [lo, 50.0, hi])
    row.update({
        "diff_point": bw.rl_gev_point - bw.rl_mev_point,
        f"diff_p{lo:g}": d_lo, "diff_p50": d_med, f"diff_p{hi:g}": d_hi,
        "share_gev_above_mev": float(np.mean(d > 0)),
        "gev_xi_point": bw.gev_point.xi,
        f"gev_xi_p{lo:g}": np.percentile(bw.xi, lo),
        f"gev_xi_p{hi:g}": np.percentile(bw.xi, hi),
        "gev_share_irregular": float(np.mean(bw.xi <= -0.5)),
        "gev_share_lower_bound": float(np.mean(bw.gev_bound & (bw.xi < 0))),
        "gev_share_upper_bound": float(np.mean(bw.gev_bound & (bw.xi > 0))),
        "mean_distinct_years": float(np.mean(bw.extras["n_distinct_years"])),
    })
    return row


def run_station(station: str, annual: pd.DataFrame, yearly: pd.DataFrame,
                windows: list[Window], B: int = config.N_BOOTSTRAP,
                T: float = config.RETURN_PERIOD,
                seed: int = config.RANDOM_SEED, verbose: bool = True
                ) -> tuple[pd.DataFrame, dict]:
    """Bootstrap every window of one station.

    Returns the summary table (one row per window) and the replicates as a
    dict of arrays (window x repetition), ready for ``np.savez_compressed``.
    """
    t0 = time.time()
    samples = window_samples(station, annual, yearly, windows)
    rows, reps = [], {k: [] for k in ("rl_gev", "rl_mev", "xi", "gev_bound")}
    for ws in samples:
        bw = bootstrap_window(ws, B, T, seed)
        rows.append(summarise_window(bw))
        for k in reps:
            reps[k].append(getattr(bw, k))
    out = {k: np.vstack(v) for k, v in reps.items()}
    out["window_end"] = np.array([ws.end for ws in samples])
    out["rl_gev_point"] = np.array([r["gev_rl"] for r in rows])
    out["rl_mev_point"] = np.array([r["mev_rl"] for r in rows])
    if verbose:
        print(f"  {station}: {len(samples)} windows x {B} resamplings "
              f"in {time.time() - t0:.0f} s")
    return pd.DataFrame(rows), out


def station_summary(summ: pd.DataFrame,
                    pct: tuple[float, float] = config.CI_PERCENTILES
                    ) -> dict:
    """Headline bootstrap numbers of one station (for the README / report)."""
    lo, hi = (f"p{p:g}" for p in pct)
    last = summ.sort_values("window_end").iloc[-1]
    # Windows whose MEVD band does not overlap the last window's band.
    no_overlap = summ[(summ[f"mev_{hi}"] < last[f"mev_{lo}"])
                      | (summ[f"mev_{lo}"] > last[f"mev_{hi}"])]
    return {
        "station": summ["station"].iat[0],
        "n_windows": len(summ),
        "gev_rel_width_median": summ["gev_rel_width"].median(),
        "mev_rel_width_median": summ["mev_rel_width"].median(),
        "width_ratio_gev_to_mev_median": (summ["gev_rel_width"]
                                          / summ["mev_rel_width"]).median(),
        "gev_rel_width_min": summ["gev_rel_width"].min(),
        "window_gev_rel_width_min": int(
            summ.loc[summ["gev_rel_width"].idxmin(), "window_end"]),
        f"gev_{hi}_max_mm": summ[f"gev_{hi}"].max(),
        f"mev_{hi}_max_mm": summ[f"mev_{hi}"].max(),
        "n_windows_gev_above_mev": int((summ[f"diff_{lo}"] > 0).sum()),
        "n_windows_mev_above_gev": int((summ[f"diff_{hi}"] < 0).sum()),
        "max_share_lower_bound": summ["gev_share_lower_bound"].max(),
        "max_share_upper_bound": summ["gev_share_upper_bound"].max(),
        "n_windows_upper_bound_ge_5pct": int(
            (summ["gev_share_upper_bound"] >= 0.05).sum()),
        "last_window_end": int(last["window_end"]),
        f"last_mev_{lo}": last[f"mev_{lo}"], "last_mev_rl": last["mev_rl"],
        f"last_mev_{hi}": last[f"mev_{hi}"],
        f"last_gev_{lo}": last[f"gev_{lo}"], "last_gev_rl": last["gev_rl"],
        f"last_gev_{hi}": last[f"gev_{hi}"],
        "n_windows_mev_band_disjoint_from_last": len(no_overlap),
    }


def largest_year_effect(ws: WindowSample, rl_gev: np.ndarray,
                        rl_mev: np.ndarray,
                        seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Bootstrap estimates grouped by how often the window's wettest year was drawn.

    In a resample of n years drawn with replacement, a given year appears
    k ~ Binomial(n, 1/n) times: it is MISSING from about (1 - 1/n)^n ~ 36%
    of the resamples. When one year dominates the GEV fit (e.g. Bologna
    1933), the bootstrap distribution of the GEV is a mixture of these
    groups, which is why it can be bimodal (Fig. 18a).
    """
    B = rl_gev.size
    idx = resample_indices(ws.n, B, ws.station, ws.end, seed)
    j = int(np.argmax(ws.amax))
    k = np.minimum((idx == j).sum(axis=1), 3)          # 0, 1, 2, 3+ copies
    rows = []
    for kk in range(4):
        sel = k == kk
        if not sel.any():
            continue
        rows.append({
            "station": ws.station, "window_end": ws.end,
            "wettest_year": int(ws.years[j]), "wettest_max_mm": ws.amax[j],
            "copies": "3+" if kk == 3 else str(kk), "n_resamples": int(sel.sum()),
            "gev_median": np.median(rl_gev[sel]), "mev_median": np.median(rl_mev[sel]),
            "gev_p5": np.percentile(rl_gev[sel], 5),
            "gev_p95": np.percentile(rl_gev[sel], 95),
        })
    return pd.DataFrame(rows)


def check_batch_fitter(samples: list[WindowSample], n_windows: int = 40,
                       per_window: int = 3, B: int = config.N_BOOTSTRAP,
                       seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Compare the fast batched GEV fitter with the reference ``gev.fit_ml``.

    Takes ``per_window`` bootstrap replicates from ``n_windows`` evenly spaced
    windows (the same replicates the bootstrap uses) and refits them with the
    reference Nelder-Mead fitter.
    """
    pick = np.unique(np.linspace(0, len(samples) - 1, n_windows).astype(int))
    rows = []
    for i in pick:
        ws = samples[i]
        idx = resample_indices(ws.n, B, ws.station, ws.end, seed)
        p0 = gev.fit_ml(ws.amax)
        sub = idx[:per_window]
        fit = gev_batch.fit_ml(ws.amax[sub], start=p0.as_tuple())
        rl_fast = fit.return_level(config.RETURN_PERIOD)
        for b in range(per_window):
            ref = gev.fit_ml(ws.amax[sub[b]])
            rows.append({
                "station": ws.station, "window_end": ws.end, "replicate": b,
                "xi_ref": ref.xi, "xi_fast": fit.xi[b],
                "nll_ref": ref.nll, "nll_fast": fit.nll[b],
                "rl_ref": float(gev.return_level(config.RETURN_PERIOD, ref)),
                "rl_fast": rl_fast[b],
                "ref_in_bounds": gev_batch.XI_MIN < ref.xi < gev_batch.XI_MAX,
            })
    df = pd.DataFrame(rows)
    df["abs_diff_rl_mm"] = (df["rl_fast"] - df["rl_ref"]).abs()
    df["diff_nll"] = df["nll_fast"] - df["nll_ref"]
    return df

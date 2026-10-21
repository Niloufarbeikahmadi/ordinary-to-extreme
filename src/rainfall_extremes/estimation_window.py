"""
MEVD estimation window: how many years share one Weibull fit (Task 5).

Background
----------
The main MEVD fits one Weibull per year (estimation window EW = 1 year, as
in mevpy and Zorzetto et al., 2016). Task 4 showed that, with ~70 wet days
per year, the yearly shape w is noisy, and that this noise biases the
200-year MEVD upward (a heavy-tailed-by-chance year dominates the tail).
Pooling the wet days of several years into one fit reduces the noise but
also smooths away real year-to-year differences. Miniussi & Marani (2020,
WRR) studied this trade-off and found EW = 1 year best for most stations.
Here the choice is tested on the two records themselves.

Variants (all keep each year's own N_j)
---------------------------------------
* EW = 1, 3, 5 years: the valid years of a window are split into
  consecutive blocks of EW years (a short remainder joins the last block);
  one PWM Weibull is fitted to the excesses of all wet days of a block.
* EW = all ("pooled"): one Weibull for the whole window, used with every
  year's N_j:  zeta(x) = (1/n) sum_j F(x)^N_j.
* SMEV (Marra et al., 2019): one Weibull for the whole window and the mean
  number of wet days:  zeta(x) = F(x)^mean(N).

For the pooled variant the module also provides the common bootstrap (same
resampled years as Task 4) and the leave-one-year-out check (as Task 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn
from scipy.stats import binomtest, kstest

from . import config, mev
from .bootstrap import resample_indices
from .validation import LOYO_HALF_WINDOW, LOYO_RETURN_PERIODS
from .windows import Window

EW_VARIANTS: tuple = (1, 3, 5, "all", "smev")


def excess_by_year(daily: pd.DataFrame, value_col: str = "precip_mm",
                   threshold: float = config.WET_THRESHOLD_MM
                   ) -> dict[int, np.ndarray]:
    """Sorted excesses P - threshold of the wet days (P > threshold) of each year."""
    out = {}
    for year, grp in daily.groupby("year", sort=True):
        p = grp[value_col].to_numpy(dtype=float)
        p = p[np.isfinite(p)]
        out[int(year)] = np.sort(p[p > threshold] - threshold)
    return out


def blocks(years: list[int], ew) -> list[list[int]]:
    """Consecutive blocks of ``ew`` valid years; a short remainder joins the last."""
    years = list(years)
    if ew in ("all", "smev"):
        return [years]
    k = len(years) // ew
    out = [years[i * ew:(i + 1) * ew] for i in range(k)]
    rest = years[k * ew:]
    if rest:
        if out:
            out[-1] = out[-1] + rest
        else:
            out = [rest]
    return out


def window_parameters(years, exc: dict[int, np.ndarray], ew
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N_j, C_j, w_j) of the window's years under estimation window ``ew``.

    For SMEV a single triplet (mean N, C, w) is returned.
    """
    N, C, w = [], [], []
    for b in blocks(years, ew):
        fit = mev.weibull_fit_pwm(np.concatenate([exc[y] for y in b]))
        for y in b:
            N.append(exc[y].size)
            C.append(fit.C)
            w.append(fit.w)
    N, C, w = (np.asarray(a, dtype=float) for a in (N, C, w))
    if ew == "smev":
        return np.array([N.mean()]), C[:1], w[:1]
    return N, C, w


def ew_over_windows(exc: dict[int, np.ndarray], windows: list[Window],
                    variants=EW_VARIANTS, T: float = config.RETURN_PERIOD
                    ) -> pd.DataFrame:
    """200-year MEVD level in every window for every estimation window."""
    rows = []
    for win in windows:
        for ew in variants:
            N, C, w = window_parameters(win.years, exc, ew)
            rows.append({"window_start": win.start, "window_end": win.end,
                         "n_valid": win.n_valid, "ew": str(ew),
                         "mean_C": C.mean(), "mean_w": w.mean(),
                         "sd_w": w.std(),
                         f"rl_{int(T)}": mev.mev_return_level(T, N, C, w)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Pooled Weibull for resampled years (bootstrap)
# ----------------------------------------------------------------------
def pooled_pwm_weighted(x_sorted: np.ndarray, labels: np.ndarray,
                        counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PWM Weibull of the pooled wet days of many resampled sets of years.

    ``x_sorted`` (K,) holds the excesses of all wet days of the window in
    ascending order, ``labels`` (K,) the index of the year each comes from,
    and ``counts`` (B, n) how many times each year was drawn in each of the
    B resamples. A value drawn c times occupies c consecutive ranks a+1..a+c
    of the pooled sample (a = copies of smaller values), so its contribution
    to the PWM sum  sum_i x_(i) (m - i)  is  x [c (m - a) - c (c + 1)/2],
    with m the pooled sample size. This gives exactly the PWM of the
    concatenated resampled data without building it. Returns (C, w), (B,).
    """
    c = counts[:, labels].astype(float)                     # (B, K)
    m = c.sum(axis=1)
    a = np.cumsum(c, axis=1) - c
    M0 = (c * x_sorted).sum(axis=1) / m
    M1 = (x_sorted * (c * (m[:, None] - a) - c * (c + 1) / 2)).sum(axis=1)
    M1 = M1 / (m * (m - 1))
    C = M0 / gamma_fn(np.log(M0 / M1) / np.log(2.0))
    w = np.log(2.0) / np.log(M0 / (2.0 * M1))
    return C, w


def counts_from_indices(idx: np.ndarray, n: int) -> np.ndarray:
    """(B, n) number of times each year appears in each resample."""
    B = idx.shape[0]
    counts = np.zeros((B, n))
    np.add.at(counts, (np.repeat(np.arange(B), idx.shape[1]), idx.ravel()), 1)
    return counts


def pooled_window_data(years, exc: dict[int, np.ndarray]):
    """Sorted pooled excesses of a window with the year label of each value."""
    vals = [exc[y] for y in years]
    x = np.concatenate(vals)
    labels = np.concatenate([np.full(v.size, j) for j, v in enumerate(vals)])
    order = np.argsort(x, kind="stable")
    N = np.array([v.size for v in vals], dtype=float)
    return x[order], labels[order], N


def bootstrap_pooled(station: str, windows: list[Window],
                     exc: dict[int, np.ndarray], B: int = config.N_BOOTSTRAP,
                     T: float = config.RETURN_PERIOD,
                     seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, dict]:
    """Common bootstrap of the pooled-Weibull MEVD (same draws as Task 4)."""
    F = 1.0 - 1.0 / T
    lo, hi = config.CI_PERCENTILES
    rows, reps = [], []
    for win in windows:
        xs, lab, N = pooled_window_data(win.years, exc)
        n = N.size
        fit = mev.weibull_fit_pwm(xs)
        point = mev.mev_quantile(F, N, np.full(n, fit.C), np.full(n, fit.w))
        idx = resample_indices(n, B, station, win.end, seed)
        C, w = pooled_pwm_weighted(xs, lab, counts_from_indices(idx, n))
        rl = mev.mev_quantile_batch(F, N[idx], np.repeat(C[:, None], n, 1),
                                    np.repeat(w[:, None], n, 1))
        q_lo, q_hi = np.percentile(rl, [lo, hi])
        rows.append({"station": station, "window_start": win.start,
                     "window_end": win.end, "n_valid": n, "C": fit.C,
                     "w": fit.w, "mevp_rl": point, f"mevp_p{lo:g}": q_lo,
                     f"mevp_p{hi:g}": q_hi,
                     "mevp_rel_width": (q_hi - q_lo) / point})
        reps.append(rl)
    return pd.DataFrame(rows), {"rl_mevp": np.vstack(reps),
                                "window_end": np.array([w.end for w in windows])}


# ----------------------------------------------------------------------
# Leave-one-year-out check of the pooled variant
# ----------------------------------------------------------------------
def loyo_ew(annual: pd.DataFrame, exc: dict[int, np.ndarray], ew="all",
            half: int = LOYO_HALF_WINDOW,
            return_periods=LOYO_RETURN_PERIODS) -> pd.DataFrame:
    """As ``validation.loyo_table`` (same years, same rule) for any EW.

    The fitting years of each test are the other valid years of the window
    centred on year j; for EW > 1 they are split into blocks as above.
    """
    av = annual.loc[annual["valid"]].set_index("year")
    years = av.index.to_numpy()
    rows = []
    for j in years:
        others = [y for y in years if j - half <= y <= j + half - 1 and y != j]
        if len(others) < config.MIN_VALID_YEARS:
            continue
        N, C, w = window_parameters(others, exc, ew)
        mj = float(av.at[j, "annual_max_mm"])
        row = {"year": int(j), "ew": str(ew), "annual_max_mm": mj,
               "u": float(mev.mev_cdf(mj, N, C, w)[0])}
        for T in return_periods:
            row[f"exceed_{T}"] = bool(mj > mev.mev_return_level(T, N, C, w))
        rows.append(row)
    return pd.DataFrame(rows)


def loyo_ew_summary(table: pd.DataFrame,
                    return_periods=LOYO_RETURN_PERIODS) -> dict:
    """KS uniformity and binomial exceedance tests (as ``loyo_summary``)."""
    n = len(table)
    row = {"model": f"MEVD EW={table['ew'].iat[0]}", "ew": table["ew"].iat[0],
           "n_years": n, "ks_pvalue": kstest(table["u"], "uniform").pvalue}
    for T in return_periods:
        k = int(table[f"exceed_{T}"].sum())
        row[f"exceed_{T}_observed"] = k
        row[f"exceed_{T}_expected"] = n / T
        row[f"exceed_{T}_binom_p"] = binomtest(k, n, 1 / T).pvalue
    return row

"""
Metastatistical Extreme Value Distribution with Weibull ordinary events
(MEVD-Weibull; Marani & Ignaccolo, 2015; Zorzetto, Botter & Marani, 2016).

Idea
----
Instead of modelling the annual maxima directly, model the ORDINARY events,
i.e. every wet day, year by year. If year j has N_j wet days whose amounts
are independent draws from a Weibull distribution F_j(x) = 1 - exp[-(x/C_j)^w_j],
the probability that its maximum stays below x is F_j(x)^N_j (exact for any
N_j; no asymptotic limit is needed). Averaging over the T years of a window
gives the MEVD:

    zeta(x) = (1/T) * sum_j [ 1 - exp(-((x - theta)/C_j)^w_j) ]^N_j

The T-year return level solves zeta(x) = 1 - 1/T_R.

Assumptions (stated explicitly, to be checked)
----------------------------------------------
1. Within a year, wet-day amounts are independent and identically Weibull
   distributed. The same F_j holds for every wet day of year j, so
   seasonality is absorbed into a single yearly distribution.
2. Year-to-year variability is represented by the sample of yearly
   (N_j, C_j, w_j) in the window (sample-average approximation of the
   distribution g(N, C, w)).
3. N_j counts the days with P > theta (theta = 1 mm). Ordinary events are
   modelled as EXCESSES over theta (mevpy ``potmode=True``), and theta is
   added back to every return level, as the task specifies.

Estimators
----------
* ``pwm`` (main): probability weighted moments (Greenwood et al., 1979), as
  in mevpy and Zorzetto et al. (2016). PWM gives more weight to the tail
  than ML and behaves well in small samples.
* ``ml``  (sensitivity): maximum likelihood.
* ``ml_censored`` (sensitivity, Task 5): left-censored maximum likelihood.
  Values below a censoring depth only count as "below the depth", so the
  fit is driven by the upper part of the distribution.
* ``pwm_censored``: a port of mevpy ``wei_fit_pwm_cens``, kept for
  verification only. It inserts partial PWMs into the uncensored formulas
  and is therefore biased (see its docstring); it is not used for results.

Formulations
------------
* ``excess`` (main): fit the Weibull to (P - theta) for P > theta, and
  return quantile + theta. This is mevpy ``potmode=True``, and it is what the
  personal-email version of the task spells out explicitly.
* ``raw`` (variant): fit the Weibull to the wet-day values P > theta
  themselves, with no shift and nothing added back (mevpy ``potmode=False``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import gamma as gamma_fn

from . import config

MIN_WET_DAYS_FOR_FIT: int = 10   # fewer events -> yearly fit not attempted


# ----------------------------------------------------------------------
# Weibull distribution and estimators
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class WeibullFit:
    n: int          # number of values used (wet days)
    C: float        # scale [mm]
    w: float        # shape [-]


def weibull_cdf(x, C, w):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return 1.0 - np.exp(-(x / C) ** w)


def weibull_quantile(F, C, w):
    return C * (-np.log(1.0 - np.asarray(F, dtype=float))) ** (1.0 / w)


def weibull_fit_pwm(x) -> WeibullFit:
    """Weibull by probability weighted moments (Greenwood et al., 1979).

    Uses M0 = E[X] and M1 = E[X (1 - F(X))], estimated from the ascending
    order statistics as  M1 = sum_i x_(i) (n - i) / (n (n - 1)). Then

        w = ln 2 / ln(M0 / (2 M1)),   C = M0 / Gamma(1 + 1/w)

    (written in mevpy as C = M0 / Gamma(ln(M0/M1)/ln2), which is identical
    because ln(M0/M1)/ln2 = 1 + 1/w).
    """
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x > 0]
    n = x.size
    if n < 2:
        return WeibullFit(n, np.nan, np.nan)
    i = np.arange(1, n + 1)
    M0 = x.mean()
    M1 = np.sum(x * (n - i)) / (n * (n - 1))
    C = M0 / gamma_fn(np.log(M0 / M1) / np.log(2.0))
    w = np.log(2.0) / np.log(M0 / (2.0 * M1))
    return WeibullFit(n, float(C), float(w))


def weibull_fit_pwm_censored(x, censor: float) -> WeibullFit:
    """Port of ``mevpy.wei_fit_pwm_cens`` (kept for verification only).

    All positive values define the ranks, but only values above ``censor``
    contribute to the moments. The resulting PARTIAL moments are then
    inserted into the formulas for the UNCENSORED Weibull, which do not hold
    for partial moments (Wang, 1990, derives separate relations). The fit is
    therefore biased: for a large Weibull(C = 8, w = 0.8) sample censored at
    its median it returns C = 5.4, w = 0.60. Use ``weibull_fit_ml_censored``.
    """
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x > 0]
    n = x.size
    if n < 2:
        return WeibullFit(n, np.nan, np.nan)
    i = np.arange(1, n + 1)
    keep = x > censor
    b0 = np.sum(x[keep]) / n
    b1 = np.sum(x[keep] * (i[keep] - 1)) / (n * (n - 1))
    M0, M1 = b0, b0 - b1
    C = M0 / gamma_fn(np.log(M0 / M1) / np.log(2.0))
    w = np.log(2.0) / np.log(M0 / (2.0 * M1))
    return WeibullFit(n, float(C), float(w))


def weibull_fit_ml(x) -> WeibullFit:
    """Weibull by maximum likelihood (profile score equation in w).

    d lnL / dw = 0  <=>  1/w + mean(ln x) - sum(x^w ln x) / sum(x^w) = 0,
    which has a unique root; then C = (mean(x^w))^(1/w).
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    n = x.size
    if n < 2:
        return WeibullFit(n, np.nan, np.nan)
    lx = np.log(x)
    xs = x / x.max()                 # rescale for numerical stability

    def score(w):
        xw = xs ** w
        return 1.0 / w + lx.mean() - np.sum(xw * lx) / np.sum(xw)

    w = brentq(score, 1e-3, 50.0, xtol=1e-12)
    C = float(np.mean(x ** w) ** (1.0 / w))
    return WeibullFit(n, C, float(w))


def weibull_fit_ml_censored(x, censor: float) -> WeibullFit:
    """Weibull by maximum likelihood with left censoring at ``censor``.

    The n_c values <= censor contribute only P(X <= censor); the others
    contribute their density:

        ln L = n_c ln F(c) + sum_{x > c} [ln w - ln C + (w - 1) ln(x/C) - (x/C)^w]

    Maximised over (ln C, ln w) from the uncensored ML estimate.
    """
    from scipy.optimize import minimize
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    n = x.size
    up = x[x > censor]
    if up.size < 5:
        return WeibullFit(n, np.nan, np.nan)
    n_c = n - up.size
    lu = np.log(up)
    start = weibull_fit_ml(x)

    def nll(par):
        lC, lw = par
        C, w = np.exp(lC), np.exp(lw)
        zc = (censor / C) ** w if censor > 0 else 0.0
        below = n_c * np.log(-np.expm1(-zc)) if n_c > 0 else 0.0
        z = np.exp(w * (lu - lC))
        return -(below + np.sum(lw - lC + (w - 1) * (lu - lC) - z))

    res = minimize(nll, [np.log(start.C), np.log(start.w)],
                   method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
    return WeibullFit(n, float(np.exp(res.x[0])), float(np.exp(res.x[1])))


def fit_weibull(x, method: str = "pwm", censor: float | None = None
                ) -> WeibullFit:
    if method == "pwm":
        return weibull_fit_pwm(x)
    if method == "ml":
        return weibull_fit_ml(x)
    if method == "ml_censored":
        return weibull_fit_ml_censored(x, censor or 0.0)
    if method == "pwm_censored":
        return weibull_fit_pwm_censored(x, censor or 0.0)
    raise ValueError(f"Unknown Weibull method {method!r}.")


# ----------------------------------------------------------------------
# Yearly ordinary-event parameters
# ----------------------------------------------------------------------
def yearly_parameters(daily: pd.DataFrame, value_col: str = "precip_mm",
                      threshold: float = config.WET_THRESHOLD_MM,
                      formulation: str = "excess", method: str = "pwm",
                      censor: float | None = None) -> pd.DataFrame:
    """(N_j, C_j, w_j) for every year with observations.

    N_j = number of days with P > threshold. The Weibull is fitted to the
    excesses P - threshold ('excess') or to the values P ('raw').
    """
    rows = []
    for year, grp in daily.groupby("year", sort=True):
        p = grp[value_col].to_numpy(dtype=float)
        p = p[np.isfinite(p)]
        if p.size == 0:
            continue
        wet = p[p > threshold]
        sample = wet - threshold if formulation == "excess" else wet
        if wet.size >= MIN_WET_DAYS_FOR_FIT:
            fit = fit_weibull(sample, method, censor)
        else:
            fit = WeibullFit(wet.size, np.nan, np.nan)
        rows.append({"year": int(year), "N": int(wet.size), "C": fit.C,
                     "w": fit.w, "annual_max_mm": float(p.max())})
    out = pd.DataFrame(rows)
    out["formulation"] = formulation
    out["method"] = method
    return out


# ----------------------------------------------------------------------
# MEV distribution
# ----------------------------------------------------------------------
def _shift(formulation: str, threshold: float) -> float:
    if formulation == "excess":
        return threshold
    if formulation == "raw":
        return 0.0
    raise ValueError(f"Unknown formulation {formulation!r}.")


def mev_cdf(x, N, C, w, threshold: float = config.WET_THRESHOLD_MM,
            formulation: str = "excess") -> np.ndarray:
    """MEVD non-exceedance probability zeta(x) for yearly arrays N, C, w."""
    N, C, w = (np.asarray(a, dtype=float) for a in (N, C, w))
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.maximum(x - _shift(formulation, threshold), 0.0)[:, None]
    Fj = 1.0 - np.exp(-(y / C[None, :]) ** w[None, :])
    with np.errstate(divide="ignore"):
        # F^N computed in log space for accuracy when F -> 1 and N ~ 100
        logH = N[None, :] * np.log1p(-np.exp(-(y / C[None, :]) ** w[None, :]))
    H = np.where(Fj > 0, np.exp(logH), 0.0)
    return H.mean(axis=1)


def mev_quantile(F: float, N, C, w,
                 threshold: float = config.WET_THRESHOLD_MM,
                 formulation: str = "excess") -> float:
    """Depth x with zeta(x) = F, by bracketed root finding.

    zeta is continuous and increasing, so a bracket [lo, hi] is built by
    doubling ``hi`` until zeta(hi) > F, and the root is found with Brent's
    method. This replaces the unbracketed ``fsolve`` of mevpy, whose result
    depends on the starting guess.
    """
    shift = _shift(formulation, threshold)

    def f(x):
        return float(mev_cdf(x, N, C, w, threshold, formulation)[0]) - F

    lo, hi = shift, shift + 50.0
    while f(hi) < 0:
        lo, hi = hi, hi * 2.0
        if hi > 1e6:
            return np.nan
    return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12))


def mev_return_level(T: float, N, C, w,
                     threshold: float = config.WET_THRESHOLD_MM,
                     formulation: str = "excess") -> float:
    """T-year MEVD return level (the threshold is already added back)."""
    return mev_quantile(1.0 - 1.0 / T, N, C, w, threshold, formulation)


def mev_quantile_batch(F: float, N, C, w,
                       threshold: float = config.WET_THRESHOLD_MM,
                       formulation: str = "excess",
                       tol: float = 1e-9) -> np.ndarray:
    """MEVD quantile for many windows at once (rows of (B, n) arrays).

    Used by the bootstrap, where each row is one resampled set of years.

    Bracket: zeta(x) is the AVERAGE of the yearly probabilities
    H_j(x) = F_j(x)^N_j. So at the smallest yearly quantile every H_j <= F,
    and at the largest every H_j >= F, and the root lies between them. The
    yearly quantiles are analytic,

        x_j = C_j * [-ln(1 - F^(1/N_j))]^(1/w_j),

    with 1 - F^(1/N_j) computed as -expm1(ln F / N_j) to keep precision
    (F^(1/N) is within 1e-4 of 1). The bracket is then halved (bisection)
    until it is narrower than ``tol`` mm: about 40 steps, each one array
    operation over all rows.
    """
    N, C, w = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (N, C, w))
    one_minus = -np.expm1(np.log(F) / N)
    xj = C * (-np.log(one_minus)) ** (1.0 / w)
    lo, hi = xj.min(axis=1), xj.max(axis=1)

    def zeta(y):
        with np.errstate(divide="ignore", over="ignore"):
            logH = N * np.log1p(-np.exp(-(y[:, None] / C) ** w))
        return np.exp(logH).mean(axis=1)

    while np.max(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        below = zeta(mid) < F
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi) + _shift(formulation, threshold)


def pit_annual_maxima(params: pd.DataFrame,
                      threshold: float = config.WET_THRESHOLD_MM) -> pd.Series:
    """Probability integral transform of each annual maximum under its own
    year's model: u_j = F_j(max_j)^N_j.

    If the yearly building block of the MEVD is right (the maximum of N_j
    independent Weibull draws), the u_j are Uniform(0, 1). The fit uses the
    same data as the maximum, so the test is mildly optimistic.
    """
    shift = threshold if (params["formulation"] == "excess").all() else 0.0
    y = np.maximum(params["annual_max_mm"] - shift, 0.0)
    F = weibull_cdf(y, params["C"], params["w"])
    return pd.Series(F ** params["N"], index=params.index, name="pit")

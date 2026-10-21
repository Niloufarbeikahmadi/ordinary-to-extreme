"""
Generalized Extreme Value (GEV) distribution for annual maxima.

Model and assumptions
---------------------
Let ``M`` be the largest daily rainfall of a year. Classical extreme value
theory (Fisher-Tippett-Gnedenko) states that, IF the year contains n
independent and identically distributed daily values and n -> infinity,
then the suitably normalised maximum converges to the GEV:

    F(x) = exp{ -[1 + xi (x - mu) / sigma]^(-1/xi) },   1 + xi (x - mu)/sigma > 0

with location ``mu``, scale ``sigma > 0`` and shape ``xi``:

* xi > 0  Frechet type: heavy upper tail, no upper bound (common for rainfall)
* xi = 0  Gumbel type:  F(x) = exp(-exp(-(x - mu)/sigma))
* xi < 0  reversed-Weibull type: bounded upper tail

The block-maxima approach therefore assumes (i) one maximum per year,
(ii) maxima independent from year to year, (iii) maxima identically
distributed within the fitting window (stationarity), and (iv) that a year
holds "enough" events for the asymptotic limit to apply. Assumption (iv) is
exactly what the MEVD relaxes (Task 3).

Sign convention
---------------
``xi > 0`` means a heavy tail (Coles, 2001; ``csi`` in mevpy).
``scipy.stats.genextreme`` uses ``c = -xi``.

Return level
------------
The T-year return level is the quantile with annual non-exceedance
probability ``F = 1 - 1/T``:

    x_T = mu + sigma/xi * [ (-ln F)^(-xi) - 1 ]     (xi != 0)
    x_T = mu - sigma * ln(-ln F)                    (xi = 0)

Estimators
----------
* ``fit_ml``   - maximum likelihood (Marani & Zanetti, 2015): main estimator.
* ``fit_lmom`` - L-moments with Hosking's (1985) approximation for the
  shape (mevpy default; robust in small samples): sensitivity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn

_GUMBEL_TOL = 1e-7          # |xi| below this is treated as the Gumbel limit
_EULER = 0.5772156649015329


@dataclass(frozen=True)
class GEVParams:
    """GEV parameters (shape ``xi`` > 0 = heavy tail), plus fit diagnostics."""
    xi: float
    sigma: float
    mu: float
    method: str = ""
    n: int = 0
    nll: float = np.nan          # negative log-likelihood at the estimate
    converged: bool = True

    def as_tuple(self) -> tuple[float, float, float]:
        return self.xi, self.sigma, self.mu


# ----------------------------------------------------------------------
# Distribution functions
# ----------------------------------------------------------------------
def cdf(x, xi: float, sigma: float, mu: float) -> np.ndarray:
    """GEV non-exceedance probability F(x)."""
    z = (np.asarray(x, dtype=float) - mu) / sigma
    if abs(xi) < _GUMBEL_TOL:
        return np.exp(-np.exp(-z))
    t = 1.0 + xi * z
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.exp(-np.power(np.where(t > 0, t, np.nan), -1.0 / xi))
    # Outside the support: below the lower bound (xi>0) F=0, above the
    # upper bound (xi<0) F=1.
    return np.where(t > 0, out, 0.0 if xi > 0 else 1.0)


def quantile(F, xi: float, sigma: float, mu: float) -> np.ndarray:
    """GEV quantile for non-exceedance probability ``F``."""
    y = -np.log(np.asarray(F, dtype=float))
    if abs(xi) < _GUMBEL_TOL:
        return mu - sigma * np.log(y)
    return mu + sigma / xi * (np.power(y, -xi) - 1.0)


def return_level(T, params: GEVParams | tuple) -> np.ndarray:
    """T-year return level: quantile with F = 1 - 1/T."""
    xi, sigma, mu = params.as_tuple() if isinstance(params, GEVParams) else params
    return quantile(1.0 - 1.0 / np.asarray(T, dtype=float), xi, sigma, mu)


# ----------------------------------------------------------------------
# L-moments
# ----------------------------------------------------------------------
def sample_lmoments(x) -> tuple[float, float, float]:
    """First two sample L-moments and the L-skewness ``t3 = l3 / l2``.

    Computed from the unbiased probability-weighted moments b0, b1, b2
    (Hosking, 1990), with the sample sorted in ascending order.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n < 3:
        raise ValueError("At least 3 values are needed for L-moments.")
    j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    return l1, l2, l3 / l2


def fit_lmom(x) -> GEVParams:
    """GEV by L-moments (Hosking, Wallis & Wood, 1985 approximation).

    k = 7.8590 c + 2.9554 c^2 with c = 2/(3 + t3) - ln2/ln3, and xi = -k.
    This is exactly the formulation of ``mevpy.gev_fun.gev_fit_lmom``.
    The approximation is accurate to about 1e-4 in k for -0.5 < t3 < 0.5.
    """
    x = np.asarray(x, dtype=float)
    l1, l2, t3 = sample_lmoments(x)
    c = 2.0 / (3.0 + t3) - np.log(2.0) / np.log(3.0)
    k = 7.8590 * c + 2.9554 * c ** 2
    if abs(k) < _GUMBEL_TOL:                       # Gumbel limit
        sigma = l2 / np.log(2.0)
        mu = l1 - _EULER * sigma
    else:
        sigma = l2 * k / ((1.0 - 2.0 ** (-k)) * gamma_fn(1.0 + k))
        mu = l1 - sigma * (1.0 - gamma_fn(1.0 + k)) / k
    xi = -k
    return GEVParams(xi, sigma, mu, method="lmom", n=x.size,
                     nll=negloglik((xi, sigma, mu), x))


# ----------------------------------------------------------------------
# Maximum likelihood
# ----------------------------------------------------------------------
def negloglik(theta, x) -> float:
    """GEV negative log-likelihood; +inf outside the parameter space."""
    xi, sigma, mu = theta
    x = np.asarray(x, dtype=float)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.inf
    z = (x - mu) / sigma
    if abs(xi) < _GUMBEL_TOL:
        return float(x.size * np.log(sigma) + np.sum(z) + np.sum(np.exp(-z)))
    t = 1.0 + xi * z
    if np.any(t <= 0):
        return np.inf
    return float(x.size * np.log(sigma) + (1.0 + 1.0 / xi) * np.sum(np.log(t))
                 + np.sum(t ** (-1.0 / xi)))


def fit_ml(x, starts: list[tuple[float, float, float]] | None = None
           ) -> GEVParams:
    """GEV by maximum likelihood.

    The likelihood is maximised with Nelder-Mead (as in mevpy) over
    (xi, log sigma, mu); the log keeps sigma positive. Because the GEV
    likelihood can be flat or multi-modal in samples of 20-30 maxima, the
    optimiser is started from three points: the L-moment estimate, the
    moment-based Gumbel start with xi = 0.1 used by mevpy, and a pure Gumbel
    start. The solution with the lowest negative log-likelihood is kept.

    Note: for xi < -1 the GEV likelihood is unbounded (Smith, 1985) and the
    ML estimator is not regular for xi <= -0.5. Such fits are reported with
    ``converged=False`` rather than silently accepted.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if starts is None:
        lm = fit_lmom(x)
        s_mom = np.sqrt(6.0 * np.var(x)) / np.pi
        m_mom = np.mean(x) - _EULER * s_mom
        starts = [lm.as_tuple(), (0.1, s_mom, m_mom), (0.0, s_mom, m_mom)]

    def objective(p):
        return negloglik((p[0], np.exp(p[1]), p[2]), x)

    best = None
    for xi0, s0, m0 in starts:
        if not (np.isfinite(s0) and s0 > 0):
            continue
        p0 = np.array([xi0, np.log(s0), m0])
        if not np.isfinite(objective(p0)):        # start outside support
            p0 = np.array([0.0, np.log(s0), m0])
        res = minimize(objective, p0, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-10,
                                "maxiter": 20000, "maxfev": 40000})
        if best is None or res.fun < best.fun:
            best = res
    xi, sigma, mu = best.x[0], float(np.exp(best.x[1])), best.x[2]
    ok = bool(best.success and np.isfinite(best.fun) and xi > -0.5)
    return GEVParams(float(xi), sigma, float(mu), method="ml", n=x.size,
                     nll=float(best.fun), converged=ok)


FITTERS = {"ml": fit_ml, "lmom": fit_lmom}


def fit(x, method: str = "ml") -> GEVParams:
    """Fit the GEV with ``method`` in {'ml', 'lmom'}."""
    try:
        return FITTERS[method](x)
    except KeyError as err:
        raise ValueError(f"Unknown GEV method {method!r}.") from err


def weibull_plotting_positions(n: int) -> np.ndarray:
    """Empirical non-exceedance probabilities i/(n+1) for sorted maxima."""
    return np.arange(1, n + 1) / (n + 1.0)

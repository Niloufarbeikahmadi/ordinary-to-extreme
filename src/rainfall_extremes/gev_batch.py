"""
Vectorised GEV maximum likelihood for many samples at once (Task 4).

Why this module exists
----------------------
The common bootstrap needs one GEV fit per window and per resampling:
427 windows x 500 resamplings = 213 500 fits for the two stations. The
reference fitter ``gev.fit_ml`` (three Nelder-Mead runs per sample) takes
20-70 ms per fit, i.e. one to four hours. Here all B samples of a window are
fitted *simultaneously* with numpy array operations: about 0.1 s per window,
about one minute for everything.

Method
------
* Parameters: theta = (mu, phi = log sigma, xi), as in ``gev.fit_ml``.
* Objective: the GEV negative log-likelihood (nll), +inf outside the support.
* Gradient and Hessian: analytic (formulas in ``_grad`` and ``_grad_hess``).
* Optimiser: Levenberg-Marquardt, run independently for every sample (each
  sample has its own damping ``lambda``). A step is accepted only if it
  lowers the nll; a rejected step increases ``lambda``, which shortens the
  step towards gradient descent. Every sample therefore improves
  monotonically.
* Starts: (1) a common start given by the caller (in the bootstrap: the ML
  estimate of the original window, which supports every resampled value)
  and (2) each sample's own L-moment estimate: the same multi-start idea as
  ``gev.fit_ml``.

Shape bounds and the "interior first" rule
------------------------------------------
The GEV likelihood has two degenerate directions in which it grows without
limit, so the unconstrained maximum does not exist there:

* xi < -1: the density diverges as the UPPER endpoint approaches the sample
  maximum (Smith, 1985).
* xi -> +infinity with the LOWER endpoint approaching the sample minimum.
  Bootstrap resamples often repeat the smallest maxima several times, which
  makes this path easy for an optimiser to fall into (the unconstrained
  reference fitter returned xi = 8.3 for one such resample).

The shape is therefore restricted to [-1, 1]. xi > 1 would imply an infinite
mean annual maximum, far outside anything reported for daily rainfall
(typical xi ~ 0.1), so the bound only removes degenerate fits. Among the
solutions from the different starts, an INTERIOR stationary point (a genuine
local maximum of the likelihood) is preferred to a solution on a bound,
whatever their nll; a bound solution is used only when no interior maximum
is found, and it is flagged ``at_boundary``. Fits with xi <= -0.5 are
flagged ``irregular`` (ML is not regular there; Smith, 1985).

For every original 30-year window all fits are interior, and the results
coincide with ``gev.fit_ml``; agreement on resamples is checked in
``scripts/04_bootstrap.py`` and in the unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as gamma_fn

XI_MIN: float = -1.0           # lower shape bound (likelihood unbounded below)
XI_MAX: float = 1.0            # upper shape bound (degenerate fits above)
_XI_EPS: float = 1e-6          # |xi| is kept >= this (see _safe_xi)
#: A fit creeping towards a bound (the likelihood still rising there) stops
#: within ~1e-4 of it, so "on a bound" means within this distance.
_BOUND_TOL: float = 1e-3
_EULER = 0.5772156649015329


@dataclass
class BatchFit:
    """GEV fits of B samples (arrays of length B)."""
    xi: np.ndarray
    sigma: np.ndarray
    mu: np.ndarray
    nll: np.ndarray
    converged: np.ndarray       # gradient tolerance met (or no further progress)
    at_boundary: np.ndarray     # xi ended on XI_MIN or XI_MAX (degenerate)
    n_iter: np.ndarray

    @property
    def irregular(self) -> np.ndarray:
        """ML is not regular for xi <= -0.5 (Smith, 1985)."""
        return self.xi <= -0.5

    def return_level(self, T: float) -> np.ndarray:
        return return_level(T, self.xi, self.sigma, self.mu)


def _safe_xi(xi: np.ndarray) -> np.ndarray:
    """Keep |xi| >= 1e-6 so that 1/xi is finite.

    The likelihood is smooth through xi = 0 (the Gumbel limit), so this
    changes the nll by less than 1e-6: far below any meaningful precision.
    """
    return np.where(np.abs(xi) < _XI_EPS, np.copysign(_XI_EPS, xi + 0.0), xi)


def _terms(theta: np.ndarray, x: np.ndarray):
    """Common quantities for one batch: sigma, z, t, log t, t^(-1/xi)."""
    mu, phi, xi = (theta[:, k:k + 1] for k in range(3))
    xi = _safe_xi(xi)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        sigma = np.exp(phi)
        z = (x - mu) / sigma
        xz = xi * z
        valid = (np.all(xz > -1.0, axis=1) & np.isfinite(sigma[:, 0])
                 & (sigma[:, 0] > 0))
        L = np.log1p(xz)                   # log t, accurate for small xi z
        w = np.exp(-L / xi)                # t^(-1/xi)
    return phi, xi, sigma, z, 1.0 + xz, L, w, valid


def negloglik(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Negative log-likelihood of each row of ``x`` (B, n) at ``theta`` (B, 3)."""
    phi, xi, _, _, _, L, w, valid = _terms(theta, x)
    n = x.shape[1]
    with np.errstate(invalid="ignore", over="ignore"):
        f = n * phi[:, 0] + np.sum((1.0 + 1.0 / xi) * L + w, axis=1)
    return np.where(valid & np.isfinite(f), f, np.inf)


def _grad(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Analytic gradient of the negative log-likelihood, shape (B, 3).

    With z = (x - mu)/sigma, t = 1 + xi z, L = ln t and w = t^(-1/xi), one
    value contributes  l = ln sigma + (1 + 1/xi) L + w, and

        dl/dmu  = -(1 + xi - w) / (sigma t)
        dl/dphi = 1 - (1 + xi - w) z / t                  (phi = ln sigma)
        dl/dxi  = -(1 - w) L / xi^2 + (1 + xi - w) z / (xi t)
    """
    _, xi, sigma, z, t, L, w, valid = _terms(theta, x)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        a = (1.0 + xi - w) / t
        g = np.stack([
            np.sum(-a / sigma, axis=1),
            np.sum(1.0 - a * z, axis=1),
            np.sum(-(1.0 - w) * L / xi ** 2 + a * z / xi, axis=1),
        ], axis=1)
    g[~valid] = np.nan
    return g


def _grad_hess(theta: np.ndarray, x: np.ndarray):
    """Analytic gradient (B, 3) and Hessian (B, 3, 3) in one pass.

    Notation as in ``_grad``, plus a = (1 + xi - w)/t and
    K = xi (1 + xi - w) - w. Derivatives of the building blocks:

        dw/dmu = w/(sigma t),  dw/dphi = w z/t,  dw/dxi = w (L/xi^2 - z/(xi t))
        da/dmu = K/(sigma t^2),  da/dphi = z K/t^2,
        da/dxi = (1 - dw/dxi)/t - (1 + xi - w) z/t^2

    which give, per value,

        H_mu,mu   = -K/(sigma t)^2
        H_mu,phi  = -z K/(sigma t^2) + a/sigma
        H_mu,xi   = -(da/dxi)/sigma
        H_phi,phi = -z^2 K/t^2 + a z
        H_phi,xi  = -z da/dxi
        H_xi,xi   = (dw/dxi) L/xi^2 - (1 - w) z/(t xi^2) + 2 (1 - w) L/xi^3
                    + z (da/dxi)/xi - a z/xi^2

    Checked against central differences of the gradient in the unit tests.
    One pass costs one log and one exp per value, against six gradient
    evaluations for a finite-difference Hessian.
    """
    _, xi, sigma, z, t, L, w, valid = _terms(theta, x)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        u = 1.0 + xi - w
        a = u / t
        K = xi * u - w
        dw_dxi = w * (L / xi ** 2 - z / (xi * t))
        da_dxi = (1.0 - dw_dxi) / t - u * z / t ** 2
        g = np.stack([
            np.sum(-a / sigma, axis=1),
            np.sum(1.0 - a * z, axis=1),
            np.sum(-(1.0 - w) * L / xi ** 2 + a * z / xi, axis=1),
        ], axis=1)
        h_mm = np.sum(-K / (sigma * t) ** 2, axis=1)
        h_mp = np.sum(-z * K / (sigma * t ** 2) + a / sigma, axis=1)
        h_mx = np.sum(-da_dxi / sigma, axis=1)
        h_pp = np.sum(-z ** 2 * K / t ** 2 + a * z, axis=1)
        h_px = np.sum(-z * da_dxi, axis=1)
        h_xx = np.sum(dw_dxi * L / xi ** 2 - (1.0 - w) * z / (t * xi ** 2)
                      + 2.0 * (1.0 - w) * L / xi ** 3
                      + z * da_dxi / xi - a * z / xi ** 2, axis=1)
    H = np.stack([np.stack([h_mm, h_mp, h_mx], axis=1),
                  np.stack([h_mp, h_pp, h_px], axis=1),
                  np.stack([h_mx, h_px, h_xx], axis=1)], axis=1)
    g[~valid] = np.nan
    H[~valid] = np.nan
    return g, H


def lmoment_start(x: np.ndarray) -> np.ndarray:
    """L-moment estimates (Hosking et al., 1985) of every row, as theta (B, 3).

    Same formulas as ``gev.fit_lmom``. Rows whose L-moment fit does not
    support all their values (possible for xi < 0, when the upper endpoint
    falls below the sample maximum) get a Gumbel method-of-moments start
    instead, which supports any sample.
    """
    xs = np.sort(x, axis=1)
    _, n = xs.shape
    j = np.arange(1, n + 1)
    b0 = xs.mean(axis=1)
    b1 = np.sum((j - 1) / (n - 1) * xs, axis=1) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * xs, axis=1) / n
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    with np.errstate(invalid="ignore", divide="ignore"):
        t3 = l3 / l2
        c = 2.0 / (3.0 + t3) - np.log(2.0) / np.log(3.0)
        k = 7.8590 * c + 2.9554 * c ** 2
        k = np.where(np.abs(k) < _XI_EPS, _XI_EPS, k)
        sigma = l2 * k / ((1.0 - 2.0 ** (-k)) * gamma_fn(1.0 + k))
        mu = l1 - sigma * (1.0 - gamma_fn(1.0 + k)) / k
        theta = np.stack([mu, np.log(np.abs(sigma)),
                          np.clip(-k, XI_MIN + 0.05, XI_MAX - 0.05)], axis=1)
    bad = ~np.isfinite(negloglik(theta, x)) | ~(sigma > 0)
    if bad.any():
        s = np.sqrt(6.0 * x[bad].var(axis=1)) / np.pi
        theta[bad] = np.stack([x[bad].mean(axis=1) - _EULER * s, np.log(s),
                               np.full(s.shape, _XI_EPS)], axis=1)
    return theta


def _levenberg_marquardt(theta0: np.ndarray, x: np.ndarray,
                         max_iter: int = 200, gtol: float = 1e-6):
    """Minimise the GEV nll of every row of ``x`` from ``theta0``."""
    B = x.shape[0]
    theta = theta0.astype(float).copy()
    f = negloglik(theta, x)
    lam = np.full(B, 1e-3)
    active = np.isfinite(f)
    converged = np.zeros(B, dtype=bool)
    n_iter = np.zeros(B, dtype=int)
    eye = np.eye(3)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        th, xx = theta[idx], x[idx]
        g, H = _grad_hess(th, xx)
        # Convergence: gradient small in natural units (d/dmu scaled by sigma).
        scale = np.stack([np.exp(th[:, 1]), np.ones(idx.size),
                          np.ones(idx.size)], axis=1)
        with np.errstate(invalid="ignore"):
            gnorm = np.max(np.abs(g * scale), axis=1)
        done = np.isfinite(gnorm) & (gnorm < gtol)
        converged[idx[done]] = True
        active[idx[done]] = False
        keep = ~done
        idx, th, xx, g, H = idx[keep], th[keep], xx[keep], g[keep], H[keep]
        if idx.size == 0:
            break
        n_iter[idx] += 1
        D = np.abs(np.diagonal(H, axis1=1, axis2=2))
        D = np.where(np.isfinite(D) & (D > 1e-10), D, 1.0)
        A = H + lam[idx, None, None] * D[:, :, None] * eye
        finite = (np.all(np.isfinite(A), axis=(1, 2))
                  & np.all(np.isfinite(g), axis=1))
        A[~finite] = eye * D[~finite][:, :, None]       # fall back: scaled descent
        g = np.where(np.isfinite(g), g, 0.0)
        step = -np.linalg.solve(A, g[:, :, None])[:, :, 0]
        step[:, 2] = np.clip(step[:, 2], -0.25, 0.25)    # cautious shape steps
        trial = th + step
        trial[:, 2] = np.clip(trial[:, 2], XI_MIN, XI_MAX)
        f_new = negloglik(trial, xx)
        better = f_new < f[idx]
        acc, rej = idx[better], idx[~better]
        small = np.abs(f[acc] - f_new[better]) < 1e-12 * (1.0 + np.abs(f[acc]))
        theta[acc], f[acc] = trial[better], f_new[better]
        lam[acc] = np.maximum(lam[acc] / 3.0, 1e-12)
        lam[rej] *= 10.0
        # No further progress possible: objective unchanged, or damping huge.
        converged[acc[small]] = True
        active[acc[small]] = False
        active[rej[lam[rej] > 1e12]] = False
    return theta, f, converged, n_iter


def fit_ml(x, start: tuple[float, float, float] | None = None,
           max_iter: int = 200) -> BatchFit:
    """GEV maximum likelihood for every row of ``x`` (shape (B, n)).

    ``start`` is an optional starting point (xi, sigma, mu), e.g. the ML
    estimate of the original sample: one tuple shared by all rows, or an
    array of shape (B, 3) with one start per row. Each row's own L-moment
    estimate is always tried as well; an interior solution is preferred to a
    bound one, then the lower nll wins (see module notes).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    starts = [lmoment_start(x)]
    if start is not None:
        s = np.broadcast_to(np.asarray(start, dtype=float), (x.shape[0], 3))
        common = np.stack([s[:, 2], np.log(s[:, 1]),
                           np.clip(s[:, 0], XI_MIN, XI_MAX)], axis=1)
        starts.insert(0, common)
    best, best_score = None, None
    for th0 in starts:
        res = _levenberg_marquardt(th0, x, max_iter=max_iter)
        xi = res[0][:, 2]
        interior = (xi > XI_MIN + _BOUND_TOL) & (xi < XI_MAX - _BOUND_TOL)
        score = np.where(interior, res[1], res[1] + 1e8)   # interior first
        if best is None:
            best, best_score = [r.copy() for r in res], score
            continue
        better = score < best_score
        for k in range(4):
            best[k][better] = res[k][better]
        best_score = np.where(better, score, best_score)
    theta, f, conv, nit = best
    xi = theta[:, 2]
    bound = (xi <= XI_MIN + _BOUND_TOL) | (xi >= XI_MAX - _BOUND_TOL)
    return BatchFit(xi=xi, sigma=np.exp(theta[:, 1]), mu=theta[:, 0], nll=f,
                    converged=conv, at_boundary=bound, n_iter=nit)


def return_level(T: float, xi, sigma, mu) -> np.ndarray:
    """Vectorised GEV T-year return level (Gumbel limit for |xi| < 1e-7)."""
    xi, sigma, mu = (np.asarray(a, dtype=float) for a in (xi, sigma, mu))
    y = -np.log(1.0 - 1.0 / T)
    gumbel = np.abs(xi) < 1e-7
    xi_s = np.where(gumbel, 1.0, xi)
    out = mu + sigma / xi_s * (y ** (-xi_s) - 1.0)
    return np.where(gumbel, mu - sigma * np.log(y), out)

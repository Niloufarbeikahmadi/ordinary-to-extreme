"""
Monte Carlo check of the bootstrap intervals (Task 4).

Question
--------
A "90% bootstrap interval" is only a label. Does the 5th-95th percentile band
from 500 year-resamplings of a 30-year window really contain the true
200-year level 90% of the time? With real data the truth is unknown, so we
generate synthetic 30-year records from worlds where it IS known, and apply
exactly the same estimation and bootstrap code to each record.

Two synthetic worlds
--------------------
A. "Ordinary-events world" (the MEVD assumptions hold). Each synthetic
   year draws one (N, C, w) triplet from the station's own yearly estimates
   (all valid years), then N independent Weibull(C, w) excesses over 1 mm.
   Its true 200-year level is the MEVD of the whole pool of triplets. Both
   models are fitted and bootstrapped on the same simulated years. The GEV
   is only an approximation in this world.
B. "GEV world" (the GEV assumptions hold). Thirty annual maxima are drawn
   from the GEV fitted by ML to all valid annual maxima of the station; the
   truth is that GEV's 200-year level. Only the GEV can be assessed here,
   because there are no daily values.

World A is the generator used by Zorzetto et al. (2016) to compare the two
methods; world B checks whether the GEV bands fail even when the GEV is the
correct model. In both worlds the years are independent and identically
distributed, so the check isolates SAMPLING uncertainty: real windows can
only be harder (inhomogeneities, trends).

Computation
-----------
Records are processed in chunks: all resamples of a chunk of records are
stacked into one batch for the vectorised GEV fitter and MEVD solver. The
point GEV estimate of each record is the ML fit of ``gev_batch`` (identical
to ``gev.fit_ml`` for regular samples).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config, gev, gev_batch, mev

_CHUNK = 25          # synthetic records per vectorised batch


def _percentiles(a: np.ndarray, pct=config.CI_PERCENTILES) -> np.ndarray:
    return np.percentile(a, pct, axis=1).T          # (R, 2)


def _gev_points_and_bands(amax: np.ndarray, B: int, T: float,
                          rng: np.random.Generator):
    """Point ML estimates and bootstrap bands of the T-year GEV level.

    ``amax`` is (R, n). Returns (point (R,), band (R, 2), idx (R, B, n)).
    """
    R, n = amax.shape
    pt = gev_batch.fit_ml(amax)
    rl_pt = pt.return_level(T)
    idx = rng.integers(0, n, size=(R, B, n))
    rl_bs = np.empty((R, B))
    for c in range(0, R, _CHUNK):
        sl = slice(c, min(c + _CHUNK, R))
        r = np.arange(sl.start, sl.stop)
        X = np.take_along_axis(amax[r][:, None, :].repeat(B, axis=1),
                               idx[r], axis=2).reshape(-1, n)
        starts = np.repeat(np.stack([pt.xi[r], pt.sigma[r], pt.mu[r]], 1),
                           B, axis=0)
        fit = gev_batch.fit_ml(X, start=starts)
        rl_bs[r] = fit.return_level(T).reshape(-1, B)
    return rl_pt, _percentiles(rl_bs), idx


def ordinary_world(pool: pd.DataFrame, station: str,
                   R: int = config.COVERAGE_RECORDS,
                   n_years: int = config.COVERAGE_YEARS,
                   B: int = config.N_BOOTSTRAP, T: float = config.RETURN_PERIOD,
                   seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """World A: synthetic daily records generated from the yearly (N, C, w)."""
    from .bootstrap import station_code
    rng = np.random.default_rng([seed, station_code(station), 1])
    thr = config.WET_THRESHOLD_MM
    P = pool[["N", "C", "w"]].to_numpy(float)
    truth = mev.mev_return_level(T, P[:, 0], P[:, 1], P[:, 2])
    pick = rng.integers(0, len(P), size=(R, n_years))
    N = P[pick, 0]
    C_hat, w_hat, amax = (np.empty((R, n_years)) for _ in range(3))
    for r in range(R):
        for j in range(n_years):
            n_wet, C, w = int(N[r, j]), P[pick[r, j], 1], P[pick[r, j], 2]
            exc = C * rng.weibull(w, size=n_wet)      # excesses over 1 mm
            fit = mev.weibull_fit_pwm(exc)
            C_hat[r, j], w_hat[r, j] = fit.C, fit.w
            amax[r, j] = exc.max() + thr
    gev_pt, gev_band, idx = _gev_points_and_bands(amax, B, T, rng)
    F = 1.0 - 1.0 / T
    mev_pt = mev.mev_quantile_batch(F, N, C_hat, w_hat)
    mev_band = np.empty((R, 2))
    for r in range(R):
        i = idx[r]
        bs = mev.mev_quantile_batch(F, N[r][i], C_hat[r][i], w_hat[r][i])
        mev_band[r] = np.percentile(bs, config.CI_PERCENTILES)
    return pd.DataFrame({
        "world": "ordinary", "station": station, "record": np.arange(R),
        "truth": truth,
        "gev_point": gev_pt, "gev_lo": gev_band[:, 0], "gev_hi": gev_band[:, 1],
        "mev_point": mev_pt, "mev_lo": mev_band[:, 0], "mev_hi": mev_band[:, 1],
    })


def gev_world(amax_all: np.ndarray, station: str,
              R: int = config.COVERAGE_RECORDS,
              n_years: int = config.COVERAGE_YEARS,
              B: int = config.N_BOOTSTRAP, T: float = config.RETURN_PERIOD,
              seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """World B: synthetic annual maxima from the station's full-record GEV."""
    from .bootstrap import station_code
    rng = np.random.default_rng([seed, station_code(station), 2])
    p = gev.fit_ml(amax_all)
    truth = float(gev.return_level(T, p))
    amax = gev.quantile(rng.uniform(size=(R, n_years)), p.xi, p.sigma, p.mu)
    gev_pt, gev_band, _ = _gev_points_and_bands(amax, B, T, rng)
    return pd.DataFrame({
        "world": "gev", "station": station, "record": np.arange(R),
        "truth": truth,
        "gev_point": gev_pt, "gev_lo": gev_band[:, 0], "gev_hi": gev_band[:, 1],
        "mev_point": np.nan, "mev_lo": np.nan, "mev_hi": np.nan,
        "true_xi": p.xi, "true_sigma": p.sigma, "true_mu": p.mu,
    })


def estimation_noise_experiment(N: int, C: float, w: float, station: str,
                                R: int = config.COVERAGE_RECORDS,
                                n_years: int = config.COVERAGE_YEARS,
                                return_periods=(10.0, 50.0, config.RETURN_PERIOD),
                                seed: int = config.RANDOM_SEED
                                ) -> pd.DataFrame:
    """Isolate the effect of yearly Weibull estimation noise on the MEVD.

    Every synthetic year is IDENTICAL: N wet days, Weibull(C, w) excesses.
    The true T-year level is then exact, theta + C [-ln(1 - F^(1/N))]^(1/w),
    and the only source of error is estimation from finite samples. Two
    estimators are compared on the same 30-year records:

    * ``yearly``: one PWM Weibull per year (the main MEVD configuration,
      estimation window EW = 1 year), then the MEVD average over years;
    * ``pooled``: one PWM Weibull fitted to all wet days of the record
      (EW = whole sample, the idea behind SMEV; Marra et al., 2019), used
      with the yearly N.

    If the yearly estimator is biased here, where nothing varies between
    years, the bias comes from estimation noise alone: random scatter in the
    yearly shape w adds spurious heavy-tailed years, and the MEVD tail is
    dominated by its heaviest-tailed years (a Jensen-type effect).
    """
    from .bootstrap import station_code
    rng = np.random.default_rng([seed, station_code(station), 3])
    rows = []
    for r in range(R):
        X = [C * rng.weibull(w, size=N) for _ in range(n_years)]
        fits = [mev.weibull_fit_pwm(x) for x in X]
        pooled = mev.weibull_fit_pwm(np.concatenate(X))
        for T in return_periods:
            truth = mev.mev_return_level(T, [N], [C], [w])
            rows.append({
                "station": station, "record": r, "T": T, "truth": truth,
                "yearly": mev.mev_return_level(T, [N] * n_years,
                                               [f.C for f in fits],
                                               [f.w for f in fits]),
                "pooled": mev.mev_return_level(T, [N], [pooled.C], [pooled.w]),
                "w_sd_yearly": float(np.std([f.w for f in fits])),
            })
    df = pd.DataFrame(rows)
    df["N"], df["C"], df["w"] = N, C, w
    return df


def shape_noise_share(pool: pd.DataFrame, station: str, R: int = 200,
                      seed: int = config.RANDOM_SEED) -> dict:
    """How much of the observed year-to-year scatter of w is estimation noise?

    For every valid year j, R synthetic years of N_j Weibull(C_j, w_j)
    excesses are refitted by PWM; the variance of these refits is the
    sampling (noise) variance of w_j. Its average over years is compared
    with the observed variance of the yearly estimates across years:

        observed var(w_hat) = true between-year var(w) + mean noise var

    so the ratio estimates the share of the scatter that is noise.
    """
    from .bootstrap import station_code
    rng = np.random.default_rng([seed, station_code(station), 4])
    noise_var = []
    for N, C, w in pool[["N", "C", "w"]].to_numpy(float):
        refits = [mev.weibull_fit_pwm(C * rng.weibull(w, size=int(N))).w
                  for _ in range(R)]
        noise_var.append(np.var(refits))
    obs_var = float(np.var(pool["w"]))
    mean_noise = float(np.mean(noise_var))
    return {"station": station, "n_years": len(pool),
            "observed_sd_w": np.sqrt(obs_var),
            "noise_sd_w": np.sqrt(mean_noise),
            "noise_share_of_variance": mean_noise / obs_var,
            "implied_true_sd_w": np.sqrt(max(obs_var - mean_noise, 0.0))}


def summarise_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Median relative error and spread of the two estimators, by T."""
    out = []
    for (st, T), g in df.groupby(["station", "T"]):
        for est in ("yearly", "pooled"):
            rel = g[est] / g["truth"] - 1.0
            out.append({"station": st, "T": T, "estimator": est,
                        "truth_mm": float(g["truth"].iat[0]),
                        "median_rel_error": float(rel.median()),
                        "p05_rel_error": float(rel.quantile(0.05)),
                        "p95_rel_error": float(rel.quantile(0.95)),
                        "N": int(g["N"].iat[0]), "C": float(g["C"].iat[0]),
                        "w": float(g["w"].iat[0])})
    return pd.DataFrame(out)


def summarise(df: pd.DataFrame, nominal: float = 0.90) -> pd.DataFrame:
    """Coverage, miss direction, point-estimate error and band width."""
    rows = []
    for (world, st), g in df.groupby(["world", "station"], sort=False):
        for m in ("gev", "mev"):
            if g[f"{m}_point"].isna().all():
                continue
            lo, hi, pt, tr = (g[f"{m}_lo"], g[f"{m}_hi"], g[f"{m}_point"],
                              g["truth"])
            cover = ((lo <= tr) & (tr <= hi)).mean()
            R = len(g)
            rel = pt / tr - 1.0
            rows.append({
                "world": world, "station": st, "model": m.upper(),
                "n_records": R, "truth_mm": float(tr.iat[0]),
                "coverage": cover,
                # Monte Carlo standard error of the coverage estimate
                "coverage_se": np.sqrt(cover * (1 - cover) / R),
                "nominal": nominal,
                "share_truth_above_band": float((tr > hi).mean()),
                "share_truth_below_band": float((tr < lo).mean()),
                "median_rel_error": float(rel.median()),
                "rmse_rel": float(np.sqrt(np.mean(rel ** 2))),
                "median_rel_width": float(((hi - lo) / tr).median()),
            })
    return pd.DataFrame(rows)


def run(station: str, annual: pd.DataFrame, yearly: pd.DataFrame,
        R: int = config.COVERAGE_RECORDS, verbose: bool = True
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both worlds and the estimation-noise experiment for one station.

    Uses the valid years of the record as provided ('asis'). The identical-
    years experiment uses the station's median yearly N, C and w.
    Returns (coverage records, noise-experiment records).
    """
    t0 = time.time()
    valid = annual.loc[annual["valid"], "year"]
    pool = yearly[yearly["year"].isin(valid)]
    amax_all = annual.loc[annual["valid"], "annual_max_mm"].to_numpy(float)
    cov = pd.concat([ordinary_world(pool, station, R),
                     gev_world(amax_all, station, R)], ignore_index=True)
    med = pool[["N", "C", "w"]].median()
    noise = estimation_noise_experiment(int(round(med["N"])), float(med["C"]),
                                        float(med["w"]), station, R)
    if verbose:
        print(f"  {station}: coverage check (2 worlds) and estimation-noise "
              f"experiment, {R} synthetic records each, "
              f"in {time.time() - t0:.0f} s")
    return cov, noise

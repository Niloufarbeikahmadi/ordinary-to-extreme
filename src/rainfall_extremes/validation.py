"""
Out-of-sample check of GEV and MEVD: leave-one-year-out (LOYO).

For every valid year j, both models are fitted to the OTHER valid years of
the 30-year calendar window centred on j (years j-15 ... j+14, without j).
The observed annual maximum of year j is then compared with the fitted
models:

* PIT value u_j = F_{-j}(max_j). If the model is well calibrated, the u_j
  are uniform on (0, 1).
* Exceedance counts: how often max_j exceeds the model's T-year level. A
  calibrated model is exceeded in about 1/T of the years. This is a direct
  test of the upper tail, where the two methods differ.

Because year j is never used to fit the model that judges it, this avoids
the optimism of in-sample checks. Caveats: neighbouring tests share most of
their fitting years (so they are not independent), and slow changes in
climate or observation practice add spread to both models alike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kstest

from . import config, gev, mev

LOYO_HALF_WINDOW: int = 15
LOYO_RETURN_PERIODS: tuple[int, ...] = (10, 50)


def loyo_table(annual: pd.DataFrame, yearly: pd.DataFrame,
               half: int = LOYO_HALF_WINDOW,
               return_periods=LOYO_RETURN_PERIODS) -> pd.DataFrame:
    """One row per tested year: PIT values and exceedance indicators."""
    av = annual.loc[annual["valid"]].set_index("year")
    yp = yearly.set_index("year")
    years = av.index.to_numpy()
    rows = []
    for j in years:
        others = [y for y in years if j - half <= y <= j + half - 1 and y != j]
        if len(others) < config.MIN_VALID_YEARS:
            continue
        x = av.loc[others, "annual_max_mm"].to_numpy(dtype=float)
        p = yp.loc[others]
        mj = float(av.at[j, "annual_max_mm"])
        g = gev.fit_ml(x)
        row = {"year": int(j), "n_fit": len(others), "annual_max_mm": mj,
               "u_gev": float(gev.cdf(mj, *g.as_tuple())),
               "u_mev": float(mev.mev_cdf(mj, p["N"], p["C"], p["w"])[0])}
        for T in return_periods:
            row[f"exceed_gev_{T}"] = bool(mj > gev.return_level(T, g))
            row[f"exceed_mev_{T}"] = bool(
                mj > mev.mev_return_level(T, p["N"], p["C"], p["w"]))
        rows.append(row)
    return pd.DataFrame(rows)


def loyo_summary(table: pd.DataFrame,
                 return_periods=LOYO_RETURN_PERIODS) -> pd.DataFrame:
    """KS uniformity test and binomial exceedance tests for both models."""
    n = len(table)
    rows = []
    for m in ("gev", "mev"):
        row = {"model": m.upper(), "n_years": n,
               "ks_pvalue": kstest(table[f"u_{m}"], "uniform").pvalue}
        for T in return_periods:
            k = int(table[f"exceed_{m}_{T}"].sum())
            row[f"exceed_{T}_observed"] = k
            row[f"exceed_{T}_expected"] = n / T
            row[f"exceed_{T}_binom_p"] = binomtest(k, n, 1 / T).pvalue
        rows.append(row)
    return pd.DataFrame(rows)


def binomial_interval(n: int, p: float, level: float = 0.90
                      ) -> tuple[float, float]:
    """Central ``level`` range of a Binomial(n, p) count."""
    from scipy.stats import binom
    a = (1 - level) / 2
    return float(binom.ppf(a, n, p)), float(binom.ppf(1 - a, n, p))

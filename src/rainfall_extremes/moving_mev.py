"""
Moving-window MEVD-Weibull analysis (Task 3).

Yearly ordinary-event parameters (N_j, C_j, w_j) are estimated ONCE per year
and per configuration. Each window's MEVD is then the average of its valid
years' yearly maximum distributions. The windows are exactly the same objects
used for the GEV (same valid years), so GEV and MEVD are always compared on
identical samples. The same yearly table is the input of the common
bootstrap in Task 4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import config, mev
from .datasets import SCENARIO_COLUMN
from .windows import Window, build_windows

RETURN_PERIODS: tuple[float, ...] = (100.0, config.RETURN_PERIOD)


@dataclass(frozen=True)
class MEVConfig:
    """One MEVD configuration. The default is the main analysis."""
    scenario: str = "asis"
    window_mode: str = "calendar"
    formulation: str = "excess"
    method: str = "pwm"

    @property
    def label(self) -> str:
        return (f"{self.scenario}/{self.window_mode}/"
                f"{self.formulation}/{self.method}")


#: Main configuration plus one-at-a-time sensitivity variants.
CONFIGS: tuple[MEVConfig, ...] = (
    MEVConfig(),                                  # main
    MEVConfig(scenario="qc"),                     # data screening
    MEVConfig(window_mode="consecutive"),         # window definition
    MEVConfig(formulation="raw"),                 # no threshold subtraction
    MEVConfig(method="ml"),                       # Weibull estimator
)


def yearly_table(daily: pd.DataFrame, cfg: MEVConfig) -> pd.DataFrame:
    """Yearly (N, C, w) for one configuration."""
    yp = mev.yearly_parameters(daily, SCENARIO_COLUMN[cfg.scenario],
                               formulation=cfg.formulation, method=cfg.method)
    yp.insert(0, "scenario", cfg.scenario)
    return yp


def mev_over_windows(yearly: pd.DataFrame, windows: list[Window],
                     formulation: str = "excess",
                     return_periods: Iterable[float] = RETURN_PERIODS
                     ) -> pd.DataFrame:
    """MEVD return levels in every window from the yearly parameters."""
    yp = yearly.set_index("year")
    rows = []
    for w in windows:
        p = yp.loc[list(w.years)]
        ok = p[["C", "w"]].notna().all(axis=1)
        p = p[ok]
        row = {"window_start": w.start, "window_end": w.end,
               "window_mode": w.mode, "n_valid": int(ok.sum()),
               "mean_N": float(p["N"].mean()), "mean_C": float(p["C"].mean()),
               "mean_w": float(p["w"].mean())}
        for T in return_periods:
            row[f"rl_{int(T)}"] = mev.mev_return_level(
                T, p["N"], p["C"], p["w"], formulation=formulation)
        rows.append(row)
    return pd.DataFrame(rows)


def run_station(daily: pd.DataFrame, annual_long: pd.DataFrame,
                configs: Iterable[MEVConfig] = CONFIGS
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MEVD for every configuration of one station.

    Returns (window results, yearly parameters), both in long format.
    """
    station = str(annual_long["station"].iat[0])
    res_parts, yearly_parts, cache = [], [], {}
    for cfg in configs:
        key = (cfg.scenario, cfg.formulation, cfg.method)
        if key not in cache:
            cache[key] = yearly_table(daily, cfg)
            yearly_parts.append(cache[key])
        annual = annual_long.loc[annual_long["scenario"] == cfg.scenario]
        wins = build_windows(annual, cfg.window_mode)
        res = mev_over_windows(cache[key], wins, cfg.formulation)
        res.insert(0, "method", cfg.method)
        res.insert(0, "formulation", cfg.formulation)
        res.insert(0, "scenario", cfg.scenario)
        res.insert(0, "station", station)
        res_parts.append(res)
    yearly = pd.concat(yearly_parts, ignore_index=True)
    yearly.insert(0, "station", station)
    return pd.concat(res_parts, ignore_index=True), yearly


def select(results: pd.DataFrame, scenario: str = "asis",
           mode: str = "calendar", formulation: str = "excess",
           method: str = "pwm") -> pd.DataFrame:
    sel = ((results["scenario"] == scenario)
           & (results["window_mode"] == mode)
           & (results["formulation"] == formulation)
           & (results["method"] == method))
    return results.loc[sel].sort_values("window_end").reset_index(drop=True)


def compare_with_gev(mev_res: pd.DataFrame, gev_res: pd.DataFrame,
                     T: float = config.RETURN_PERIOD) -> pd.DataFrame:
    """Main MEVD vs main GEV (ML), window by window."""
    from . import moving_gev
    col = f"rl_{int(T)}"
    m = select(mev_res)[["window_start", "window_end", "n_valid", col]]
    g = moving_gev.select(gev_res)[["window_end", col, "xi"]]
    out = m.merge(g, on="window_end", suffixes=("_mev", "_gev"))
    out["diff_gev_minus_mev"] = out[f"{col}_gev"] - out[f"{col}_mev"]
    return out


def stability_summary(cmp: pd.DataFrame, T: float = config.RETURN_PERIOD
                      ) -> dict:
    """How variable is each method across windows?

    * range and median of the return level;
    * coefficient of variation across windows;
    * median and maximum absolute change between consecutive windows
      (which share 29 of 30 years, so large jumps reflect single years).
    """
    col = f"rl_{int(T)}"
    out = {}
    for m in ("gev", "mev"):
        s = cmp.set_index("window_end")[f"{col}_{m}"]
        step = s.diff().where(s.index.to_series().diff() == 1).abs()
        out.update({
            f"{m}_min": s.min(), f"{m}_median": s.median(), f"{m}_max": s.max(),
            f"{m}_cv": s.std() / s.mean(),
            f"{m}_median_step": step.median(), f"{m}_max_step": step.max(),
        })
    out["share_gev_above_mev"] = float((cmp["diff_gev_minus_mev"] > 0).mean())
    out["corr_gev_mev"] = float(np.corrcoef(cmp[f"{col}_gev"],
                                            cmp[f"{col}_mev"])[0, 1])
    return out

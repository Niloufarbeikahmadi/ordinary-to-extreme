"""
Task 4 - Common bootstrap of the GEV and MEVD 200-year levels, and a Monte
Carlo check of the bootstrap intervals.

Run in Spyder (F5) or:  python scripts/04_bootstrap.py
(needs the processed data of script 01; about 3-4 minutes in total)

Bootstrap (as the task specifies)
    In every retained 30-year window, the n valid years are resampled with
    replacement 500 times. Each drawn year brings its annual maximum (GEV)
    AND its (N, C, w) (MEVD): the same synthetic samples feed both models.
    The pointwise 90% interval is the 5th-95th percentile of the 500
    estimates. Seeds are fixed per window (reproducible, order-independent).

Monte Carlo check (our addition)
    Synthetic 30-year records from worlds where the true 200-year level is
    known, analysed with exactly the same code: do the 90% bands contain the
    truth 90% of the time? Plus an experiment that isolates the effect of
    yearly Weibull estimation noise on the MEVD.

Outputs: outputs/tables/bootstrap_{station}.csv           (one row per window)
         outputs/tables/bootstrap_replicates_{station}.npz (all 500 replicates)
         outputs/tables/bootstrap_summary.csv
         outputs/tables/bootstrap_wettest_year_effect.csv
         outputs/tables/gev_batch_check.csv
         outputs/tables/coverage_records.csv, coverage_summary.csv
         outputs/tables/estimation_noise_summary.csv, shape_noise_share.csv
         outputs/figures/fig17_* ... fig20_*
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    _ROOT = Path(__file__).resolve().parents[1]
except NameError:
    _ROOT = Path.cwd()
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rainfall_extremes import (bootstrap, bootstrap_plots, config,  # noqa: E402
                               coverage, datasets, mev)
from rainfall_extremes.plot_style import save_figure, use_style  # noqa: E402
from rainfall_extremes.windows import build_windows  # noqa: E402

STATION_ORDER = ("bologna", "padova")
SHOW_FIGURES = False
#: The Monte Carlo check takes about two minutes. If False, the tables of a
#: previous run are reused (when present) for Figs. 19-20.
RUN_COVERAGE = True
#: Compare the fast batched GEV fitter with the reference fitter (~20 s).
CHECK_FITTER = True
#: Windows shown in Fig. 18: (station, final year, note).
DIST_CASES = [
    ("bologna", 1962, "includes 24 Feb 1933 (155.7 mm)"),
    ("bologna", 1963, "first window without 1933"),
    ("padova", 1976, "highest GEV estimate"),
    ("padova", 2019, "most recent window"),
]


def _save(fig, name):
    save_figure(fig, name)
    if not SHOW_FIGURES:
        plt.close(fig)


def _expected_offset(cov_summary: pd.DataFrame) -> dict[str, float]:
    """(GEV - MEVD) / MEVD expected from the median errors in world A."""
    out = {}
    ordw = cov_summary[cov_summary["world"] == "ordinary"]
    for st, g in ordw.groupby("station"):
        e = g.set_index("model")["median_rel_error"]
        out[st] = (e["GEV"] - e["MEV"]) / (1.0 + e["MEV"])
    return out


def main():
    use_style()
    config.ensure_output_dirs()
    t_start = time.time()
    summaries, replicates, annuals = {}, {}, {}
    checks, cov_parts, noise_parts, shares, effects = [], [], [], [], []
    for st in STATION_ORDER:
        daily = datasets.load_daily(st)
        annual = datasets.load_annual(st, "asis")
        annuals[st] = annual
        yearly = mev.yearly_parameters(daily)            # excess, PWM (main)
        windows = build_windows(annual, "calendar")
        summ, reps = bootstrap.run_station(st, annual, yearly, windows)
        summaries[st], replicates[st] = summ, reps
        summ.to_csv(config.TABLES / f"bootstrap_{st}.csv", index=False,
                    float_format="%.4f")
        np.savez_compressed(config.TABLES / f"bootstrap_replicates_{st}.npz",
                            **reps)
        samples = bootstrap.window_samples(st, annual, yearly, windows)
        by_end = {ws.end: (k, ws) for k, ws in enumerate(samples)}
        for case_st, end, _ in DIST_CASES:
            if case_st == st:
                k, ws = by_end[end]
                effects.append(bootstrap.largest_year_effect(
                    ws, reps["rl_gev"][k], reps["rl_mev"][k]))
        if CHECK_FITTER:
            checks.append(bootstrap.check_batch_fitter(samples))
        if RUN_COVERAGE:
            cov, noise = coverage.run(st, annual, yearly)
            cov_parts.append(cov)
            noise_parts.append(noise)
            pool = yearly[yearly["year"].isin(annual.loc[annual["valid"], "year"])]
            shares.append(coverage.shape_noise_share(pool, st))

    station_summary = pd.DataFrame([bootstrap.station_summary(summaries[s])
                                    for s in STATION_ORDER])
    station_summary.to_csv(config.TABLES / "bootstrap_summary.csv",
                           index=False, float_format="%.4f")
    effect = pd.concat(effects, ignore_index=True)
    effect.to_csv(config.TABLES / "bootstrap_wettest_year_effect.csv",
                  index=False, float_format="%.2f")

    if CHECK_FITTER:
        chk = pd.concat(checks, ignore_index=True)
        chk.to_csv(config.TABLES / "gev_batch_check.csv", index=False,
                   float_format="%.6f")
        reg = chk[chk["ref_in_bounds"]]
        print(f"\nFast vs reference GEV fitter on {len(chk)} bootstrap "
              f"replicates: max |200-yr diff| = "
              f"{reg['abs_diff_rl_mm'].max():.2e} mm where the reference "
              f"shape lies in [-1, 1] ({len(reg)} cases); "
              f"{len(chk) - len(reg)} reference fits left that range.")

    cov_path = config.TABLES / "coverage_summary.csv"
    if RUN_COVERAGE:
        cov_records = pd.concat(cov_parts, ignore_index=True)
        cov_summary = coverage.summarise(cov_records)
        noise_records = pd.concat(noise_parts, ignore_index=True)
        noise_summary = coverage.summarise_noise(noise_records)
        noise_share = pd.DataFrame(shares)
        cov_records.to_csv(config.TABLES / "coverage_records.csv",
                           index=False, float_format="%.4f")
        cov_summary.to_csv(cov_path, index=False, float_format="%.4f")
        noise_summary.to_csv(config.TABLES / "estimation_noise_summary.csv",
                             index=False, float_format="%.4f")
        noise_share.to_csv(config.TABLES / "shape_noise_share.csv",
                           index=False, float_format="%.4f")
    elif cov_path.exists():
        cov_records = pd.read_csv(config.TABLES / "coverage_records.csv")
        cov_summary = pd.read_csv(cov_path)
        noise_summary = pd.read_csv(config.TABLES / "estimation_noise_summary.csv")
        noise_share = pd.read_csv(config.TABLES / "shape_noise_share.csv")
    else:
        cov_summary = None

    # ---------------------------------------------------------------- figures
    _save(bootstrap_plots.plot_bootstrap_bands(summaries),
          "fig17_bootstrap_bands_rl200")
    _save(bootstrap_plots.plot_bootstrap_distributions(
        DIST_CASES, replicates, summaries, annuals),
        "fig18_bootstrap_distributions")
    offset = _expected_offset(cov_summary) if cov_summary is not None else None
    _save(bootstrap_plots.plot_paired_difference(summaries, offset),
          "fig19_paired_difference_gev_mev")
    if cov_summary is not None:
        _save(bootstrap_plots.plot_calibration(cov_records, cov_summary,
                                               noise_summary, noise_share),
              "fig20_bootstrap_calibration")

    with pd.option_context("display.width", 160, "display.precision", 3,
                           "display.max_columns", 30):
        print("\nBootstrap summary per station:\n", station_summary.T)
        print("\nBootstrap estimates by copies of the wettest year drawn "
              "(Fig. 18 windows):\n", effect.drop(columns="station"))
        if cov_summary is not None:
            print("\nMonte Carlo check of the 90% bands:\n", cov_summary)
            print("\nYearly Weibull estimation noise (identical years):\n",
                  noise_summary)
            print("\nShare of the scatter of yearly w that is noise:\n",
                  noise_share)
    print(f"\nTask 4 finished in {time.time() - t_start:.0f} s.")
    if SHOW_FIGURES:
        plt.show()
    return summaries


if __name__ == "__main__":
    RESULTS = main()

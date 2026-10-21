"""
Task 3 - Moving-window MEVD-Weibull analysis and comparison with the GEV.

Run in Spyder (F5) or:  python scripts/03_moving_window_mev.py
(requires the outputs of scripts 01 and 02)

Main configuration (task + Zorzetto et al., 2016 + mevpy):
    wet day: P > 1 mm; Weibull fitted year by year to the EXCESSES P - 1 mm
    by probability weighted moments; MEVD averaged over the valid years of
    each 30-calendar-year window (>= 20 valid years); 200-year level solved
    from zeta(x) = 1 - 1/200, with 1 mm added back.
Sensitivity variants: QC-screened record, consecutive-year windows, raw
wet-day values (no threshold subtraction), maximum-likelihood Weibull.
Diagnostics: GEV vs MEVD stability; leave-one-year-out calibration.

Outputs: outputs/tables/mev_moving_windows_{station}.csv
         outputs/tables/mev_yearly_parameters_{station}.csv
         outputs/tables/gev_vs_mev_{station}.csv
         outputs/tables/loyo_{station}.csv, loyo_summary.csv
         outputs/tables/gev_vs_mev_summary.csv
         outputs/figures/fig12_* ... fig16_*
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
import pandas as pd  # noqa: E402

from rainfall_extremes import (config, datasets, mev_plots,  # noqa: E402
                               moving_gev, moving_mev, validation)
from rainfall_extremes.plot_style import save_figure, use_style  # noqa: E402

STATION_ORDER = ("bologna", "padova")
SHOW_FIGURES = False
#: Windows shown in Fig. 15: Bologna with and without the 1933 record day,
#: and the Padova window with the highest GEV estimate.
FIT_CASES = [("bologna", 1933, 1962), ("bologna", 1934, 1963),
             ("padova", 1947, 1976)]


def _save(fig, name):
    save_figure(fig, name)
    if not SHOW_FIGURES:
        plt.close(fig)


def _gev_results(st: str, annual_long: pd.DataFrame) -> pd.DataFrame:
    path = config.TABLES / f"gev_moving_windows_{st}.csv"
    if path.exists():
        return pd.read_csv(path)
    print(f"  {path.name} missing: running the GEV analysis (Task 2) ...")
    return moving_gev.run_station(annual_long)


def main():
    use_style()
    config.ensure_output_dirs()
    mev_res, gev_res, yearlies, annuals = {}, {}, {}, {}
    loyo_tabs, loyo_sums, summary_rows = {}, {}, []
    for st in STATION_ORDER:
        t0 = time.time()
        daily = datasets.load_daily(st)
        annual_long = datasets.load_annual(st)
        annuals[st] = annual_long[annual_long["scenario"] == "asis"]
        res, yearly = moving_mev.run_station(daily, annual_long)
        mev_res[st], yearlies[st] = res, yearly
        gev_res[st] = _gev_results(st, annual_long)
        res.to_csv(config.TABLES / f"mev_moving_windows_{st}.csv",
                   index=False, float_format="%.5f")
        yearly.to_csv(config.TABLES / f"mev_yearly_parameters_{st}.csv",
                      index=False, float_format="%.6f")

        cmp = moving_mev.compare_with_gev(res, gev_res[st])
        cmp.to_csv(config.TABLES / f"gev_vs_mev_{st}.csv", index=False,
                   float_format="%.3f")
        summary_rows.append({"station": st,
                             **moving_mev.stability_summary(cmp)})

        main_yearly = yearly[(yearly["scenario"] == "asis")
                             & (yearly["formulation"] == "excess")
                             & (yearly["method"] == "pwm")]
        tab = validation.loyo_table(annuals[st], main_yearly)
        summ = validation.loyo_summary(tab).assign(station=st)
        tab.to_csv(config.TABLES / f"loyo_{st}.csv", index=False,
                   float_format="%.5f")
        loyo_tabs[st], loyo_sums[st] = tab, summ
        print(f"{st}: MEVD + LOYO in {time.time() - t0:.1f} s")

        _save(mev_plots.plot_yearly_parameters(yearly, annuals[st]),
              f"fig12_mev_yearly_parameters_{st}")
        _save(mev_plots.plot_mev_sensitivity(res), f"fig14_mev_sensitivity_{st}")

    _save(mev_plots.plot_gev_vs_mev(gev_res, mev_res, annuals),
          "fig13_gev_vs_mev_rl200")
    _save(mev_plots.plot_window_fits_comparison(FIT_CASES, annuals, yearlies),
          "fig15_gev_vs_mev_window_fits")
    _save(mev_plots.plot_loyo(loyo_tabs, loyo_sums),
          "fig16_out_of_sample_calibration")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.TABLES / "gev_vs_mev_summary.csv", index=False,
                   float_format="%.3f")
    loyo_all = pd.concat(loyo_sums.values(), ignore_index=True)
    loyo_all.to_csv(config.TABLES / "loyo_summary.csv", index=False,
                    float_format="%.4f")
    with pd.option_context("display.width", 140, "display.precision", 3):
        print("\nStability of the 200-year level across windows:\n",
              summary.T)
        print("\nLeave-one-year-out calibration:\n", loyo_all)
    if SHOW_FIGURES:
        plt.show()
    return mev_res


if __name__ == "__main__":
    RESULTS = main()

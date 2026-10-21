"""
Task 5 - Results for Bologna, then Padova, and sensitivity analyses.

Run in Spyder (F5) or:  python scripts/05_results_and_sensitivity.py
(needs the tables of scripts 01-04; about 1 minute)

1. Station results: 200-year GEV and MEVD levels with the 90% bands of the
   common bootstrap, the MEVD drivers (N, C, w) and the GEV shape.
2. MEVD estimation window (EW = 1, 3, 5 years, whole window, SMEV): point
   estimates, bootstrap of the pooled variant, leave-one-year-out check, and
   a tail-versus-bulk check of the Weibull.
3. One-at-a-time sensitivity of every analysis choice.
4. Recent change between disjoint 30-year windows, and the Bologna-Padova
   comparison (inputs of the report's discussion, Task 6).
5. The compact figures of the final report.

Outputs: outputs/tables/mev_estimation_window_{station}.csv,
         bootstrap_pooled_{station}.csv, loyo_estimation_window.csv,
         tail_check_{station}.csv, sensitivity_long.csv,
         sensitivity_summary.csv, valid_year_rule.csv, recent_change.csv,
         station_correlations_yearly.csv, station_correlations_windows.csv
         outputs/figures/fig21_* ... fig25_*, report_fig1-3_*, report_equation
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

from rainfall_extremes import (comparison, config, datasets,  # noqa: E402
                               estimation_window as ewm, mev, results_plots,
                               sensitivity)
from rainfall_extremes.plot_style import save_figure, use_style  # noqa: E402
from rainfall_extremes.windows import build_windows  # noqa: E402

STATION_ORDER = ("bologna", "padova")
SHOW_FIGURES = False
#: Disjoint windows compared in the recent-change test: (station, first
#: window end, second window end). 1948-1977 vs 1978-2007 is common to both
#: stations; Padova's most recent pair is added.
CHANGE_CASES = [("bologna", 1977, 2007), ("padova", 1977, 2007),
                ("padova", 1989, 2019)]
#: Window and year of the report's bootstrap example (Bologna, 24 Feb 1933).
REPORT_CASE = ("bologna", 1962, 1933)


def _save(fig, name):
    save_figure(fig, name)
    if not SHOW_FIGURES:
        plt.close(fig)


def _need(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing: run scripts 02-04 "
                                "first (they produce the main results).")
    return pd.read_csv(path)


def _main_mev(table: pd.DataFrame) -> pd.DataFrame:
    sel = ((table["scenario"] == "asis") & (table["window_mode"] == "calendar")
           & (table["formulation"] == "excess") & (table["method"] == "pwm"))
    return table[sel]


def main():
    use_style()
    config.ensure_output_dirs()
    T = config.TABLES
    t0 = time.time()
    data, summaries, pooled, mev_main, ew_tables, tails = {}, {}, {}, {}, {}, {}
    loyo_rows, long_parts, valid_rules, annuals, replicates = [], [], [], {}, {}
    for st in STATION_ORDER:
        daily = datasets.load_daily(st)
        annual = datasets.load_annual(st, "asis")
        yearly = mev.yearly_parameters(daily)
        exc = ewm.excess_by_year(daily)
        windows = build_windows(annual, "calendar")
        data[st] = (annual, yearly, exc)
        annuals[st] = annual
        summaries[st] = _need(T / f"bootstrap_{st}.csv")
        replicates[st] = dict(np.load(T / f"bootstrap_replicates_{st}.npz"))
        gev_tab = _need(T / f"gev_moving_windows_{st}.csv")
        mev_tab = _need(T / f"mev_moving_windows_{st}.csv")
        mev_main[st] = _main_mev(mev_tab)

        # Estimation window: point estimates, pooled bootstrap, LOYO.
        ew_tables[st] = ewm.ew_over_windows(exc, windows)
        ew_tables[st].to_csv(T / f"mev_estimation_window_{st}.csv",
                             index=False, float_format="%.4f")
        pooled[st], _ = ewm.bootstrap_pooled(st, windows, exc)
        pooled[st].to_csv(T / f"bootstrap_pooled_{st}.csv", index=False,
                          float_format="%.4f")
        loyo_task3 = _need(T / "loyo_summary.csv")
        gev_row = loyo_task3[(loyo_task3["station"] == st)
                             & (loyo_task3["model"] == "GEV")]
        loyo_rows.append(gev_row.drop(columns="station").assign(station=st))
        for ew in (1, 3, 5, "all"):
            row = ewm.loyo_ew_summary(ewm.loyo_ew(annual, exc, ew))
            loyo_rows.append(pd.DataFrame([row]).assign(station=st))

        # One-at-a-time sensitivity.
        long, extras = sensitivity.run_station(st, daily, annual, yearly,
                                               ew_tables[st], gev_tab, mev_tab)
        long_parts.append(long)
        tails[st] = extras["tail"]
        tails[st].to_csv(T / f"tail_check_{st}.csv", float_format="%.4f")
        valid_rules.append(extras["valid_rule"].assign(station=st))
        print(f"  {st}: Task 5 analyses in {time.time() - t0:.0f} s")

    loyo = pd.concat(loyo_rows, ignore_index=True)
    loyo["model"] = loyo["model"].replace({"MEV": "MEVD EW=1"})
    loyo.to_csv(T / "loyo_estimation_window.csv", index=False,
                float_format="%.4f")
    long = pd.concat(long_parts, ignore_index=True)
    long.to_csv(T / "sensitivity_long.csv", index=False, float_format="%.4f")
    oat = sensitivity.one_at_a_time(long)
    oat.to_csv(T / "sensitivity_summary.csv", index=False, float_format="%.3f")
    pd.concat(valid_rules).to_csv(T / "valid_year_rule.csv", index=False)

    # Recent change and station comparison.
    changes = pd.concat([comparison.change_between_windows(st, *data[st], a, b)
                         for st, a, b in CHANGE_CASES], ignore_index=True)
    changes.to_csv(T / "recent_change.csv", index=False, float_format="%.4f")
    (ann_b, y_b, _), (ann_p, y_p, _) = data["bologna"], data["padova"]
    ycorr = comparison.yearly_correlations(y_b, y_p, ann_b, ann_p)
    ycorr.to_csv(T / "station_correlations_yearly.csv", index=False,
                 float_format="%.4f")
    series = {st: summaries[st].set_index("window_end").join(
        mev_main[st].set_index("window_end")[["mean_N", "mean_C", "mean_w"]])
        for st in STATION_ORDER}
    wcorr = comparison.window_correlations(
        series["bologna"], series["padova"],
        {"mev_rl": "MEVD 200-yr level", "gev_rl": "GEV 200-yr level",
         "mean_N": "wet days N", "mean_C": "Weibull scale C",
         "mean_w": "Weibull shape w"})
    wcorr.to_csv(T / "station_correlations_windows.csv", index=False,
                 float_format="%.4f")

    # Figures.
    for k, st in enumerate(STATION_ORDER):
        _save(results_plots.plot_station_results(
            st, summaries[st], pooled[st], mev_main[st], annuals[st]),
            f"fig{21 + k}_results_{st}")
    _save(results_plots.plot_estimation_window(ew_tables, tails, loyo),
          "fig23_estimation_window")
    _save(results_plots.plot_sensitivity_summary(oat),
          "fig24_sensitivity_summary")
    _save(results_plots.plot_station_comparison(summaries, mev_main, changes,
                                                ycorr, wcorr),
          "fig25_station_comparison")
    st, end, year = REPORT_CASE
    win = [w for w in build_windows(annuals[st], "calendar") if w.end == end][0]
    cov = _need(T / "coverage_summary.csv")
    _save(results_plots.report_fig_bands(summaries, annuals), "report_fig1_bands")
    _save(results_plots.report_fig_uncertainty(
        replicates[st], end, year, np.array(win.years), loyo, cov),
        "report_fig2_uncertainty")
    _save(results_plots.report_fig_comparison(mev_main, changes),
          "report_fig3_comparison")
    save_figure(results_plots.report_equation(), "report_equation",
                formats=("png",))

    with pd.option_context("display.width", 180, "display.precision", 3,
                           "display.max_columns", 20):
        print("\nOut-of-sample check by estimation window:\n",
              loyo[["station", "model", "ks_pvalue", "exceed_10_observed",
                    "exceed_10_expected", "exceed_50_observed",
                    "exceed_50_expected", "exceed_50_binom_p"]])
        print("\nSensitivity (change of the 200-yr level vs main, mm):\n",
              oat[["station", "model", "variant", "median_change_mm",
                   "max_abs_change_mm"]])
        print("\nRecent change between disjoint windows:\n",
              changes[["station", "quantity", "first", "second", "change",
                       "rel_change", "change_p5", "change_p95", "share_le_0"]])
        print("\nBologna-Padova correlations (same year):\n", ycorr)
        print("\nBologna-Padova correlations (window series):\n", wcorr)
    print(f"\nTask 5 finished in {time.time() - t0:.0f} s.")
    if SHOW_FIGURES:
        plt.show()
    return oat


if __name__ == "__main__":
    RESULTS = main()

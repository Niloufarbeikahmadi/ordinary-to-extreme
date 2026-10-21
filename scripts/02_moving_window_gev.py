"""
Task 2 - Moving-window GEV analysis of annual maxima (Bologna, then Padova).

Run in Spyder (F5) or:  python scripts/02_moving_window_gev.py

Main configuration (task + Marani & Zanetti, 2015):
    30-calendar-year windows, step 1 year, retained if >= 20 valid years;
    GEV fitted by maximum likelihood to the annual maxima of valid years;
    200-year return level indexed by the FINAL year of each window;
    record as provided (scenario 'asis').
Sensitivity variants: L-moments estimator, QC-screened record, windows of
30 consecutive valid years.

Inputs : data/processed/{station}_annual.csv  (from Task 1)
Outputs: outputs/tables/gev_moving_windows_{station}.csv
         outputs/tables/gev_summary.csv
         outputs/figures/fig7_* ... fig11_*  (PNG 300 dpi + PDF)
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

from rainfall_extremes import config, datasets, gev_plots, moving_gev  # noqa: E402
from rainfall_extremes.plot_style import save_figure, use_style  # noqa: E402

STATION_ORDER = ("bologna", "padova")
SHOW_FIGURES = False


def _save(fig, name):
    save_figure(fig, name)
    if not SHOW_FIGURES:
        plt.close(fig)


def main() -> dict[str, pd.DataFrame]:
    use_style()
    config.ensure_output_dirs()
    results, annuals = {}, {}
    for st in STATION_ORDER:
        t0 = time.time()
        annual_long = datasets.load_annual(st)
        res = moving_gev.run_station(annual_long)
        res.to_csv(config.TABLES / f"gev_moving_windows_{st}.csv",
                   index=False, float_format="%.5f")
        results[st] = res
        annuals[st] = annual_long[annual_long["scenario"] == "asis"]
        print(f"{st}: {len(res)} fits in {time.time() - t0:.1f} s")

        _save(gev_plots.plot_parameters(res), f"fig8_gev_parameters_{st}")
        _save(gev_plots.plot_sensitivity(res), f"fig9_gev_sensitivity_{st}")
        _save(gev_plots.plot_window_fits(res, annuals[st]),
              f"fig10_gev_window_fits_{st}")

    _save(gev_plots.plot_return_level_timeseries(results, annuals),
          "fig7_gev_rl200_moving_windows")
    _save(gev_plots.plot_mz2015_reproduction(results["padova"]),
          "fig11_padova_mz2015_conventions")

    summary = moving_gev.summarise(pd.concat(results.values()))
    summary.to_csv(config.TABLES / "gev_summary.csv", index=False,
                   float_format="%.3f")
    with pd.option_context("display.width", 120, "display.precision", 2):
        print("\n", summary.T)
    if SHOW_FIGURES:
        plt.show()
    return results


if __name__ == "__main__":
    RESULTS = main()

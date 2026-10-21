"""
Task 1 - Data ingestion and quality control for Bologna and Padova.

Run in Spyder (open this file and press F5), or from a terminal:

    python scripts/01_ingest_and_qc.py

Inputs : data/raw/<the two assignment files, original names>
Outputs: data/processed/{station}_daily.csv      canonical daily table
         data/processed/{station}_annual.csv     annual table, both scenarios
         outputs/tables/{station}_qc_flags.csv   years with any flag
         outputs/tables/{station}_duplicates.csv duplicated blocks
         outputs/tables/{station}_winter_spikes.csv
         outputs/qc_report.md                    human-readable QC summary
         outputs/figures/fig1_* ... fig6_*       PNG (300 dpi) + PDF
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/`` importable when the package is not pip-installed
# (e.g. when running directly from Spyder).
try:
    _ROOT = Path(__file__).resolve().parents[1]
except NameError:                        # interactive console fallback
    _ROOT = Path.cwd()
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from rainfall_extremes import config, ingest, qc, qc_plots  # noqa: E402
from rainfall_extremes.metadata import STATIONS  # noqa: E402
from rainfall_extremes.plot_style import save_figure, use_style  # noqa: E402

STATION_ORDER = ("bologna", "padova")   # the task starts with Bologna
SHOW_FIGURES = False                    # True -> also display in Spyder


def process_station(station: str) -> qc.QCResult:
    """Read, QC and save one station; return the QC result."""
    print(f"\n=== {STATIONS[station].name} " + "=" * 50)
    daily = ingest.load_station(station)
    print(f"Read {len(daily):,} daily rows "
          f"({daily['year'].min()}-{daily['year'].max()}); "
          f"max = {daily['precip_mm'].max():.1f} mm")

    res = qc.run_qc(daily)

    # --- tables ---------------------------------------------------------
    res.daily.to_csv(config.DATA_PROCESSED / f"{station}_daily.csv",
                     index=False, float_format="%.4f")
    res.annual.to_csv(config.DATA_PROCESSED / f"{station}_annual.csv",
                      index=False, float_format="%.4f")
    asis = res.scenario("asis")
    flag_cols = [c for c in asis.columns if c.startswith("flag_")]
    flagged = asis[asis[flag_cols].any(axis=1) | (asis["n_valid"] == 0)]
    flagged[["year", "status", "qc_reason", "n_valid", "n_wet",
             "mean_wet_mm", "total_mm", "annual_max_mm", "z_nwet",
             "z_log_mean_wet", "z_total"]].to_csv(
        config.TABLES / f"{station}_qc_flags.csv", index=False,
        float_format="%.3f")
    res.duplicates.to_csv(config.TABLES / f"{station}_duplicates.csv",
                          index=False)
    res.spikes.to_csv(config.TABLES / f"{station}_winter_spikes.csv",
                      index=False)

    q = res.scenario("qc")
    print(f"Years on file: {int((asis['n_valid'] > 0).sum())} | "
          f"valid as-is: {int(asis['valid'].sum())} | "
          f"valid QC: {int(q['valid'].sum())}")
    print("Status counts:", asis["status"].value_counts().to_dict())

    # --- figures --------------------------------------------------------
    figures = {
        f"fig2_annual_diagnostics_{station}": qc_plots.plot_annual_diagnostics(res),
        f"fig3_qc_fingerprint_{station}": qc_plots.plot_qc_fingerprint(res),
        f"fig4_resolution_{station}": qc_plots.plot_resolution(res),
        f"fig5_duplicate_block_{station}": qc_plots.plot_duplicate_block(res),
        f"fig6_largest_events_{station}": qc_plots.plot_largest_events(res),
    }
    for name, fig in figures.items():
        if fig is None:
            continue
        save_figure(fig, name)
        if not SHOW_FIGURES:
            plt.close(fig)
    return res


def main() -> list[qc.QCResult]:
    use_style()
    config.ensure_output_dirs()
    results = [process_station(s) for s in STATION_ORDER]

    fig = qc_plots.plot_record_timeline(results)
    save_figure(fig, "fig1_record_timeline")
    if not SHOW_FIGURES:
        plt.close(fig)

    header = ("# Quality-control report (Task 1)\n\n"
              f"Wet day: P > {config.WET_THRESHOLD_MM:g} mm. Valid year: "
              f"< {config.MAX_MISSING_FRACTION:.0%} days missing. "
              "Scenario 'asis' = record as provided (main); scenario 'qc' = "
              "duplicated blocks masked and systematic patterns excluded "
              "(sensitivity).\n\n")
    report = header + "\n".join(qc.qc_report_markdown(r) for r in results)
    (config.OUTPUTS / "qc_report.md").write_text(report, encoding="utf-8")
    print(f"\nFigures -> {config.FIGURES}\nReport  -> "
          f"{config.OUTPUTS / 'qc_report.md'}")
    if SHOW_FIGURES:
        plt.show()
    return results


if __name__ == "__main__":
    RESULTS = main()

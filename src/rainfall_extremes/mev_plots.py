"""
Figures for the MEVD-Weibull analysis and the GEV-MEVD comparison (Task 3).

Colour rule: GEV is always blue, MEVD always orange. Within a sensitivity
panel, the main configuration is solid and the variant is dashed grey.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config, gev, mev, moving_gev, moving_mev
from .gev_plots import (_flagged_maxima, _gapped, _gumbel_variate, _T_TICKS,
                        _VARIANT)
from .metadata import STATIONS
from .plot_style import (INK, METHOD_COLORS, QC_STATUS, WIDTH_DOUBLE,
                         panel_label, shade_eras)
from .validation import binomial_interval

BLUE, ORANGE = METHOD_COLORS["GEV"], METHOD_COLORS["MEV"]
T_MAIN = int(config.RETURN_PERIOD)
RL = f"rl_{T_MAIN}"


# ----------------------------------------------------------------------
# Figure 12 - yearly ordinary-event parameters
# ----------------------------------------------------------------------
def plot_yearly_parameters(yearly: pd.DataFrame, annual: pd.DataFrame
                           ) -> plt.Figure:
    st = str(yearly["station"].iat[0])
    yp = yearly[(yearly["scenario"] == "asis")
                & (yearly["formulation"] == "excess")
                & (yearly["method"] == "pwm")]
    yp = yp.merge(annual[["year", "status"]], on="year", how="left")
    full = yp.set_index("year").reindex(
        range(int(yp["year"].min()), int(yp["year"].max()) + 1)).reset_index()
    specs = [("N", f"Wet days N\n(P > {config.WET_THRESHOLD_MM:g} mm) [d]"),
             ("C", "Weibull scale C\nof excesses [mm]"),
             ("w", "Weibull shape w [-]")]
    station = STATIONS[st]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(WIDTH_DOUBLE, 4.8))
    for i, (ax, (col, label)) in enumerate(zip(axes, specs)):
        shade_eras(ax, station.eras, label=(i == 0))
        ax.plot(full["year"], full[col], color=INK["axis"], lw=0.7, zorder=2)
        ax.plot(full["year"], full[col], ls="none", marker="o",
                markersize=2.0, color=ORANGE, zorder=3)
        run = full[col].rolling(config.WINDOW_YEARS, center=True,
                                min_periods=config.MIN_VALID_YEARS).median()
        ax.plot(full["year"], run, color=INK["primary"], lw=1.3, zorder=5)
        for key in ("review", "systematic"):
            sel = full[full["status"] == key]
            s = QC_STATUS[key]
            ax.plot(sel["year"], sel[col], ls="none", marker=s["marker"],
                    color=s["color"], markersize=3.6, markeredgewidth=0.4,
                    markeredgecolor="white", zorder=4)
        ax.set_ylabel(label)
        panel_label(ax, "abc"[i])
    axes[-1].set_xlabel("Year")
    present = [k for k in ("review", "systematic")
               if (full["status"] == k).any()]
    handles = [Line2D([], [], ls="none", marker="o", color=ORANGE,
                      markersize=3, label="Yearly estimate (PWM)"),
               Line2D([], [], color=INK["primary"], lw=1.3,
                      label="30-yr running median")]
    handles += [Line2D([], [], ls="none", marker=QC_STATUS[k]["marker"],
                       color=QC_STATUS[k]["color"], markersize=4,
                       label=QC_STATUS[k]["label"]) for k in present]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{station.name}: yearly ordinary-event parameters of the "
                 f"MEVD (N, C, w)", x=0.0, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


# ----------------------------------------------------------------------
# Figure 13 - GEV vs MEVD 200-year level (both stations): key figure
# ----------------------------------------------------------------------
def plot_gev_vs_mev(gev_res: dict, mev_res: dict, annuals: dict
                    ) -> plt.Figure:
    stations = list(mev_res)
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 2.45 * len(stations) + 0.5))
    gs = fig.add_gridspec(2 * len(stations), 1,
                          height_ratios=[4, 0.9] * len(stations), hspace=0.18)
    first = None
    for i, st in enumerate(stations):
        ax = fig.add_subplot(gs[2 * i], sharex=first)
        first = first or ax
        axn = fig.add_subplot(gs[2 * i + 1], sharex=first)
        g = _gapped(moving_gev.select(gev_res[st]))
        m = _gapped(moving_mev.select(mev_res[st]))
        _flagged_maxima(ax, annuals[st])
        ax.plot(g["window_end"], g[RL], color=BLUE, lw=1.3, zorder=4)
        ax.plot(m["window_end"], m[RL], color=ORANGE, lw=1.9, zorder=5)
        # direct end labels (supplement the legend)
        for df, name, col in ((g, "GEV", BLUE), (m, "MEVD", ORANGE)):
            last = df.dropna(subset=[RL]).iloc[-1]
            ax.annotate(name, (last["window_end"], last[RL]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=6.5,
                        color=INK["primary"])
        ax.set_ylabel("Daily rainfall [mm]")
        ax.set_ylim(0, max(np.nanmax(g[RL]) * 1.08, 180))
        panel_label(ax, "ab"[i], STATIONS[st].name)
        plt.setp(ax.get_xticklabels(), visible=False)
        mm = moving_mev.select(mev_res[st])
        axn.bar(mm["window_end"], mm["n_valid"], width=1.0, color=INK["axis"],
                lw=0)
        axn.axhline(config.MIN_VALID_YEARS, color=INK["primary"], lw=0.6)
        axn.set_ylim(0, config.WINDOW_YEARS + 2)
        axn.set_yticks([config.MIN_VALID_YEARS, config.WINDOW_YEARS])
        axn.set_ylabel("valid\nyears", fontsize=6.5)
        axn.grid(False)
        if i < len(stations) - 1:
            plt.setp(axn.get_xticklabels(), visible=False)
    axn.set_xlim(1748, 2030)
    axn.set_xlabel("Final year of the 30-year window")
    handles = [
        Line2D([], [], color=ORANGE, lw=1.9,
               label=f"MEVD-Weibull {T_MAIN}-yr level (PWM, excesses > 1 mm)"),
        Line2D([], [], color=BLUE, lw=1.3,
               label=f"GEV {T_MAIN}-yr level (ML on annual maxima)"),
        Line2D([], [], ls="none", marker="o", markersize=3, color=INK["axis"],
               label="Annual maximum (at its own year)"),
        Line2D([], [], ls="none", marker=QC_STATUS["systematic"]["marker"],
               color=QC_STATUS["systematic"]["color"], markersize=4,
               label="Annual max, systematic-pattern year"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Asymptotic vs non-asymptotic: {T_MAIN}-year daily rainfall "
                 f"in moving 30-year windows", x=0.0, ha="left", y=0.995,
                 fontsize=10, fontweight="bold")
    return fig


# ----------------------------------------------------------------------
# Figure 14 - MEVD sensitivity
# ----------------------------------------------------------------------
def plot_mev_sensitivity(res: pd.DataFrame) -> plt.Figure:
    st = str(res["station"].iat[0])
    main = moving_mev.select(res)
    variants = [
        ("Data screening", moving_mev.select(res, scenario="qc"),
         "QC-screened record", "Record as provided"),
        ("Window definition", moving_mev.select(res, mode="consecutive"),
         "30 consecutive valid years", "30 calendar years"),
        ("Threshold treatment", moving_mev.select(res, formulation="raw"),
         "Raw wet-day values (no subtraction)", "Excesses over 1 mm (+1 mm)"),
        ("Weibull estimator", moving_mev.select(res, method="ml"),
         "Maximum likelihood", "PWM"),
    ]
    fig, axes = plt.subplots(len(variants), 1, sharex=True, sharey=True,
                             figsize=(WIDTH_DOUBLE, 6.0))
    for i, (ax, (title, alt, alt_lab, ref_lab)) in enumerate(
            zip(axes, variants)):
        gm, ga = _gapped(main), _gapped(alt)
        ax.plot(gm["window_end"], gm[RL], color=ORANGE, lw=1.6, label=ref_lab)
        ax.plot(ga["window_end"], ga[RL], **_VARIANT, label=alt_lab)
        both = main.merge(alt, on="window_end", suffixes=("", "_a"))
        diff = both[RL] - both[f"{RL}_a"]
        note = (f"median diff = {diff.median():+.1f} mm, max |diff| = "
                f"{diff.abs().max():.1f} mm") if len(diff) else ""
        panel_label(ax, "abcd"[i], f"{title}  ({note})")
        ax.legend(loc="upper left", fontsize=6.3, ncol=2)
        ax.set_ylabel(f"{T_MAIN}-yr level [mm]")
    axes[-1].set_xlabel("Final year of the window")
    fig.suptitle(f"{STATIONS[st].name}: sensitivity of the MEVD {T_MAIN}-year "
                 f"level to analysis choices", x=0.0, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ----------------------------------------------------------------------
# Figure 15 - GEV vs MEVD fits in selected windows
# ----------------------------------------------------------------------
def _mev_curve(p: pd.DataFrame, Tgrid: np.ndarray) -> np.ndarray:
    return np.array([mev.mev_return_level(T, p["N"], p["C"], p["w"])
                     for T in Tgrid])


def plot_window_fits_comparison(cases: list[tuple[str, int, int]],
                                annuals: dict, yearlies: dict) -> plt.Figure:
    """``cases``: (station, window_start, window_end) triples."""
    fig, axes = plt.subplots(1, len(cases), sharey=True,
                             figsize=(WIDTH_DOUBLE, 2.7))
    Tgrid = np.logspace(np.log10(1.1), np.log10(600), 80)
    ymax = 0
    for i, (ax, (st, a0, a1)) in enumerate(zip(np.atleast_1d(axes), cases)):
        an = annuals[st]
        av = an[an["valid"] & an["year"].between(a0, a1)]
        x = np.sort(av["annual_max_mm"].to_numpy())
        yp = yearlies[st]
        yp = yp[(yp["scenario"] == "asis") & (yp["formulation"] == "excess")
                & (yp["method"] == "pwm")].set_index("year").loc[av["year"]]
        g = gev.fit_ml(x)
        g_curve = gev.return_level(Tgrid, g)
        m_curve = _mev_curve(yp, Tgrid)
        ax.axvspan(_gumbel_variate(x.size + 1), _gumbel_variate(600),
                   color=INK["band"], lw=0, zorder=0)
        ax.plot(_gumbel_variate(Tgrid), g_curve, color=BLUE, lw=1.3,
                label="GEV (ML)")
        ax.plot(_gumbel_variate(Tgrid), m_curve, color=ORANGE, lw=1.8,
                label="MEVD-Weibull")
        F = gev.weibull_plotting_positions(x.size)
        ax.plot(-np.log(-np.log(F)), x, ls="none", marker="o",
                markersize=3.0, color=INK["secondary"], zorder=4,
                label="Annual maxima")
        ax.axvline(_gumbel_variate(T_MAIN), color=INK["primary"], lw=0.6)
        rl_g = float(gev.return_level(T_MAIN, g))
        rl_m = mev.mev_return_level(T_MAIN, yp["N"], yp["C"], yp["w"])
        for val, col, dy in ((rl_g, BLUE, -10), (rl_m, ORANGE, 4)):
            ax.plot(_gumbel_variate(T_MAIN), val, marker="o", color=col,
                    markersize=5, mec="white", mew=1, zorder=6)
            ax.annotate(f"{val:.0f}", (_gumbel_variate(T_MAIN), val),
                        xytext=(4, dy), textcoords="offset points",
                        fontsize=6.5)
        ymax = max(ymax, x.max(), min(rl_g, 400), rl_m)
        ax.set_xticks(_gumbel_variate(_T_TICKS))
        ax.set_xticklabels([str(t) for t in _T_TICKS])
        ax.set_xlabel("Return period T [years]")
        panel_label(ax, "abc"[i], f"{STATIONS[st].name} {a0}-{a1}  "
                                  f"(xi={g.xi:+.2f})")
    axes[0].set_ylabel("Daily rainfall [mm]")
    axes[0].set_ylim(0, ymax * 1.12)
    axes[0].legend(loc="upper left", fontsize=6.2)
    fig.suptitle("GEV and MEVD in selected windows: the effect of a single "
                 "year (Gumbel paper)", x=0.0, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ----------------------------------------------------------------------
# Figure 16 - out-of-sample (leave-one-year-out) calibration
# ----------------------------------------------------------------------
def plot_loyo(tables: dict, summaries: dict) -> plt.Figure:
    stations = list(tables)
    fig, axes = plt.subplots(len(stations), 2, figsize=(WIDTH_DOUBLE, 2.5 *
                             len(stations) + 0.4),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    axes = np.atleast_2d(axes)
    for r, st in enumerate(stations):
        tab, summ = tables[st], summaries[st].set_index("model")
        n = len(tab)
        # (left) PIT QQ against the uniform, with a KS 95% band
        ax = axes[r, 0]
        q = (np.arange(1, n + 1) - 0.5) / n
        band = 1.36 / np.sqrt(n)
        ax.fill_between([0, 1], [-band, 1 - band], [band, 1 + band],
                        color=INK["band"], lw=0)
        ax.plot([0, 1], [0, 1], color=INK["primary"], lw=0.6)
        for m, col, lw in (("gev", BLUE, 1.3), ("mev", ORANGE, 1.8)):
            ax.plot(q, np.sort(tab[f"u_{m}"]), color=col, lw=lw,
                    label=f"{m.upper().replace('MEV', 'MEVD')} (KS p = "
                          f"{summ.at[m.upper(), 'ks_pvalue']:.2f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Uniform quantile")
        ax.set_ylabel("Sorted PIT value")
        ax.legend(loc="upper left", fontsize=6)
        ax.grid(True, axis="both")
        panel_label(ax, "ac"[r], f"{STATIONS[st].name}: calibration "
                                 f"(n = {n} years)")
        # (right) exceedances of the 10- and 50-year levels
        ax = axes[r, 1]
        xs, labels = [], []
        for k, T in enumerate((10, 50)):
            lo, hi = binomial_interval(n, 1 / T)
            x0 = k * 3
            ax.add_patch(plt.Rectangle((x0 - 0.5, lo), 2.0, hi - lo,
                                       color=INK["band"], lw=0, zorder=0))
            ax.plot([x0 - 0.5, x0 + 1.5], [n / T] * 2, color=INK["primary"],
                    lw=0.8, zorder=1)
            for j, (m, col) in enumerate((("GEV", BLUE), ("MEV", ORANGE))):
                obs = summ.at[m, f"exceed_{T}_observed"]
                ax.bar(x0 + j, obs, width=0.6, color=col, zorder=2)
                ax.text(x0 + j, obs + 0.4, f"{obs:d}", ha="center",
                        va="bottom", fontsize=6.5)
            xs += [x0, x0 + 1]
            labels += [f"GEV\n{T}-yr", f"MEVD\n{T}-yr"]
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel("Years exceeding the level")
        panel_label(ax, "bd"[r], "exceedances (line: expected, "
                                 "band: binomial 90%)")
    fig.suptitle("Out-of-sample check: each year predicted from the other "
                 "years of its 30-year window", x=0.0, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig

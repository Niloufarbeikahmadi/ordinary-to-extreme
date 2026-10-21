"""
Figures of Task 5 (station results, sensitivity, comparison) and the compact
figures of the final report.

Colour rule as in every task: GEV blue, MEVD orange, the pooled-Weibull
MEVD (estimation window = whole window) aqua. Quantities that are not
methods (wet days N, Weibull scale C and shape w, stations) are drawn in
ink tones and told apart by line style, marker and direct labels.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config
from .bootstrap_plots import _end_labels
from .gev_plots import _flagged_maxima, _gapped
from .metadata import STATIONS
from .plot_style import (INK, METHOD_COLORS, STATION_LINESTYLES, WIDTH_DOUBLE,
                         panel_label, shade_eras)

BLUE, ORANGE, AQUA = (METHOD_COLORS[k] for k in ("GEV", "MEV", "REF"))
T_MAIN = int(config.RETURN_PERIOD)
LO, HI = (f"p{p:g}" for p in config.CI_PERCENTILES)
XLIM = (1748, 2030)
_MARK = {"bologna": "o", "padova": "s"}
_DRIVER_STYLE = {"C": dict(color=INK["primary"], ls="-", lw=1.3),
                 "N": dict(color=INK["secondary"], ls=(0, (4, 1.5)), lw=1.1),
                 "w": dict(color=INK["muted"], ls=(0, (1, 1.2)), lw=1.3)}


def _offscale_carets(ax, x, y, ymax, color):
    off = y > ymax
    if off.any():
        ax.plot(x[off], np.full(off.sum(), ymax * 0.985), ls="none",
                marker="^", markersize=2.4, color=color, clip_on=False,
                zorder=6)
    return int(off.sum())


def _bands(ax, s, ymax, pooled=None, alpha_gev=0.15, alpha_mev=0.38):
    """GEV and MEVD point estimates with 90% bands on one axis."""
    x = s["window_end"]
    ax.fill_between(x, s[f"gev_{LO}"], s[f"gev_{HI}"].clip(upper=ymax),
                    color=BLUE, alpha=alpha_gev, lw=0, zorder=1)
    ax.fill_between(x, s[f"mev_{LO}"], s[f"mev_{HI}"], color=ORANGE,
                    alpha=alpha_mev, lw=0, zorder=3)
    ax.plot(x, s["gev_rl"], color=BLUE, lw=1.1, zorder=4)
    ax.plot(x, s["mev_rl"], color=ORANGE, lw=1.6, zorder=5)
    if pooled is not None:
        p = _gapped(pooled.sort_values("window_end"))
        ax.plot(p["window_end"], p["mevp_rl"], color=AQUA, lw=1.1,
                ls=(0, (4, 1.5)), zorder=5)
    return _offscale_carets(ax, x.to_numpy(), s[f"gev_{HI}"].to_numpy(),
                            ymax, BLUE)


# ----------------------------------------------------------------------
# Figures 21-22 - results of one station (task: Bologna first, then Padova)
# ----------------------------------------------------------------------
def plot_station_results(station: str, summ: pd.DataFrame,
                         pooled: pd.DataFrame, mev_main: pd.DataFrame,
                         annual: pd.DataFrame, ymax: float = 420.0
                         ) -> plt.Figure:
    """(a) 200-year levels with 90% bands, (b) MEVD drivers, (c) GEV shape."""
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 6.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 1.7, 1.5], hspace=0.28,
                          top=0.9, bottom=0.12)
    ax = fig.add_subplot(gs[0])
    axd = fig.add_subplot(gs[1], sharex=ax)
    axx = fig.add_subplot(gs[2], sharex=ax)
    s = _gapped(summ.sort_values("window_end"))
    eras = STATIONS[station].eras
    for a in (ax, axd, axx):
        if eras:
            shade_eras(a, eras, label=(a is ax))
    _flagged_maxima(ax, annual)
    n_off = _bands(ax, s, ymax, pooled)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(f"{T_MAIN}-year daily rainfall [mm]")
    last = s.dropna(subset=["mev_rl"]).iloc[-1]
    note = (f"last window {int(last['window_start'])}-{int(last['window_end'])}:"
            f"  MEVD {last['mev_rl']:.0f} [{last[f'mev_{LO}']:.0f}-"
            f"{last[f'mev_{HI}']:.0f}] mm,  GEV {last['gev_rl']:.0f} "
            f"[{last[f'gev_{LO}']:.0f}-{last[f'gev_{HI}']:.0f}] mm")
    ax.text(0.0, 0.97, note, transform=ax.transAxes, fontsize=6.5,
            color=INK["secondary"], va="top")
    if n_off:
        ax.text(1.0, 0.97, f"GEV 95th percentile above {ymax:.0f} mm in "
                f"{n_off} windows", transform=ax.transAxes, ha="right",
                va="top", fontsize=6, color=INK["muted"])
    panel_label(ax, "a", f"{STATIONS[station].name}: {T_MAIN}-year level "
                "with 90% bootstrap bands (500 resamplings of years)",
                pad=13 if eras else 4)
    plt.setp(ax.get_xticklabels(), visible=False)

    # (b) drivers of the MEVD: window means relative to their median
    m = _gapped(mev_main.sort_values("window_end"))
    lim = 40.0
    ends = []
    for col, key, lab in (("mean_C", "C", "C"), ("mean_N", "N", "N"),
                          ("mean_w", "w", "w")):
        rel = 100 * (m[col] / m[col].median() - 1)
        axd.plot(m["window_end"], rel.clip(-lim * 1.2, lim * 1.2),
                 **_DRIVER_STYLE[key])
        ends.append((float(rel.dropna().iloc[-1]), lab))
        if (rel.abs() > lim).any():
            off = m.loc[rel.abs() > lim, "window_end"]
            if lab == "C":
                axd.text(off.mean(), -lim * 0.92, f"off scale {off.min():.0f}-"
                         f"{off.max():.0f}: accumulated readings (C up, N down)",
                         fontsize=5.6, color=INK["muted"], ha="center",
                         va="bottom", bbox=dict(fc="white", ec="none",
                                                alpha=0.85, pad=0.6))
    _end_labels(axd, m["window_end"].dropna().iloc[-1], ends, min_gap=12)
    axd.axhline(0, color=INK["axis"], lw=0.6)
    axd.set_ylabel("window mean vs\nmedian [%]", fontsize=6.5)
    axd.set_ylim(-lim, lim)
    panel_label(axd, "b", "what moves the MEVD: yearly wet days N and "
                "Weibull parameters C, w (30-year means)")
    plt.setp(axd.get_xticklabels(), visible=False)

    # (c) GEV shape with its bootstrap band
    axx.fill_between(s["window_end"], s[f"gev_xi_{LO}"], s[f"gev_xi_{HI}"],
                     color=BLUE, alpha=0.15, lw=0)
    axx.plot(s["window_end"], s["gev_xi_point"], color=BLUE, lw=1.0)
    axx.axhline(0, color=INK["axis"], lw=0.6)
    axx.axhline(-0.5, color=INK["muted"], lw=0.6, ls=(0, (2, 2)))
    axx.text(XLIM[0] + 1, -0.5, "ML irregular below -0.5", fontsize=5.8,
             color=INK["muted"], va="bottom")
    axx.set_ylim(-1.05, 1.05)
    axx.set_ylabel("GEV shape ξ", fontsize=6.5)
    panel_label(axx, "c", "why the GEV jumps: shape parameter and its 90% "
                "bootstrap band (ξ > 0 heavy tail, ξ < 0 bounded tail)")
    axx.set_xlim(*XLIM)
    axx.set_xlabel("Final year of the 30-year window")
    handles = [
        (Line2D([], [], color=ORANGE, lw=1.6), "MEVD (yearly Weibull, main)"),
        (Patch(color=ORANGE, alpha=0.38, lw=0), "MEVD 90% band"),
        (Line2D([], [], color=AQUA, lw=1.1, ls=(0, (4, 1.5))),
         "MEVD, one Weibull per window"),
        (Line2D([], [], color=BLUE, lw=1.1), "GEV (ML)"),
        (Patch(color=BLUE, alpha=0.15, lw=0), "GEV 90% band"),
        (Line2D([], [], ls="none", marker="o", ms=2, color=INK["axis"]),
         "annual maxima"),
    ]
    fig.legend(*zip(*handles), loc="lower center", ncol=6, fontsize=6,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=1.0)
    return fig


# ----------------------------------------------------------------------
# Figure 23 - MEVD estimation window
# ----------------------------------------------------------------------
_EW_STYLE = {"1": dict(color=ORANGE, lw=1.6, label="EW = 1 year (main)"),
             "3": dict(color=ORANGE, lw=1.0, alpha=0.65, ls=(0, (4, 1.5)),
                       label="EW = 3 years"),
             "5": dict(color=ORANGE, lw=1.0, alpha=0.45, ls=(0, (1, 1.2)),
                       label="EW = 5 years"),
             "all": dict(color=AQUA, lw=1.3, label="EW = whole window")}


def plot_estimation_window(ew_tables: dict[str, pd.DataFrame],
                           tails: dict[str, pd.DataFrame],
                           loyo: pd.DataFrame) -> plt.Figure:
    """(a-b) 200-yr MEVD for several estimation windows, (c) out-of-sample
    exceedances of the 50-year level, (d) tail vs bulk Weibull shape."""
    stations = list(ew_tables)
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.6, 1],
                          hspace=0.42, wspace=0.28, top=0.9, bottom=0.14)
    for i, st in enumerate(stations):
        ax = fig.add_subplot(gs[i, 0])
        t = ew_tables[st]
        for ew, sty in _EW_STYLE.items():
            g = _gapped(t[t["ew"] == ew].sort_values("window_end"))
            ax.plot(g["window_end"], g[f"rl_{T_MAIN}"], **sty)
        tl = _gapped(tails[st].reset_index().sort_values("window_end"))
        ax.plot(tl["window_end"], tl[f"rl_{T_MAIN}"], color=INK["secondary"],
                lw=0.9, ls=(0, (5, 1.5, 1, 1.5)),
                label="one Weibull per window, fitted to the top 25%")
        ax.set_xlim(*XLIM)
        ax.set_ylim(80, 200)
        ax.set_ylabel(f"MEVD {T_MAIN}-yr [mm]")
        panel_label(ax, "ab"[i], f"{STATIONS[st].name}: estimation window (EW)")
        if i == 0:
            plt.setp(ax.get_xticklabels(), visible=False)
            handles, labels = ax.get_legend_handles_labels()
        else:
            ax.set_xlabel("Final year of the 30-year window")
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6,
               bbox_to_anchor=(0.5, 0.0))

    # (c) out-of-sample exceedances of the 50-year level
    axc = fig.add_subplot(gs[0, 1])
    models = ["GEV", "MEVD EW=1", "MEVD EW=3", "MEVD EW=5", "MEVD EW=all"]
    cols = [BLUE, ORANGE, ORANGE, ORANGE, AQUA]
    for k, st in enumerate(stations):
        g = loyo[loyo["station"] == st].set_index("model")
        exp = g["exceed_50_expected"].iat[0]
        n = g["n_years"].iat[0]
        from .validation import binomial_interval
        lo_, hi_ = binomial_interval(n, 1 / 50)
        y0 = k * (len(models) + 1.2)
        for j, (mname, col) in enumerate(zip(models, cols)):
            obs = g.loc[mname, "exceed_50_observed"]
            axc.plot(obs, y0 + j, marker=_MARK[st], color=col, ms=4,
                     ls="none")
        axc.plot([lo_, hi_], [y0 - 0.7] * 2, color=INK["secondary"], lw=2.0,
                 solid_capstyle="butt")
        axc.plot([exp], [y0 - 0.7], marker="|", color=INK["primary"], ms=7)
        axc.text(hi_ + 0.3, y0 - 0.7, f"{STATIONS[st].name}: expected "
                 f"{exp:.1f}, 90% range", fontsize=5.6, va="center",
                 color=INK["secondary"])
    yt = [k * (len(models) + 1.2) + j for k in range(len(stations))
          for j in range(len(models))]
    axc.set_yticks(yt)
    axc.set_yticklabels(models * len(stations), fontsize=6)
    axc.invert_yaxis()
    axc.grid(True, axis="x")
    axc.grid(False, axis="y")
    axc.set_xlabel("Years exceeding the 50-yr level (left out)")
    panel_label(axc, "c", "out-of-sample check")

    # (d) tail versus bulk shape
    axd = fig.add_subplot(gs[1, 1])
    for st in stations:
        tl = tails[st]
        axd.plot(tl["w_pwm"], tl["w_tail"], ls="none", marker=_MARK[st],
                 ms=2.4, mfc="none", mew=0.5,
                 color=INK["primary"] if st == stations[0] else INK["muted"],
                 label=STATIONS[st].name)
    lims = [0.6, 1.1]
    axd.plot(lims, lims, color=INK["axis"], lw=0.8)
    axd.set_xlim(*lims)
    axd.set_ylim(*lims)
    axd.text(0.97, 0.05, "below the line: tail heavier\nthan the bulk; above: lighter",
             transform=axd.transAxes, ha="right", fontsize=5.8,
             color=INK["muted"])
    axd.set_xlabel("w, all wet days (PWM)")
    axd.set_ylabel("w, top 25% (censored ML)")
    axd.legend(loc="upper left", fontsize=6, handletextpad=0.2)
    axd.grid(True)
    panel_label(axd, "d", "one Weibull for bulk and tail?")
    fig.suptitle("MEVD estimation window: noise of yearly fits vs real "
                 "year-to-year differences", x=0.0, ha="left", y=0.995,
                 fontsize=10, fontweight="bold")
    return fig


# ----------------------------------------------------------------------
# Figure 24 - one-at-a-time sensitivity summary
# ----------------------------------------------------------------------
def plot_sensitivity_summary(oat: pd.DataFrame) -> plt.Figure:
    """Forest plot: change of the 200-yr level for every analysis choice."""
    stations = list(dict.fromkeys(oat["station"]))
    order = (oat.groupby(["model", "variant"])["median_abs_change_mm"].max()
             .reset_index().sort_values(["model", "median_abs_change_mm"],
                                        ascending=[True, False]))
    rows = list(zip(order["model"], order["variant"]))
    fig, axes = plt.subplots(1, len(stations), sharey=True,
                             figsize=(WIDTH_DOUBLE, 0.16 * len(rows) + 1.2))
    for ax, st in zip(np.atleast_1d(axes), stations):
        g = oat[oat["station"] == st].set_index(["model", "variant"])
        for y, key in enumerate(rows):
            if key not in g.index:
                ax.text(0, y, "Padova only", fontsize=5.5, color=INK["muted"],
                        va="center", ha="center",
                        bbox=dict(fc="white", ec="none", pad=0.6))
                continue
            r = g.loc[key]
            col = BLUE if key[0] == "GEV" else ORANGE
            lo_, hi_ = r["p05_change_mm"], r["p95_change_mm"]
            ax.plot([max(lo_, -80), min(hi_, 80)], [y, y], color=col, lw=1.6,
                    alpha=0.55, solid_capstyle="butt")
            ax.plot(r["median_change_mm"], y, marker="o", color=col, ms=3.2)
            if hi_ > 80:
                ax.plot(80, y, marker=">", ms=3.5, color=col)
            if lo_ < -80:
                ax.plot(-80, y, marker="<", ms=3.5, color=col)
        ax.axvline(0, color=INK["primary"], lw=0.7)
        ax.set_xlim(-82, 82)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.set_xlabel(f"Change of the {T_MAIN}-yr level vs main [mm]")
        panel_label(ax, "ab"[stations.index(st)], STATIONS[st].name)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(rows)))
    ax0.set_yticklabels([f"{m.replace('MEV', 'MEVD')}: {v}" for m, v in rows],
                        fontsize=6)
    ax0.invert_yaxis()
    fig.legend([Line2D([], [], marker="o", color=INK["secondary"], ls="none",
                       ms=3.2),
                Line2D([], [], color=INK["secondary"], lw=1.6, alpha=0.55)],
               ["median change over windows", "5th-95th percentile over windows"],
               loc="lower center", ncol=2, fontsize=6.5,
               bbox_to_anchor=(0.55, -0.02))
    fig.suptitle("Sensitivity: one analysis choice changed at a time",
                 x=0.0, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


# ----------------------------------------------------------------------
# Figure 25 - Bologna vs Padova
# ----------------------------------------------------------------------
_CHANGE_ROWS = ["GEV", "MEVD (EW = 1 yr)", "MEVD (EW = window)",
                "mean N [days]", "mean C [mm]"]
_CHANGE_COLORS = {"GEV": BLUE, "MEVD (EW = 1 yr)": ORANGE,
                  "MEVD (EW = window)": AQUA, "mean N [days]": INK["secondary"],
                  "mean C [mm]": INK["primary"]}


#: Padova windows containing the 1823-1837 accumulated readings (Task 1).
_ACCUM_WINDOWS = (1823, 1866)
_CN_LIMITS = {"mean_C": (6.0, 11.5), "mean_N": (55.0, 93.0)}


def _cn_panel(ax, mev_main: dict[str, pd.DataFrame], col: str,
              note: bool = True) -> None:
    """30-year means of C or N for both stations, artefact era shaded."""
    ax.axvspan(*_ACCUM_WINDOWS, color=INK["band"], lw=0, zorder=0)
    for st in mev_main:
        m = _gapped(mev_main[st].sort_values("window_end"))
        ax.plot(m["window_end"], m[col], color=INK["primary"], lw=1.0,
                ls=STATION_LINESTYLES[st], label=STATIONS[st].name)
    ax.set_ylim(*_CN_LIMITS[col])
    if note:
        ax.text(sum(_ACCUM_WINDOWS) / 2, 0.04, "Padova 1823-37:\naccumulated\n"
                "readings", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=5.4, color=INK["muted"], linespacing=1.0,
                bbox=dict(fc=INK["band"], ec="none", pad=0.5))


def _change_forest(ax, changes: pd.DataFrame, clip: float = 60.0) -> None:
    cases = list(dict.fromkeys(zip(changes["station"], changes["first"],
                                   changes["second"])))
    y, ticks, labels = 0, [], []
    for st, a, b in cases:
        g = changes[(changes["station"] == st) & (changes["first"] == a)
                    & (changes["second"] == b)].set_index("quantity")
        ax.text(0.015, y - 1.0, f"{STATIONS[st].name}  {a} → {b}",
                transform=ax.get_yaxis_transform(), fontsize=6.2,
                fontweight="bold", color=INK["primary"], va="center",
                ha="left", zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.8))
        for q in _CHANGE_ROWS:
            r = g.loc[q]
            base = r["first_value"]
            lo_ = 100 * r[f"change_{LO}"] / base
            hi_ = 100 * r[f"change_{HI}"] / base
            mid = 100 * r["rel_change"]
            col = _CHANGE_COLORS[q]
            ax.plot([max(lo_, -clip), min(hi_, clip)], [y, y], color=col,
                    lw=1.8, alpha=0.6, solid_capstyle="butt")
            ax.plot(np.clip(mid, -clip, clip), y, marker="o", ms=3, color=col)
            if hi_ > clip:
                ax.plot(clip, y, marker=">", ms=3.5, color=col, clip_on=False)
            if lo_ < -clip:
                ax.plot(-clip, y, marker="<", ms=3.5, color=col, clip_on=False)
            ticks.append(y)
            labels.append(q)
            y += 1
        y += 1.9
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(y - 1.5, -1.8)
    ax.axvline(0, color=INK["primary"], lw=0.7)
    ax.set_xlim(-clip, clip)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_xlabel("Change between disjoint 30-yr windows (90% bootstrap)")


def plot_station_comparison(summaries: dict[str, pd.DataFrame],
                            mev_main: dict[str, pd.DataFrame],
                            changes: pd.DataFrame, ycorr: pd.DataFrame,
                            wcorr: pd.DataFrame) -> plt.Figure:
    stations = list(summaries)
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 6.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], width_ratios=[1.45, 1],
                          hspace=0.5, wspace=0.42, top=0.92, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    for st in stations:
        s = _gapped(summaries[st].sort_values("window_end"))
        ls = STATION_LINESTYLES[st]
        ax.fill_between(s["window_end"], s[f"mev_{LO}"], s[f"mev_{HI}"],
                        color=ORANGE, alpha=0.22, lw=0)
        ax.plot(s["window_end"], s["mev_rl"], color=ORANGE, lw=1.4, ls=ls)
        last = s.dropna(subset=["mev_rl"]).iloc[-1]
        ax.annotate(STATIONS[st].name, (last["window_end"], last["mev_rl"]),
                    xytext=(3, 0), textcoords="offset points", fontsize=6.5,
                    va="center", annotation_clip=False)
    ax.set_xlim(*XLIM)
    ax.set_ylim(90, 210)
    ax.set_ylabel(f"MEVD {T_MAIN}-yr [mm]")
    panel_label(ax, "a", "MEVD 200-year level with 90% bands")
    plt.setp(ax.get_xticklabels(), visible=False)
    for row, (col, lab, unit) in enumerate((("mean_C", "c", "C [mm]"),
                                            ("mean_N", "d", "N [days]")), 1):
        a2 = fig.add_subplot(gs[row, 0], sharex=ax)
        _cn_panel(a2, mev_main, col, note=(row == 1))
        a2.set_ylabel(unit)
        title = ("wet-day intensity: Weibull scale C (30-year mean)"
                 if col == "mean_C" else "wet days per year N (30-year mean)")
        panel_label(a2, lab, title)
        if row == 1:
            plt.setp(a2.get_xticklabels(), visible=False)
            a2.legend(loc="upper right", fontsize=6, ncol=2)
        else:
            a2.set_xlabel("Final year of the 30-year window")
    axb = fig.add_subplot(gs[0, 1])
    yc = ycorr.iloc[::-1]
    axb.barh(range(len(yc)), yc["spearman_r"], color=INK["axis"], height=0.6)
    for k, (r, p) in enumerate(zip(yc["spearman_r"], yc["p_value"])):
        axb.text(r + 0.02, k, f"{r:.2f}", va="center", fontsize=6,
                 color=INK["secondary"])
    axb.set_yticks(range(len(yc)))
    axb.set_yticklabels(yc["quantity"], fontsize=6)
    axb.set_xlim(0, 0.8)
    axb.grid(True, axis="x")
    axb.grid(False, axis="y")
    axb.set_xlabel(f"Rank correlation, same year ({int(yc['n_years'].iat[0])}"
                   " common years)")
    panel_label(axb, "b", "shared year to year?")
    axf = fig.add_subplot(gs[1:, 1])
    _change_forest(axf, changes)
    panel_label(axf, "e", "recent change: shared or local?")
    fig.suptitle("Bologna and Padova: wet-day occurrence and intensity change "
                 "together, extremes much less", x=0.0, ha="left", y=0.995,
                 fontsize=10, fontweight="bold")
    return fig


# ----------------------------------------------------------------------
# Compact figures for the 4-page report
# ----------------------------------------------------------------------
def report_fig_bands(summaries: dict[str, pd.DataFrame],
                     annuals: dict[str, pd.DataFrame],
                     ymax: float = 360.0) -> plt.Figure:
    """Report Fig. 1: 200-yr level with 90% bands, Bologna then Padova."""
    stations = list(summaries)
    fig, axes = plt.subplots(len(stations), 1, sharex=True,
                             figsize=(WIDTH_DOUBLE, 3.9))
    for i, (ax, st) in enumerate(zip(axes, stations)):
        s = _gapped(summaries[st].sort_values("window_end"))
        if STATIONS[st].eras:
            shade_eras(ax, STATIONS[st].eras, label=False)
        _flagged_maxima(ax, annuals[st])
        n_off = _bands(ax, s, ymax)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f"{T_MAIN}-yr depth [mm]")
        panel_label(ax, "ab"[i], STATIONS[st].name)
        if n_off:
            ax.text(1.0, 1.0, f"GEV 95th pct. > {ymax:.0f} mm in {n_off} "
                    "windows (▲)", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=6, color=INK["muted"])
    axes[-1].set_xlim(*XLIM)
    axes[-1].set_xlabel("Final year of the 30-year window")
    handles = [(Line2D([], [], color=ORANGE, lw=1.6), "MEVD-Weibull"),
               (Patch(color=ORANGE, alpha=0.38, lw=0), "MEVD 90% band"),
               (Line2D([], [], color=BLUE, lw=1.1), "GEV (ML)"),
               (Patch(color=BLUE, alpha=0.15, lw=0), "GEV 90% band"),
               (Line2D([], [], ls="none", marker="o", ms=2, color=INK["axis"]),
                "annual maxima")]
    fig.legend(*zip(*handles), loc="lower center", ncol=5, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=0.6)
    return fig


def report_fig_uncertainty(rep_bologna: dict, case_end: int, record_year: int,
                           window_years: np.ndarray, loyo: pd.DataFrame,
                           cov_summary: pd.DataFrame) -> plt.Figure:
    """Report Fig. 2: (a) one year shapes the GEV bootstrap, (b) out-of-
    sample exceedances, (c) coverage of the 90% bands (synthetic)."""
    from .bootstrap import resample_indices
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE, 2.35),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    j = int(np.flatnonzero(rep_bologna["window_end"] == case_end)[0])
    g, m = rep_bologna["rl_gev"][j], rep_bologna["rl_mev"][j]
    n = window_years.size
    idx = resample_indices(n, g.size, "bologna", case_end)
    k = np.minimum((idx == int(np.flatnonzero(window_years == record_year)[0]))
                   .sum(axis=1), 3)
    for kk in range(4):
        sel = k == kk
        y = 3 - kk
        ax.plot(np.clip(g[sel], 0, 400), y + 0.13 + np.random.default_rng(kk)
                .uniform(-0.08, 0.08, sel.sum()), ls="none", marker="o",
                ms=1.3, color=BLUE, alpha=0.5)
        ax.plot(m[sel], y - 0.13 + np.random.default_rng(kk + 9)
                .uniform(-0.08, 0.08, sel.sum()), ls="none", marker="o",
                ms=1.3, color=ORANGE, alpha=0.5)
        ax.text(405, y, f"{sel.mean():.0%}", fontsize=5.8, va="center",
                color=INK["muted"])
    ax.set_yticks(range(4))
    ax.set_yticklabels(["3+", "2", "1", "0"])
    ax.set_ylabel(f"copies of {record_year} drawn")
    ax.set_xlim(40, 400)
    ax.set_xlabel(f"Bootstrap {T_MAIN}-yr level [mm]")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    panel_label(ax, "a", f"Bologna {case_end - 29}-{case_end}")
    # (b) LOYO
    ax = axes[1]
    models = [("GEV", BLUE), ("MEVD EW=1", ORANGE), ("MEVD EW=all", AQUA)]
    stations = list(dict.fromkeys(loyo["station"]))
    for si, st in enumerate(stations):
        gg = loyo[loyo["station"] == st].set_index("model")
        for mi, (mname, col) in enumerate(models):
            for ti, T in enumerate((10, 50)):
                ratio = (gg.loc[mname, f"exceed_{T}_observed"]
                         / gg.loc[mname, f"exceed_{T}_expected"])
                x = ti * 4 + mi + 0.25 * (si - 0.5)
                ax.plot(x, ratio, marker=_MARK[st], color=col, ms=3.5,
                        ls="none")
                if gg.loc[mname, f"exceed_{T}_binom_p"] < 0.05:
                    ax.plot(x, ratio, marker="o", ms=8, mfc="none",
                            mec=INK["primary"], mew=0.6, ls="none")
    ax.axhline(1, color=INK["primary"], lw=0.7)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(["10-yr level", "50-yr level"])
    ax.set_ylabel("observed / expected\nexceedances")
    ax.set_ylim(0, 3.9)
    ax.legend([Line2D([], [], marker=_MARK[s], color=INK["secondary"],
                      ls="none", ms=3.5) for s in stations]
              + [Patch(color=c) for _, c in models]
              + [Line2D([], [], marker="o", ms=7, mfc="none",
                        mec=INK["primary"], mew=0.6, ls="none")],
              [STATIONS[s].name for s in stations]
              + ["GEV", "MEVD", "MEVD, EW = window"] + ["binomial p < 0.05"],
              fontsize=5.4, loc="upper left", handletextpad=0.3, ncol=2,
              columnspacing=0.6, handlelength=1.0)
    panel_label(ax, "b", "leave-one-year-out")
    # (c) coverage
    ax = axes[2]
    rows = [("gev", "GEV", "GEV world", BLUE),
            ("ordinary", "GEV", "ordinary-day world", BLUE),
            ("ordinary", "MEV", "ordinary-day world", ORANGE)]
    for ri, (world, model, wl, col) in enumerate(rows):
        for si, st in enumerate(stations):
            r = cov_summary[(cov_summary["world"] == world)
                            & (cov_summary["model"] == model)
                            & (cov_summary["station"] == st)].iloc[0]
            ax.errorbar(r["coverage"], ri + 0.18 * (si - 0.5),
                        xerr=1.96 * r["coverage_se"], fmt=_MARK[st],
                        color=col, ms=3.5, elinewidth=0.9)
    ax.axvline(0.9, color=INK["primary"], lw=0.7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{m.replace('MEV', 'MEVD')}, {w}" for _, m, w, _ in rows],
                       fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0.6, 0.95)
    ax.set_xlabel("coverage of nominal 90% band")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    panel_label(ax, "c", "synthetic test")
    fig.tight_layout(w_pad=1.2)
    return fig


def report_fig_comparison(mev_main: dict[str, pd.DataFrame],
                          changes: pd.DataFrame) -> plt.Figure:
    """Report Fig. 3: (a) wet-day intensity C, (b) wet days N, (c) changes."""
    stations = list(mev_main)
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 2.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.25,
                          wspace=0.55)
    first = None
    for row, (col, unit, title) in enumerate(
            (("mean_C", "C [mm]", "wet-day intensity C"),
             ("mean_N", "N [days]", "wet days per year N"))):
        ax = fig.add_subplot(gs[row, 0], sharex=first)
        first = first or ax
        _cn_panel(ax, mev_main, col, note=(row == 0))
        ax.set_ylabel(unit)
        panel_label(ax, "ab"[row], f"{title} (30-yr means)")
        if row == 0:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.legend(loc="upper right", fontsize=6, ncol=2)
        else:
            ax.set_xlabel("Final year of the 30-year window")
    first.set_xlim(*XLIM)
    axf = fig.add_subplot(gs[:, 1])
    sub = changes[changes["quantity"].isin(_CHANGE_ROWS)]
    _change_forest(axf, sub)
    axf.set_xlabel("Change, 90% bootstrap interval")
    panel_label(axf, "c", "disjoint 30-yr windows")
    return fig


def report_equation() -> plt.Figure:
    """The MEVD equation as an image for the Word report (mathtext)."""
    fig = plt.figure(figsize=(3.4, 0.6))
    fig.text(0.5, 0.5, r"$\zeta(x)\ =\ \frac{1}{n}\sum_{j=1}^{n}\left[1-\exp"
             r"\left\{-\left(\frac{x-\theta}{C_j}\right)^{w_j}\right\}"
             r"\right]^{N_j}$", ha="center", va="center", fontsize=13,
             color=INK["primary"])
    return fig

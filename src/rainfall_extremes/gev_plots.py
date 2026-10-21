"""
Figures for the moving-window GEV analysis (Task 2).

Colour rule: GEV is always blue. The main configuration (ML, record as
provided, calendar windows) is a solid blue line. A sensitivity variant is a
dashed grey line, so the eye always finds the main result first.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import config, gev
from .metadata import STATIONS
from .moving_gev import select, window_maxima
from .plot_style import (INK, METHOD_COLORS, QC_STATUS, WIDTH_DOUBLE,
                         panel_label)
from .windows import Window

BLUE = METHOD_COLORS["GEV"]
T_MAIN = int(config.RETURN_PERIOD)
RL = f"rl_{T_MAIN}"
_VARIANT = dict(color=INK["secondary"], ls=(0, (4, 2)), lw=1.1)


def _gapped(df: pd.DataFrame, key: str = "window_end") -> pd.DataFrame:
    """Reindex on consecutive years so dropped windows break the line."""
    if df.empty:
        return df
    full = range(int(df[key].min()), int(df[key].max()) + 1)
    return df.set_index(key).reindex(full).rename_axis(key).reset_index()


def _flagged_maxima(ax, annual: pd.DataFrame) -> None:
    """Annual maxima as faint dots; QC-flagged years in their status style."""
    obs = annual[annual["n_valid"] > 0]
    ax.plot(obs["year"], obs["annual_max_mm"], ls="none", marker="o",
            markersize=2.0, color=INK["axis"], zorder=1)
    for key in ("review", "systematic"):
        sel = obs[obs["status"] == key]
        st = QC_STATUS[key]
        ax.plot(sel["year"], sel["annual_max_mm"], ls="none",
                marker=st["marker"], markersize=3.4, color=st["color"],
                markeredgewidth=0.4, markeredgecolor="white", zorder=2)


# ----------------------------------------------------------------------
# Figure 7 - 200-year GEV return level vs final year (both stations)
# ----------------------------------------------------------------------
def plot_return_level_timeseries(results: dict[str, pd.DataFrame],
                                 annuals: dict[str, pd.DataFrame]
                                 ) -> plt.Figure:
    """Main Task 2 result: one panel per station plus a valid-years strip."""
    stations = list(results)
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 2.45 * len(stations) + 0.5))
    gs = fig.add_gridspec(2 * len(stations), 1,
                          height_ratios=[4, 0.9] * len(stations), hspace=0.18)
    xlim = (1748, 2024)
    top_axes, first = [], None
    for i, st in enumerate(stations):
        ax = fig.add_subplot(gs[2 * i], sharex=first)
        first = first or ax
        axn = fig.add_subplot(gs[2 * i + 1], sharex=first)
        main = select(results[st])
        line = _gapped(main)
        _flagged_maxima(ax, annuals[st])
        ax.plot(line["window_end"], line[RL], color=BLUE, lw=1.6, zorder=4)
        bad = main[~main["converged"]]
        ax.plot(bad["window_end"], bad[RL], ls="none", marker="o",
                markersize=4, mfc="white", mec=BLUE, mew=1.0, zorder=5)
        # Label the extremes of the curve (selective direct labels).
        for idx, va, dy in ((main[RL].idxmax(), "bottom", 6),
                            (main[RL].idxmin(), "top", -6)):
            r = main.loc[idx]
            ax.annotate(f"{r[RL]:.0f} mm\n{int(r.window_start)}-{int(r.window_end)}",
                        (r.window_end, r[RL]), xytext=(4, dy),
                        textcoords="offset points", fontsize=6, va=va,
                        color=INK["primary"])
        ax.set_ylabel("Daily rainfall [mm]")
        ax.set_ylim(0, max(main[RL].max() * 1.15, 180))
        panel_label(ax, "ab"[i], f"{STATIONS[st].name}")
        plt.setp(ax.get_xticklabels(), visible=False)

        axn.bar(main["window_end"], main["n_valid"], width=1.0,
                color=INK["axis"], lw=0)
        axn.axhline(config.MIN_VALID_YEARS, color=INK["primary"], lw=0.6)
        axn.set_ylim(0, config.WINDOW_YEARS + 2)
        axn.set_yticks([config.MIN_VALID_YEARS, config.WINDOW_YEARS])
        axn.set_ylabel("valid\nyears", fontsize=6.5)
        axn.grid(False)
        if i < len(stations) - 1:
            plt.setp(axn.get_xticklabels(), visible=False)
        top_axes.append(ax)
    axn.set_xlim(*xlim)
    axn.set_xlabel("Final year of the 30-year window")
    handles = [
        Line2D([], [], color=BLUE, lw=1.6,
               label=f"GEV {T_MAIN}-year return level (ML, 30-yr windows)"),
        Line2D([], [], ls="none", marker="o", mfc="white", mec=BLUE,
               label="ML shape xi <= -0.5 (irregular estimate)"),
        Line2D([], [], ls="none", marker="o", markersize=3, color=INK["axis"],
               label="Annual maximum (at its own year)"),
        Line2D([], [], ls="none", marker=QC_STATUS["review"]["marker"],
               color=QC_STATUS["review"]["color"], markersize=4,
               label="Annual max, QC-flagged year (retained)"),
        Line2D([], [], ls="none", marker=QC_STATUS["systematic"]["marker"],
               color=QC_STATUS["systematic"]["color"], markersize=4,
               label="Annual max, systematic-pattern year"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"How rare is rare? GEV {T_MAIN}-year daily rainfall in "
                 f"moving 30-year windows", x=0.0, ha="left", y=0.995,
                 fontsize=10, fontweight="bold")
    return fig


# ----------------------------------------------------------------------
# Figure 8 - GEV parameters through time, ML vs L-moments
# ----------------------------------------------------------------------
def plot_parameters(results: pd.DataFrame) -> plt.Figure:
    st = str(results["station"].iat[0])
    ml, lm = _gapped(select(results)), _gapped(select(results, method="lmom"))
    specs = [("mu", "Location mu [mm]"), ("sigma", "Scale sigma [mm]"),
             ("xi", "Shape xi [-]")]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(WIDTH_DOUBLE, 4.6))
    for i, (ax, (col, label)) in enumerate(zip(axes, specs)):
        ax.plot(ml["window_end"], ml[col], color=BLUE, lw=1.5,
                label="Maximum likelihood (main)")
        ax.plot(lm["window_end"], lm[col], **_VARIANT,
                label="L-moments (sensitivity)")
        ax.set_ylabel(label)
        panel_label(ax, "abc"[i])
    ax = axes[2]
    ax.axhline(0, color=INK["primary"], lw=0.7)
    ax.text(ml["window_end"].max() + 2, 0, "Gumbel\n(xi = 0)", fontsize=6,
            va="center", color=INK["primary"])
    ax.axhspan(ax.get_ylim()[0], -0.5, color=INK["band"], lw=0, zorder=0)
    ax.text(ml["window_end"].min(), -0.5, " ML irregular below -0.5",
            fontsize=5.8, va="top", color=INK["muted"])
    axes[0].legend(loc="upper left", fontsize=6.5, ncol=2)
    axes[-1].set_xlabel("Final year of the 30-year window")
    fig.suptitle(f"{STATIONS[st].name}: GEV parameters in moving 30-year "
                 f"windows", x=0.0, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.97, 0.97))
    return fig


# ----------------------------------------------------------------------
# Figure 9 - sensitivity of the 200-year level to three analysis choices
# ----------------------------------------------------------------------
def plot_sensitivity(results: pd.DataFrame) -> plt.Figure:
    st = str(results["station"].iat[0])
    main = select(results)
    variants = [
        ("Estimator", select(results, method="lmom"),
         "L-moments", "Maximum likelihood"),
        ("Data screening", select(results, scenario="qc"),
         "QC-screened record", "Record as provided"),
        ("Window definition", select(results, mode="consecutive"),
         "30 consecutive valid years", "30 calendar years, >= 20 valid"),
    ]
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True,
                             figsize=(WIDTH_DOUBLE, 4.9))
    for i, (ax, (title, alt, alt_label, ref_label)) in enumerate(
            zip(axes, variants)):
        ax.plot(_gapped(main)["window_end"], _gapped(main)[RL], color=BLUE,
                lw=1.5, label=ref_label)
        ax.plot(_gapped(alt)["window_end"], _gapped(alt)[RL], **_VARIANT,
                label=alt_label)
        both = main.merge(alt, on="window_end", suffixes=("", "_alt"))
        diff = (both[RL] - both[f"{RL}_alt"]).abs()
        note = (f"median |diff| = {diff.median():.1f} mm, "
                f"max = {diff.max():.1f} mm" if len(diff) else "")
        panel_label(ax, "abc"[i], f"{title}  ({note})")
        ax.legend(loc="upper left", fontsize=6.3, ncol=2)
        ax.set_ylabel(f"{T_MAIN}-yr level [mm]")
    axes[-1].set_xlabel("Final year of the window")
    fig.suptitle(f"{STATIONS[st].name}: sensitivity of the GEV {T_MAIN}-year "
                 f"level to analysis choices", x=0.0, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ----------------------------------------------------------------------
# Figure 10 - return-level plots for selected windows
# ----------------------------------------------------------------------
_T_TICKS = [2, 5, 10, 20, 50, 200, 500]


def _gumbel_variate(T):
    """Reduced Gumbel variate y = -ln(-ln(1 - 1/T)) used as the x-axis."""
    T = np.asarray(T, dtype=float)
    return -np.log(-np.log(1.0 - 1.0 / T))


def plot_window_fits(results: pd.DataFrame, annual: pd.DataFrame,
                     window_ends: list[int] | None = None) -> plt.Figure:
    """Annual maxima vs fitted GEV in selected windows (Gumbel paper).

    Default windows: the one with the highest and the lowest 200-year
    level, and the most recent one.
    """
    st = str(results["station"].iat[0])
    main = select(results).set_index("window_end")
    if window_ends is None:
        window_ends = [int(main[RL].idxmax()), int(main[RL].idxmin()),
                       int(main.index.max())]
    fig, axes = plt.subplots(1, len(window_ends), sharey=True,
                             figsize=(WIDTH_DOUBLE, 2.6))
    Tgrid = np.logspace(np.log10(1.05), np.log10(600), 200)
    ymax = 0
    for i, (ax, end) in enumerate(zip(np.atleast_1d(axes), window_ends)):
        r = main.loc[end]
        w = Window(int(r.window_start), int(end),
                   tuple(annual.loc[annual["valid"]
                                    & annual["year"].between(r.window_start, end),
                                    "year"]), "calendar")
        x = np.sort(window_maxima(annual, w))
        Femp = gev.weibull_plotting_positions(x.size)
        ax.axvspan(_gumbel_variate(x.size + 1), _gumbel_variate(600),
                   color=INK["band"], lw=0, zorder=0)
        ax.text(_gumbel_variate(x.size + 1), 0.02, " extrapolation\n beyond n+1",
                transform=ax.get_xaxis_transform(), fontsize=5.8,
                color=INK["muted"], va="bottom")
        for method, style in (("ml", dict(color=BLUE, lw=1.5)),
                              ("lmom", _VARIANT)):
            p = gev.fit(x, method)
            ax.plot(_gumbel_variate(Tgrid), gev.return_level(Tgrid, p),
                    **style, label="ML" if method == "ml" else "L-moments")
        ax.plot(-np.log(-np.log(Femp)), x, ls="none", marker="o",
                markersize=3.2, color=INK["secondary"], zorder=3,
                label="Annual maxima")
        ax.axvline(_gumbel_variate(T_MAIN), color=INK["primary"], lw=0.6)
        ax.plot(_gumbel_variate(T_MAIN), r[RL], marker="o", color=BLUE,
                markersize=5, mec="white", mew=1, zorder=5)
        ax.annotate(f"{r[RL]:.0f} mm", (_gumbel_variate(T_MAIN), r[RL]),
                    xytext=(-4, 4), textcoords="offset points", ha="right",
                    fontsize=6.5)
        ymax = max(ymax, r[RL], x.max())
        ax.set_xticks(_gumbel_variate(_T_TICKS))
        ax.set_xticklabels([str(t) for t in _T_TICKS])
        ax.set_xlabel("Return period T [years]")
        panel_label(ax, "abc"[i],
                    f"{int(r.window_start)}-{end}  (n={int(r.n_valid)}, "
                    f"xi={r.xi:+.2f})")
    axes[0].set_ylabel("Daily rainfall [mm]")
    axes[0].set_ylim(0, ymax * 1.12)
    axes[0].legend(loc="upper left", fontsize=6.2)
    fig.suptitle(f"{STATIONS[st].name}: GEV fits in selected windows "
                 f"(Gumbel probability paper)", x=0.0, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ----------------------------------------------------------------------
# Figure 11 - Marani & Zanetti (2015) conventions, for direct comparison
# ----------------------------------------------------------------------
def plot_mz2015_reproduction(results: pd.DataFrame) -> plt.Figure:
    """Padova, ML, 30-yr windows indexed by START year, record to 2006."""
    main = select(results)
    main = _gapped(main[main["window_end"] <= 2006])
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 2.5))
    ax.plot(main["window_start"], main["rl_200"], color=BLUE, lw=1.5,
            label="200-year return level")
    ax.plot(main["window_start"], main["rl_100"], color=BLUE, lw=1.0,
            ls=(0, (4, 2)), label="100-year return level")
    ax.set_xlabel("START year of the 30-year window (Marani & Zanetti, 2015 "
                  "convention)")
    ax.set_ylabel("Daily rainfall [mm]")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title("Padova: GEV (ML) return levels with the conventions of "
                 "Marani & Zanetti (2015, Fig. 3), data to 2006", loc="left")
    return fig

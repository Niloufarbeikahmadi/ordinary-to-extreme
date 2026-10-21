"""
Quality-control figures (Task 1).

Every function takes a ``QCResult`` (or a list of them), draws one
publication-ready figure in the shared project style, and returns the
Figure, so the caller decides whether to save or show it.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config
from .metadata import STATIONS
from .plot_style import (INK, QC_STATUS, WIDTH_DOUBLE, panel_label,
                         shade_eras)
from .qc import QCResult

_YEAR_LIMITS = (1720, 2025)


def _status_legend_handles(statuses=("valid", "review", "systematic",
                                     "missing"), with_markers=True):
    handles = []
    for key in statuses:
        st = QC_STATUS[key]
        if with_markers:
            handles.append(Line2D([], [], ls="none", marker=st["marker"],
                                  color=st["color"], markersize=4.5,
                                  markeredgewidth=0.6, label=st["label"]))
        else:
            hatch = "////" if key == "systematic" else None
            handles.append(Patch(facecolor=st["color"], hatch=hatch,
                                 edgecolor="white", lw=0, label=st["label"]))
    return handles


def _overlay_status(ax, annual: pd.DataFrame, col: str) -> None:
    """Re-draw flagged years on top of the neutral series."""
    for key in ("review", "systematic"):
        sel = annual[annual["status"] == key]
        st = QC_STATUS[key]
        ax.plot(sel["year"], sel[col], ls="none", marker=st["marker"],
                color=st["color"], markersize=4.2, markeredgewidth=0.6,
                markeredgecolor="white" if key == "review" else st["color"],
                zorder=4)


# ----------------------------------------------------------------------
# Figure 1 - record timeline (both stations)
# ----------------------------------------------------------------------
def plot_record_timeline(results: list[QCResult]) -> plt.Figure:
    """One status strip per station: which years exist and how usable."""
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 0.55 + 0.42 * len(results)))
    ax.grid(False)
    for row, res in enumerate(reversed(results)):
        a = res.scenario("asis")
        for key in ("valid", "review", "systematic", "missing"):
            years = a.loc[a["status"] == key, "year"]
            st = QC_STATUS[key]
            ax.bar(years, 0.62, bottom=row - 0.31, width=1.0,
                   color=st["color"], linewidth=0,
                   hatch="////" if key == "systematic" else None,
                   edgecolor="white")
        q = res.scenario("qc")
        ax.text(_YEAR_LIMITS[1] + 2, row,
                f"{int((a['n_valid'] > 0).sum())} yrs on file\n"
                f"{int(q['valid'].sum())} valid (QC)",
                va="center", ha="left", fontsize=6.5, color=INK["secondary"])
        station = STATIONS[res.station]
        for era in station.eras[1:]:
            ax.plot([era.start - 0.5] * 2, [row + 0.31, row + 0.45],
                    color=INK["muted"], lw=0.6)
        for era in station.eras:
            ax.text((era.start + era.end) / 2, row + 0.40, era.short,
                    ha="center", va="bottom", fontsize=5.8,
                    color=INK["muted"])
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([STATIONS[r.station].name for r in reversed(results)],
                       fontsize=8, color=INK["primary"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(*_YEAR_LIMITS)
    ax.set_ylim(-0.5, len(results) - 0.5 + 0.25)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Year")
    ax.legend(handles=_status_legend_handles(with_markers=False),
              loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=4,
              fontsize=6.5)
    ax.set_title("Daily rainfall records: availability and QC status by "
                 "year", loc="left")
    return fig


# ----------------------------------------------------------------------
# Figure 2 - annual diagnostics (one station)
# ----------------------------------------------------------------------
def plot_annual_diagnostics(res: QCResult) -> plt.Figure:
    """Four stacked panels: annual max, wet days, wet-day depth, total."""
    a = res.scenario("asis").set_index("year").reindex(
        range(int(res.annual["year"].min()), int(res.annual["year"].max()) + 1))
    a.index.name = "year"
    a = a.reset_index()
    a.loc[a["n_valid"] == 0, ["annual_max_mm", "n_wet", "mean_wet_mm",
                              "total_mm"]] = np.nan
    panels = [
        ("annual_max_mm", "Annual maximum\ndaily rainfall [mm]"),
        ("n_wet", f"Wet days per year\n(P > {config.WET_THRESHOLD_MM:g} mm) [d]"),
        ("mean_wet_mm", "Mean wet-day\ndepth [mm]"),
        ("total_mm", "Annual total [mm]"),
    ]
    station = STATIONS[res.station]
    fig, axes = plt.subplots(len(panels), 1, sharex=True,
                             figsize=(WIDTH_DOUBLE, 6.3))
    for i, (ax, (col, label)) in enumerate(zip(axes, panels)):
        shade_eras(ax, station.eras, label=(i == 0))
        ax.plot(a["year"], a[col], color=INK["axis"], lw=0.8, zorder=2)
        ax.plot(a["year"], a[col], ls="none", marker="o", markersize=2.2,
                color=INK["secondary"], zorder=3)
        # 30-year centred running median: a first look at slow variability,
        # computed on calendar years (missing years reduce the sample).
        run = a[col].rolling(config.WINDOW_YEARS, center=True,
                             min_periods=config.MIN_VALID_YEARS).median()
        ax.plot(a["year"], run, color=INK["primary"], lw=1.4, zorder=5)
        _overlay_status(ax, a.dropna(subset=["status"]), col)
        ax.set_ylabel(label)
        panel_label(ax, "abcd"[i])
        ax.set_xlim(a["year"].min() - 2, a["year"].max() + 2)
    present = [s for s in ("review", "systematic")
               if (a["status"] == s).any()]
    handles = [Line2D([], [], color=INK["secondary"], marker="o", ls="none",
                      markersize=3, label="Annual value"),
               Line2D([], [], color=INK["primary"], lw=1.4,
                      label=f"{config.WINDOW_YEARS}-yr running median "
                            f"(>= {config.MIN_VALID_YEARS} valid yrs)"),
               *_status_legend_handles(present)]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03), fontsize=6.5)
    axes[-1].set_xlabel("Year")
    fig.suptitle(f"{station.name}: annual diagnostics of the daily record",
                 x=0.0, ha="left", y=1.0, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return fig


# ----------------------------------------------------------------------
# Figure 3 - QC fingerprint (wet days vs wet-day depth)
# ----------------------------------------------------------------------
def plot_qc_fingerprint(res: QCResult) -> plt.Figure:
    """Wet days N vs mean wet-day depth, with iso-total hyperbolas.

    Along a hyperbola N x depth = const the wet-day total is unchanged.
    Multi-day amounts reported on one day slide a year up-left along its
    hyperbola: the total is preserved, N falls and the depth rises.
    """
    a = res.scenario("asis")
    a = a[a["n_valid"] > 0]
    xlim, ylim = (15, 135), (5, 36)
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE * 0.62, 3.4))
    ax.grid(True, axis="both")
    n = np.linspace(xlim[0], xlim[1], 400)
    for total in (300, 500, 700, 900, 1100, 1300):
        depth = total / n
        ax.plot(n, depth, color=INK["grid"], lw=0.9, zorder=1)
        # Label where the curve leaves the plot: right edge, else top edge.
        if total / (xlim[1] - 4) >= ylim[0]:
            nx, dy = xlim[1] - 4, total / (xlim[1] - 4)
        else:
            dy = ylim[0] * 1.08
            nx = total / dy
        ax.text(nx, dy, f"{total} mm", fontsize=5.8, color=INK["muted"],
                ha="center", va="center",
                bbox=dict(fc=INK["surface"], ec="none", pad=0.6), zorder=2)
    for key in ("valid", "review", "systematic"):
        sel = a[a["status"] == key]
        st = QC_STATUS[key]
        ax.plot(sel["n_wet"], sel["mean_wet_mm"], ls="none",
                marker=st["marker"], color=st["color"],
                markersize=3.0 if key == "valid" else 4.5,
                alpha=0.55 if key == "valid" else 1.0,
                markeredgewidth=0.6,
                markeredgecolor="white" if key != "systematic" else st["color"],
                label=st["label"], zorder=3 if key == "valid" else 4)
    # Label selectively: isolated review years individually, the systematic
    # cluster once (its years are listed in the QC report).
    for r in a[a["status"] == "review"].itertuples():
        ax.annotate(str(r.year), (r.n_wet, r.mean_wet_mm), fontsize=5.5,
                    color=INK["secondary"], xytext=(3, 2),
                    textcoords="offset points")
    sysm = a[a["status"] == "systematic"]
    if not sysm.empty:
        from .qc import compress_years
        top = sysm.loc[sysm["mean_wet_mm"].idxmax()]
        ax.annotate(compress_years(sysm["year"]),
                    (top["n_wet"], top["mean_wet_mm"]), xytext=(10, 4),
                    textcoords="offset points", fontsize=6,
                    color=INK["primary"],
                    arrowprops=dict(arrowstyle="-", lw=0.5,
                                    color=INK["secondary"]))
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([5, 7, 10, 15, 20, 30])
    ax.set_yticklabels(["5", "7", "10", "15", "20", "30"])
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel(f"Wet days per year, N (P > {config.WET_THRESHOLD_MM:g} mm) [d]")
    ax.set_ylabel("Mean wet-day depth [mm]  (log scale)")
    ax.set_title(f"{STATIONS[res.station].name}: QC fingerprint "
                 f"(grey curves: constant wet-day total)", loc="left")
    ax.legend(loc="upper right", fontsize=6.3)
    return fig


# ----------------------------------------------------------------------
# Figure 4 - measurement resolution and its consequence for "wet"
# ----------------------------------------------------------------------
def plot_resolution(res: QCResult) -> plt.Figure:
    """Detected resolution, smallest value counted as wet, small-rain share."""
    a = res.scenario("asis")
    a = a[a["n_valid"] > 0]
    station = STATIONS[res.station]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(WIDTH_DOUBLE, 4.6))

    # (a) Resolution as a categorical "barcode": one row per detected step,
    # ordered by size, plus a row for years that fit no single step.
    ax = axes[0]
    shade_eras(ax, station.eras)
    steps = sorted(a["resolution_mm"].dropna().unique())
    has_mixed = bool(a["resolution_mm"].isna().any())
    rows = {q: i + int(has_mixed) for i, q in enumerate(steps)}
    ypos = a["resolution_mm"].map(rows).fillna(0).to_numpy()
    ax.bar(a["year"], 0.7, bottom=ypos - 0.35, width=1.0,
           color=np.where(a["resolution_mm"].isna(), INK["axis"],
                          INK["secondary"]))
    ticks = ([0] if has_mixed else []) + list(rows.values())
    labels = (["mixed"] if has_mixed else []) + [f"{q:g}" for q in steps]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.3)
    ax.set_ylim(min(ticks) - 0.6, max(ticks) + 0.6)
    ax.grid(False)
    ax.set_ylabel("Recording\nresolution [mm]")
    panel_label(ax, "a")

    ax = axes[1]
    shade_eras(ax, station.eras, label=False)
    ax.plot(a["year"], a["min_wet_value_mm"], ls="none", marker="o",
            markersize=2.4, color=INK["secondary"])
    ax.axhline(config.WET_THRESHOLD_MM, color=INK["primary"], lw=0.8)
    ax.text(a["year"].max() + 2, config.WET_THRESHOLD_MM,
            f"threshold\n{config.WET_THRESHOLD_MM:g} mm", fontsize=5.8,
            color=INK["primary"], va="center", ha="left")
    ax.set_ylim(0.9, 2.2)
    ax.set_ylabel("Smallest value\ncounted as wet [mm]")
    panel_label(ax, "b")

    ax = axes[2]
    shade_eras(ax, station.eras, label=False)
    ax.bar(a["year"], 100 * a["frac_small"], width=0.8, color=INK["secondary"])
    ax.set_ylabel("Rain days with\nP <= 1 mm [%]")
    ax.set_xlabel("Year")
    panel_label(ax, "c")
    ax.set_xlim(a["year"].min() - 2, a["year"].max() + 8)
    fig.suptitle(f"{station.name}: recording resolution and its effect on "
                 f"the wet-day definition", x=0.0, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ----------------------------------------------------------------------
# Figure 5 - duplicated block (only if one exists)
# ----------------------------------------------------------------------
def plot_duplicate_block(res: QCResult, pad: int = 5) -> plt.Figure | None:
    """Mirror plot of the first duplicated block and its repeat."""
    if res.duplicates.empty:
        return None
    b = res.duplicates.iloc[0]
    p = res.daily["precip_mm"].to_numpy()
    n = int(b.length_days) + 2 * pad
    i0, j0 = int(b.first_index) - pad, int(b.repeat_index) - pad
    x = np.arange(n) - pad + int(b.first_day_of_year)
    first, rep = p[i0:i0 + n], p[j0:j0 + n]
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE * 0.62, 2.6))
    ax.axvspan(int(b.first_day_of_year) - 0.5,
               int(b.first_day_of_year) + int(b.length_days) - 0.5,
               color=INK["band"], lw=0, zorder=0)
    ax.bar(x, first, width=0.75, color=INK["secondary"],
           label=f"{int(b.first_year)}")
    ax.bar(x, -rep, width=0.75, color=INK["muted"],
           label=f"{int(b.repeat_year)} (drawn downward)")
    ax.axhline(0, color=INK["axis"], lw=0.6)
    lim = np.nanmax(np.abs(np.r_[first, rep])) * 1.15
    ax.set_ylim(-lim, lim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
    ax.set_xlabel("Day of year (position in the file)")
    ax.set_ylabel("Daily rainfall [mm]")
    ax.set_title(f"{STATIONS[res.station].name}: {int(b.length_days)} "
                 f"identical days in {int(b.first_year)} and "
                 f"{int(b.repeat_year)} (shaded)", loc="left")
    ax.legend(loc="lower right", fontsize=6.5)
    return fig


# ----------------------------------------------------------------------
# Figure 6 - context of the largest daily values
# ----------------------------------------------------------------------
def plot_largest_events(res: QCResult, n: int = 6, half: int = 10
                        ) -> plt.Figure:
    """Small multiples: the ``n`` largest daily values with +/- ``half`` days.

    A storm usually has rain on neighbouring days. An isolated spike after a
    dry spell, especially in winter, deserves a second look.
    """
    d = res.daily.reset_index(drop=True)
    top = d.nlargest(n, "precip_mm")
    spike_dates = set(res.spikes["date"]) if not res.spikes.empty else set()
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharey=True,
                             figsize=(WIDTH_DOUBLE, 1.55 * nrows + 0.4))
    axes = np.atleast_1d(axes).ravel()
    ymax = float(top["precip_mm"].max()) * 1.12
    for ax, (idx, row) in zip(axes, top.iterrows()):
        lo, hi = max(0, idx - half), min(len(d), idx + half + 1)
        seg = d.iloc[lo:hi]
        lag = np.arange(lo, hi) - idx
        colors = [INK["primary"] if k == 0 else INK["muted"] for k in lag]
        ax.bar(lag, seg["precip_mm"], width=0.8, color=colors)
        when = row["date"].strftime("%d %b %Y") if pd.notna(row["date"]) \
            else f"{row['year']} d{row['day_of_year']}"
        flag = "  - isolated winter spike" if row["date"] in spike_dates else ""
        ax.set_title(f"{when}: {row['precip_mm']:.1f} mm{flag}",
                     fontsize=7, loc="left")
        ax.set_ylim(0, ymax)
        ax.set_xlim(-half - 0.8, half + 0.8)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_ylabel("Daily rainfall [mm]")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Days from event")
    fig.suptitle(f"{STATIONS[res.station].name}: the {n} largest daily "
                 f"values in their +/-{half}-day context", x=0.0, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig

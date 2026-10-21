"""
Figures for the common bootstrap and its Monte Carlo check (Task 4).

Colour rule (as in every task): GEV is blue, MEVD is orange; a third,
reference estimator (the pooled-Weibull MEVD of the estimation-noise test)
is aqua. Bands use the method colour at low opacity, so the point estimate
(the solid line) stays the loudest element.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config
from .gev_plots import _gapped
from .metadata import STATIONS
from .plot_style import INK, METHOD_COLORS, WIDTH_DOUBLE, panel_label

BLUE, ORANGE, AQUA = (METHOD_COLORS[k] for k in ("GEV", "MEV", "REF"))
T_MAIN = int(config.RETURN_PERIOD)
LO, HI = (f"p{p:g}" for p in config.CI_PERCENTILES)
_XLIM = (1748, 2030)
_STATION_MARKER = {"bologna": "o", "padova": "s"}


def _end_labels(ax, x, items, min_gap):
    """Direct labels at the line ends, pushed apart if closer than min_gap.

    ``items`` is a list of (y, text); ``min_gap`` is in data units.
    """
    items = sorted(items)
    ys = [y for y, _ in items]
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] < min_gap:
            mid = 0.5 * (ys[k] + ys[k - 1])
            ys[k - 1], ys[k] = mid - min_gap / 2, mid + min_gap / 2
    for (y0, text), y in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color=INK["primary"],
                    annotation_clip=False)


# ----------------------------------------------------------------------
# Figure 17 - point estimates with 90% bootstrap bands (key figure)
# ----------------------------------------------------------------------
def plot_bootstrap_bands(summaries: dict[str, pd.DataFrame],
                         ymax: float = 420.0, width_strip: bool = True,
                         title: bool = True) -> plt.Figure:
    """200-year level vs final year with pointwise 90% bands, both models.

    ``width_strip`` adds a strip with the band width relative to the
    estimate under each station; the compact version (False, no title) is
    the one used in the report.
    """
    stations = list(summaries)
    if width_strip:
        fig = plt.figure(figsize=(WIDTH_DOUBLE, 2.6 * len(stations) + 0.6))
        gs = fig.add_gridspec(2 * len(stations), 1, top=0.93, bottom=0.08,
                              height_ratios=[4, 1.25] * len(stations),
                              hspace=0.3)
    else:
        fig = plt.figure(figsize=(WIDTH_DOUBLE, 1.75 * len(stations) + 0.45))
        gs = fig.add_gridspec(len(stations), 1, top=0.95, bottom=0.17,
                              hspace=0.32)
    first = None
    for i, st in enumerate(stations):
        ax = fig.add_subplot(gs[2 * i if width_strip else i], sharex=first)
        first = first or ax
        axw = (fig.add_subplot(gs[2 * i + 1], sharex=first) if width_strip
               else ax)
        s = _gapped(summaries[st].sort_values("window_end"))
        x = s["window_end"]
        g_hi = s[f"gev_{HI}"].clip(upper=ymax)
        ax.fill_between(x, s[f"gev_{LO}"], g_hi, color=BLUE, alpha=0.15,
                        lw=0, zorder=1)
        for col in (s[f"gev_{LO}"], g_hi):
            ax.plot(x, col, color=BLUE, lw=0.45, alpha=0.55, zorder=2)
        ax.fill_between(x, s[f"mev_{LO}"], s[f"mev_{HI}"], color=ORANGE,
                        alpha=0.38, lw=0, zorder=3)
        ax.plot(x, s["gev_rl"], color=BLUE, lw=1.2, zorder=4)
        ax.plot(x, s["mev_rl"], color=ORANGE, lw=1.7, zorder=5)
        # GEV upper bounds beyond the axis: small carets on the top edge
        off = s[s[f"gev_{HI}"] > ymax]
        if len(off):
            ax.plot(off["window_end"], np.full(len(off), ymax * 0.985),
                    ls="none", marker="^", markersize=2.6, color=BLUE,
                    clip_on=False, zorder=6)
            ax.text(1.0, 1.02, f"GEV 95th percentile above {ymax:.0f} mm in "
                    f"{len(off)} windows (max {off[f'gev_{HI}'].max():.0f} mm)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=6, color=INK["muted"])
        last = s.dropna(subset=["mev_rl"]).iloc[-1]
        _end_labels(ax, last["window_end"], [(last["mev_rl"], "MEVD"),
                                             (last["gev_rl"], "GEV")],
                    min_gap=0.05 * ymax)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f"{T_MAIN}-year daily\nrainfall [mm]")
        panel_label(ax, "ab"[i], STATIONS[st].name)
        if not width_strip:
            if i < len(stations) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            continue
        plt.setp(ax.get_xticklabels(), visible=False)
        # strip: band width relative to the estimate (log scale)
        axw.plot(x, s["gev_rel_width"], color=BLUE, lw=1.0)
        axw.plot(x, s["mev_rel_width"], color=ORANGE, lw=1.4)
        axw.set_yscale("log")
        axw.set_ylim(0.05, 8)
        axw.set_yticks([0.1, 0.3, 1, 3])
        axw.set_yticklabels(["0.1", "0.3", "1", "3"])
        axw.minorticks_off()
        axw.set_ylabel("band width /\nestimate", fontsize=6.5)
        if i < len(stations) - 1:
            plt.setp(axw.get_xticklabels(), visible=False)
    axw.set_xlim(*_XLIM)
    axw.set_xlabel("Final year of the 30-year window")
    handles = [
        (Line2D([], [], color=ORANGE, lw=1.7),
         f"MEVD {T_MAIN}-yr level"),
        (Patch(color=ORANGE, alpha=0.38, lw=0), "MEVD 90% bootstrap band"),
        (Line2D([], [], color=BLUE, lw=1.2), f"GEV {T_MAIN}-yr level (ML)"),
        (Patch(color=BLUE, alpha=0.15, lw=0), "GEV 90% bootstrap band"),
        (Line2D([], [], ls="none", marker="^", markersize=3, color=BLUE),
         f"GEV 95th percentile off scale (> {ymax:.0f} mm)"),
    ]
    fig.legend(*zip(*handles), loc="lower center", ncol=3, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.06 if width_strip else -0.01))
    if title:
        fig.suptitle(f"Common bootstrap: {config.N_BOOTSTRAP} resamplings of "
                     f"the same years for both models, 5th-95th percentile "
                     f"bands", x=0.0, ha="left", y=0.995, fontsize=10,
                     fontweight="bold")
    return fig


# ----------------------------------------------------------------------
# Figure 18 - bootstrap distributions in selected windows
# ----------------------------------------------------------------------
def plot_bootstrap_distributions(cases: list[tuple[str, int, str]],
                                 replicates: dict[str, dict],
                                 summaries: dict[str, pd.DataFrame],
                                 annuals: dict[str, pd.DataFrame],
                                 xmax: float = 400.0) -> plt.Figure:
    """Histograms of the 500 bootstrap estimates for a few windows.

    ``cases`` holds (station, final year, short note) tuples.
    """
    n = len(cases)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH_DOUBLE, 2.6 * nrow + 0.5),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    bins = np.arange(0, xmax + 5, 5.0)
    for k, (ax, (st, end, note)) in enumerate(zip(axes, cases)):
        rep = replicates[st]
        j = int(np.flatnonzero(rep["window_end"] == end)[0])
        row = summaries[st].set_index("window_end").loc[end]
        ann = annuals[st]
        sel = ann[(ann["year"] > end - config.WINDOW_YEARS) & (ann["year"] <= end)
                  & ann["valid"]]
        top, n_over = 0.0, 0
        for m, col, alpha in (("gev", BLUE, 0.35), ("mev", ORANGE, 0.6)):
            vals = rep[f"rl_{m}"][j]
            if m == "gev":
                n_over = int(np.sum(vals > xmax))
            # values beyond the axis are piled into the last bin
            h, _ = np.histogram(np.minimum(vals, xmax - 1e-9), bins=bins)
            ax.stairs(h, bins, fill=True, color=col, alpha=alpha, lw=0)
            ax.stairs(h, bins, color=col, lw=0.7)
            top = max(top, h.max())
        # point estimates (full height) and 5-95% bands (below the baseline)
        for m, col, yfrac in (("mev", ORANGE, -0.08), ("gev", BLUE, -0.17)):
            y = top * yfrac
            lo_, hi_ = row[f"{m}_{LO}"], min(row[f"{m}_{HI}"], xmax)
            ax.plot([lo_, hi_], [y, y], color=col, lw=2.2,
                    solid_capstyle="butt", clip_on=False)
            ax.plot([row[f"{m}_rl"]] * 2, [y - 0.04 * top, top * 1.02],
                    color=col, lw=1.0)
        amax = sel["annual_max_mm"].max()
        ax.plot([amax, amax], [-0.24 * top, top * 1.22], color=INK["secondary"],
                lw=0.8, ls=(0, (3, 2)))
        # label to the LEFT of the line; the notes block sits top right
        ax.text(amax, top * 1.22, f"largest annual max\nin window ({amax:.0f} mm) ",
                fontsize=5.8, color=INK["secondary"], va="top", ha="right")
        ax.axhline(0, color=INK["axis"], lw=0.6)
        # head room above the histograms keeps the notes clear of the data
        ax.set_ylim(-0.24 * top, top * 1.75)
        ax.set_xlim(0, xmax)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        start = end - config.WINDOW_YEARS + 1
        panel_label(ax, "abcdefgh"[k], f"{STATIONS[st].name} {start}-{end}")
        bound = row["gev_share_lower_bound"] + row["gev_share_upper_bound"]
        lines = [note, f"GEV shape on the window: {row['gev_xi_point']:+.2f}",
                 f"GEV refits on a shape bound: {bound:.1%}"]
        if n_over:
            lines.append(f"GEV estimates above {xmax:.0f} mm: {n_over} of "
                         f"{rep['rl_gev'].shape[1]} (last bin)")
        ax.text(0.99, 0.97, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=6, color=INK["secondary"],
                linespacing=1.35)
        if k >= n - ncol:
            ax.set_xlabel(f"Bootstrap {T_MAIN}-year level [mm]")
    for ax in axes[n:]:
        ax.set_visible(False)
    handles = [Patch(color=BLUE, alpha=0.35, lw=0), Patch(color=ORANGE, alpha=0.6, lw=0),
               Line2D([], [], color=INK["primary"], lw=1.0),
               Line2D([], [], color=INK["primary"], lw=1.6)]
    fig.legend(handles, ["GEV: 500 bootstrap estimates",
                         "MEVD: 500 bootstrap estimates",
                         "estimate on the original window",
                         "5th-95th percentile band"],
               loc="lower center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("What the bands are made of: bootstrap distributions in "
                 "selected windows", x=0.0, ha="left", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig


# ----------------------------------------------------------------------
# Figure 19 - paired difference GEV - MEVD
# ----------------------------------------------------------------------
def plot_paired_difference(summaries: dict[str, pd.DataFrame],
                           expected_offset: dict[str, float] | None = None,
                           ylim: tuple[float, float] = (-110, 260)
                           ) -> plt.Figure:
    """90% bootstrap band of GEV - MEVD (same resamples for both models).

    ``expected_offset`` (optional) is, per station, the relative difference
    (GEV - MEVD) / MEVD expected from the estimators' median errors alone,
    as measured in the synthetic ordinary-events world.
    """
    stations = list(summaries)
    fig, axes = plt.subplots(len(stations), 1, sharex=True,
                             figsize=(WIDTH_DOUBLE, 2.1 * len(stations) + 0.5))
    axes = np.atleast_1d(axes)
    for i, (ax, st) in enumerate(zip(axes, stations)):
        s = _gapped(summaries[st].sort_values("window_end"))
        x = s["window_end"]
        gev_hi = s[f"diff_{LO}"] > 0          # GEV above MEVD beyond noise
        mev_hi = s[f"diff_{HI}"] < 0          # MEVD above GEV beyond noise
        for mask, col in ((gev_hi, BLUE), (mev_hi, ORANGE)):
            for xv in x[mask.fillna(False)]:
                ax.axvspan(xv - 0.5, xv + 0.5, color=col, alpha=0.18, lw=0,
                           zorder=0)
        ax.fill_between(x, s[f"diff_{LO}"].clip(*ylim),
                        s[f"diff_{HI}"].clip(*ylim), color=INK["axis"],
                        alpha=0.55, lw=0, zorder=1)
        ax.plot(x, s["diff_point"], color=INK["primary"], lw=1.1, zorder=3)
        ax.axhline(0, color=INK["secondary"], lw=0.7, zorder=2)
        if expected_offset and st in expected_offset:
            ax.plot(x, expected_offset[st] * s["mev_rl"], color=INK["primary"],
                    lw=0.8, ls=(0, (1, 1.5)), zorder=2)
        off = s[s[f"diff_{HI}"] > ylim[1]]
        if len(off):
            ax.plot(off["window_end"], np.full(len(off), ylim[1] * 0.985),
                    ls="none", marker="^", markersize=2.4,
                    color=INK["secondary"], clip_on=False)
        ax.set_ylim(*ylim)
        ax.set_ylabel(f"GEV − MEVD,\n{T_MAIN}-yr level [mm]")
        n_g, n_m = int(gev_hi.sum()), int(mev_hi.sum())
        panel_label(ax, "ab"[i], f"{STATIONS[st].name}: GEV higher beyond "
                    f"noise in {n_g} windows, MEVD higher in {n_m}")
    axes[-1].set_xlim(*_XLIM)
    axes[-1].set_xlabel("Final year of the 30-year window")
    handles = [Line2D([], [], color=INK["primary"], lw=1.1),
               Patch(color=INK["axis"], alpha=0.55, lw=0),
               Patch(color=BLUE, alpha=0.18, lw=0),
               Patch(color=ORANGE, alpha=0.18, lw=0)]
    labels = ["difference on the original window",
              "90% band of the paired bootstrap difference",
              "band entirely above 0 (GEV higher)",
              "band entirely below 0 (MEVD higher)"]
    if expected_offset:
        handles.append(Line2D([], [], color=INK["primary"], lw=0.8,
                              ls=(0, (1, 1.5))))
        labels.append("offset expected from estimator bias alone "
                      "(synthetic test)")
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Where do GEV and MEVD disagree by more than sampling noise?",
                 x=0.0, ha="left", y=0.995, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


# ----------------------------------------------------------------------
# Figure 20 - Monte Carlo check of the bootstrap intervals
# ----------------------------------------------------------------------
def plot_calibration(cov_records: pd.DataFrame, cov_summary: pd.DataFrame,
                     noise_summary: pd.DataFrame, noise_share: pd.DataFrame
                     ) -> plt.Figure:
    """(a) coverage, (b) point-estimate error, (c) yearly-fit noise bias,
    (d) share of the observed scatter of w that is estimation noise."""
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_DOUBLE, 5.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    rows = [("gev", "GEV", "GEV world"), ("ordinary", "GEV", "ordinary-events world"),
            ("ordinary", "MEV", "ordinary-events world")]
    stations = list(dict.fromkeys(cov_summary["station"]))
    ylab, ypos = [], []
    y = 0
    cs = cov_summary.set_index(["world", "model", "station"])
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    for world, model, wname in rows:
        for st in stations:
            r = cs.loc[(world, model, st)]
            col = BLUE if model == "GEV" else ORANGE
            mk = _STATION_MARKER.get(st, "o")
            ax_a.errorbar(r["coverage"], y, xerr=1.96 * r["coverage_se"],
                          fmt=mk, color=col, markersize=4, elinewidth=1,
                          capsize=0)
            ax_a.text(1.055, y, f"{r['share_truth_above_band']:.0%} / "
                      f"{r['share_truth_below_band']:.0%}",
                      va="center", ha="center", fontsize=6,
                      color=INK["secondary"])
            g = cov_records[(cov_records["world"] == world)
                            & (cov_records["station"] == st)]
            rel = (g[f"{model.lower()}_point"] / g["truth"] - 1.0).to_numpy()
            q05, q25, q50, q75, q95 = np.percentile(rel, [5, 25, 50, 75, 95])
            ax_b.plot([q05, q95], [y, y], color=col, lw=0.9)
            ax_b.add_patch(plt.Rectangle((q25, y - 0.28), q75 - q25, 0.56,
                                         color=col, alpha=0.45, lw=0))
            ax_b.plot([q50, q50], [y - 0.32, y + 0.32], color=INK["primary"],
                      lw=1.2)
            ylab.append(f"{model.replace('MEV', 'MEVD')} · {wname} · "
                        f"{STATIONS[st].name}")
            ypos.append(y)
            y += 1
        y += 0.6
    for ax in (ax_a, ax_b):
        ax.set_yticks(ypos)
        # head room inside the axes for the "missed" column header
        ax.set_ylim(ypos[-1] + 0.7, ypos[0] - 1.5)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    ax_a.set_yticklabels(ylab, fontsize=6)
    ax_b.set_yticklabels([])
    ax_a.axvline(cov_summary["nominal"].iat[0], color=INK["primary"], lw=0.8)
    ax_a.set_xlim(0.55, 1.12)
    ax_a.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.axvspan(1.005, 1.12, color=INK["band"], lw=0, zorder=0)
    ax_a.text(1.0625, ypos[0] - 0.65, "truth missed:\nabove / below",
              fontsize=5.6, color=INK["muted"], ha="center", va="bottom",
              linespacing=1.1)
    ax_a.set_xlabel("Coverage of the nominal 90% band")
    panel_label(ax_a, "a", "does the 90% band contain the truth?")
    ax_b.axvline(0, color=INK["primary"], lw=0.8)
    ax_b.set_xlim(-0.8, 1.2)
    ax_b.tick_params(axis="y", length=0)
    ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax_b.set_xlabel(f"Relative error of the {T_MAIN}-yr estimate")
    panel_label(ax_b, "b", "error of the point estimate")
    # (c) estimation-noise experiment
    ax = axes[1, 0]
    Ts = sorted(noise_summary["T"].unique())
    for k, st in enumerate(stations):
        for est, col in (("yearly", ORANGE), ("pooled", AQUA)):
            g = noise_summary[(noise_summary["station"] == st)
                              & (noise_summary["estimator"] == est)].set_index("T")
            xs = np.array(Ts) * (1.0 + 0.06 * (k - 0.5))
            ax.errorbar(xs, g.loc[Ts, "median_rel_error"],
                        yerr=[g.loc[Ts, "median_rel_error"] - g.loc[Ts, "p05_rel_error"],
                              g.loc[Ts, "p95_rel_error"] - g.loc[Ts, "median_rel_error"]],
                        fmt=_STATION_MARKER.get(st, "o"), color=col, ms=3.5,
                        lw=1.0, elinewidth=0.8, capsize=0,
                        ls="-" if st == stations[0] else (0, (3, 1.5)))
    ax.axhline(0, color=INK["primary"], lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([f"{t:g}" for t in Ts])
    ax.minorticks_off()
    ax.set_xlabel("Return period [years]")
    ax.set_ylabel("Relative error of the MEVD level")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_ylim(-0.12, 0.62)
    panel_label(ax, "c", "identical years: noise alone biases yearly fits")
    ax.legend(handles=[Line2D([], [], color=ORANGE, marker="o", ms=3.5),
                       Line2D([], [], color=AQUA, marker="o", ms=3.5),
                       Line2D([], [], color=INK["secondary"], marker="o",
                              ls="none", ms=3.5),
                       Line2D([], [], color=INK["secondary"], marker="s",
                              ls="none", ms=3.5)],
              labels=["Weibull fitted per year (EW = 1)",
                      "one Weibull for all 30 years",
                      STATIONS[stations[0]].name + "-like year",
                      STATIONS[stations[-1]].name + "-like year"],
              loc="upper left", fontsize=6, ncol=2, columnspacing=1.0)
    # (d) variance decomposition of the yearly shape w
    ax = axes[1, 1]
    ns = noise_share.set_index("station").loc[stations]
    for k, st in enumerate(stations):
        noise = ns.at[st, "noise_share_of_variance"]
        ax.barh(k, noise, color=ORANGE, alpha=0.75, height=0.45)
        ax.barh(k, 1 - noise, left=noise, color=INK["axis"], height=0.45)
        ax.text(noise / 2, k, f"{noise:.0%} noise", ha="center", va="center",
                fontsize=6.5, color=INK["primary"])
        ax.text(noise + (1 - noise) / 2, k, f"{1 - noise:.0%} real", ha="center",
                va="center", fontsize=6.5, color=INK["primary"])
        ax.text(0.0, k + 0.3, f"sd of yearly w: observed "
                f"{ns.at[st, 'observed_sd_w']:.3f}, noise "
                f"{ns.at[st, 'noise_sd_w']:.3f}, real "
                f"{ns.at[st, 'implied_true_sd_w']:.3f}",
                va="top",
                fontsize=5.8, color=INK["secondary"])
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels([STATIONS[s].name for s in stations])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.5, len(stations) - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of the year-to-year variance of the yearly w")
    ax.grid(False)
    panel_label(ax, "d", "how much of the scatter in w is real?")
    fig.suptitle("Are the bootstrap bands trustworthy? Monte Carlo test on "
                 f"synthetic 30-year records ({cov_summary['n_records'].iat[0]} "
                 "per case)", x=0.0, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), w_pad=2.0, h_pad=2.0)
    fig.text(0.0, 0.0, "(a) bars: ±1.96 Monte Carlo standard errors.  "
             "(b) box: 25-75%, line: 5-95%, tick: median.  "
             "(c) markers: median, bars: 5-95% over synthetic records.  "
             "(c)-(d) Weibull fitted by PWM to excesses over 1 mm.",
             fontsize=5.8, color=INK["muted"], ha="left", va="bottom")
    return fig

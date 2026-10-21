"""
Shared visual style for every figure in the project.

Design rules
------------
* **Colour follows the entity, everywhere.** GEV is always blue and MEVD is
  always orange, in every task and figure. Stations are separated by panel or
  line style, never by re-using a method colour.
* **Quality-control status uses its own reserved colours.** It is always paired
  with a marker shape and a text label, so colour never carries meaning alone
  (colour-vision-deficiency safe, grayscale-print safe).
* **The data are the only loud element.** Gridlines are hairline and recessive,
  there are no top or right spines, and text is drawn in ink colours, never in
  series colours.
* **Publication output.** Figures are 180 mm (double column) or 88 mm (single
  column) wide, saved as 300-dpi PNG plus vector PDF, with units on every axis.

The categorical palette (blue, orange, aqua, in that fixed order) was checked
with a colour-vision-deficiency validator on a white surface:
all-pairs CVD dE >= 9.2 and normal-vision dE >= 24.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

# ----------------------------------------------------------------------
# Colour tokens
# ----------------------------------------------------------------------
INK = {
    "primary": "#0b0b0b",    # titles, key text
    "secondary": "#52514e",  # labels, neutral data marks
    "muted": "#898781",      # tick labels, notes
    "grid": "#e1e0d9",       # hairline gridlines
    "axis": "#c3c2b7",       # spines / baselines
    "surface": "#ffffff",    # figure background (print)
    "band": "#f3f2ef",       # era shading
}

#: Method identity: fixed across the whole project.
METHOD_COLORS = {
    "GEV": "#2a78d6",   # categorical slot 1 (blue)
    "MEV": "#eb6834",   # categorical slot 2 (orange)
    "REF": "#1baf7a",   # categorical slot 3 (aqua): a third reference series
}

#: Station identity is encoded by line style plus a direct label, so the
#: method colours keep their meaning in cross-station figures.
STATION_LINESTYLES = {"bologna": "-", "padova": (0, (5, 2))}
STATION_LABELS = {"bologna": "Bologna", "padova": "Padova"}

#: Quality-control status: colour + marker + label (never colour alone).
QC_STATUS = {
    "valid": {"color": INK["secondary"], "marker": "o", "label": "Valid year"},
    "review": {"color": "#fab219", "marker": "D",
               "label": "Flagged, retained (isolated)"},
    "systematic": {"color": "#d03b3b", "marker": "X",
                   "label": "Systematic pattern (excluded in QC scenario)"},
    "missing": {"color": "#e1e0d9", "marker": "s", "label": "Missing year"},
}

#: Single-hue sequential ramp (light -> dark) for magnitudes / heatmaps.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                   "#256abf", "#184f95", "#0d366b"]

# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
MM = 1 / 25.4                    # millimetres -> inches
WIDTH_DOUBLE = 180 * MM          # double-column journal width
WIDTH_SINGLE = 88 * MM           # single-column journal width


def use_style() -> None:
    """Apply the project-wide matplotlib style (call once per script)."""
    mpl.rcParams.update({
        # Fonts: system sans throughout, small journal sizes.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        # Ink.
        "text.color": INK["primary"],
        "axes.labelcolor": INK["secondary"],
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "axes.edgecolor": INK["axis"],
        # Recessive frame and grid.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": INK["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # Marks.
        "lines.linewidth": 1.3,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 3.5,
        "patch.linewidth": 0,
        "axes.prop_cycle": mpl.cycler(color=[METHOD_COLORS["GEV"],
                                             METHOD_COLORS["MEV"],
                                             METHOD_COLORS["REF"]]),
        # Legend.
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        # Surfaces and output.
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "figure.dpi": 110,              # on-screen (Spyder plots pane)
        "savefig.dpi": 300,             # publication raster
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,             # editable text in PDF/AI
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, name: str,
                formats: Sequence[str] = ("png", "pdf"),
                folder: Path | None = None) -> list[Path]:
    """Save ``fig`` as 300-dpi PNG and vector PDF; return the written paths."""
    folder = folder or config.FIGURES
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = folder / f"{name}.{ext}"
        # No creation timestamp in PDFs: re-running a script then produces
        # byte-identical files, so git only shows figures that really changed.
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(path, metadata=meta)
        paths.append(path)
    return paths


def panel_label(ax: plt.Axes, letter: str, title: str | None = None,
                pad: float = 4) -> None:
    """Write a bold panel letter, optionally followed by a short title."""
    text = f"$\\bf{{{letter}}}$" + (f"  {title}" if title else "")
    ax.set_title(text, loc="left", fontsize=8.5, fontweight="normal",
                 color=INK["primary"], pad=pad)


def shade_eras(ax: plt.Axes, eras: Iterable, label: bool = True,
               label_y: float = 1.0) -> None:
    """Shade alternate observational eras and optionally name them.

    ``eras`` is an iterable of objects with ``start``, ``end`` and
    ``short`` attributes (see ``metadata.Era``).
    """
    for i, era in enumerate(eras):
        if i % 2 == 0:
            ax.axvspan(era.start - 0.5, era.end + 0.5, color=INK["band"],
                       lw=0, zorder=0)
        if label:
            ax.text((era.start + era.end) / 2, label_y, era.short,
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=6, color=INK["muted"], clip_on=False)


def source_note(fig: plt.Figure, text: str) -> None:
    """Add a small provenance note under the figure."""
    fig.text(0.0, -0.01, text, ha="left", va="top", fontsize=6,
             color=INK["muted"], transform=fig.transFigure)

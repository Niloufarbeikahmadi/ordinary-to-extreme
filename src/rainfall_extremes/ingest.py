"""
Readers that turn each raw file into ONE canonical daily table.

Canonical daily schema (one row per calendar day, no gaps)
-----------------------------------------------------------
station      : str            station key ("bologna", "padova")
year         : int            calendar year of the observation
day_of_year  : int            position of the value within its year (1-based)
date         : datetime64     calendar date (NaT only where it is ambiguous)
precip_mm    : float          daily rainfall depth in mm; NaN = not observed

Design choices
--------------
* Missing information is never invented. Missing days or years become NaN
  rows, so completeness can be measured later; they are never filled.
* Raw values are never rounded or "corrected" here. Unit conversion is the
  only transformation, and it is checked against plausibility bounds.
* Every structural surprise (a wrong number of days, duplicate dates,
  negative values) raises an error or is recorded. Nothing is silently
  dropped.
"""
from __future__ import annotations

import calendar
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

_YEAR_LINE = re.compile(r"^\s*(\d{4})\s*$")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _days_in_year(year: int) -> int:
    """Number of days in a Gregorian calendar year."""
    return 366 if calendar.isleap(year) else 365


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found:\n  {path}\n"
            f"Copy the assignment files into {config.DATA_RAW} without "
            f"renaming them (see data/raw/README.md)."
        )


def _check_units(daily: pd.DataFrame, station: str) -> None:
    """Stop if the converted daily maximum is physically implausible."""
    lo, hi = config.PLAUSIBLE_DAILY_MAX_MM
    vmax = float(daily["precip_mm"].max())
    if not lo <= vmax <= hi:
        raise ValueError(
            f"{station}: largest daily value after unit conversion is "
            f"{vmax:.1f} mm, outside the plausible range [{lo}, {hi}] mm. "
            f"Check the unit conversion before continuing."
        )


def complete_calendar(daily: pd.DataFrame, station: str,
                      first_year: int, last_year: int) -> pd.DataFrame:
    """Add NaN rows for every calendar year absent from ``daily``.

    Years that are present are left untouched, so a calendar anomaly
    such as a 366-day non-leap year is preserved rather than "fixed".
    """
    present = set(daily["year"].unique())
    filler = []
    for year in range(first_year, last_year + 1):
        if year in present:
            continue
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        filler.append(pd.DataFrame({
            "station": station, "year": year,
            "day_of_year": np.arange(1, len(dates) + 1),
            "date": dates, "precip_mm": np.nan,
        }))
    out = pd.concat([daily, *filler], ignore_index=True) if filler else daily
    return (out.sort_values(["year", "day_of_year"])
               .reset_index(drop=True))


# ----------------------------------------------------------------------
# Bologna: GHCN-Daily extract (CSV, PRCP in tenths of mm)
# ----------------------------------------------------------------------
def read_bologna_ghcnd(path: Path | str | None = None) -> pd.DataFrame:
    """Read the Bologna GHCN-Daily CSV extract into the canonical schema.

    The GHCN-Daily convention stores PRCP in **tenths of a millimetre**, so
    values are multiplied by 0.1. A raw maximum of 1557 means 155.7 mm.
    Dates absent from the file are inserted as NaN (not observed).
    """
    path = Path(path) if path else config.DATA_RAW / config.RAW_FILES["bologna"]
    _require_file(path)
    raw = pd.read_csv(path, dtype={"DATE": str})

    stations = raw["STATION"].unique()
    if len(stations) != 1:
        raise ValueError(f"Expected one station, found {stations}.")
    if raw["DATE"].duplicated().any():
        raise ValueError("Duplicate dates in the Bologna file.")

    dates = pd.to_datetime(raw["DATE"], format="%Y%m%d")
    prcp = pd.to_numeric(raw["PRCP"], errors="coerce").astype(float)
    # GHCN-D uses -9999 for missing; treat any negative value as missing.
    n_negative = int((prcp < 0).sum())
    if n_negative:
        warnings.warn(f"Bologna: {n_negative} negative values set to NaN.")
    prcp = prcp.where(prcp >= 0)

    series = pd.Series(prcp.to_numpy() * config.GHCND_PRCP_SCALE,
                       index=dates).sort_index()
    # Reindex onto a complete daily calendar: absent dates -> NaN.
    full = pd.date_range(f"{series.index.min().year}-01-01",
                         f"{series.index.max().year}-12-31", freq="D")
    series = series.reindex(full)

    daily = pd.DataFrame({
        "station": "bologna",
        "year": full.year.astype(int),
        "day_of_year": full.dayofyear.astype(int),
        "date": full,
        "precip_mm": series.to_numpy(),
    })
    _check_units(daily, "bologna")
    return daily


# ----------------------------------------------------------------------
# Padova: Poleni-format text file (year label line + one value per line)
# ----------------------------------------------------------------------
def parse_poleni_blocks(lines: list[str]) -> dict[int, np.ndarray]:
    """Split the Poleni text format into ``{year: daily values}``.

    Format: a line holding only a 4-digit year (1600-2100) opens a block,
    followed by one daily value per line. A 4-digit line is accepted as a
    year label only at the start of the file or after at least 365 values,
    so a rainfall value can never be mistaken for a year.
    """
    blocks: dict[int, list[float]] = {}
    current: int | None = None
    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = _YEAR_LINE.match(line)
        is_label = (match is not None and 1600 <= int(match.group(1)) <= 2100
                    and (current is None or len(blocks[current]) >= 365))
        if is_label:
            current = int(match.group(1))
            if current in blocks:
                raise ValueError(f"Year {current} appears twice "
                                 f"(line {lineno}).")
            blocks[current] = []
            continue
        if current is None:
            raise ValueError(f"Value before first year label (line {lineno}).")
        try:
            blocks[current].append(float(line))
        except ValueError as err:
            raise ValueError(f"Non-numeric value {line!r} at line "
                             f"{lineno}.") from err
    return {y: np.asarray(v, dtype=float) for y, v in blocks.items()}


def _poleni_dates(year: int, n_values: int) -> pd.DatetimeIndex:
    """Calendar dates for one Poleni block.

    If the block length matches the Gregorian calendar, day k is Jan 1 + k-1.
    The file holds 366 values for 1800 and 1900, which are not Gregorian leap
    years. We assume the compiler applied the simple "divisible by 4" rule
    and inserted a 29 February: that position gets NaT and the later
    positions keep their true dates. The value itself is kept, because for
    annual statistics it is a real observation of that year.
    """
    expected = _days_in_year(year)
    start = pd.Timestamp(f"{year}-01-01")
    if n_values == expected:
        return pd.date_range(start, periods=n_values, freq="D")
    if n_values == 366 and expected == 365:
        pos = np.arange(1, 367)
        offsets = np.where(pos <= 59, pos - 1, pos - 2).astype(float)
        offsets[pos == 60] = np.nan                     # spurious 29 Feb
        return pd.DatetimeIndex([start + pd.Timedelta(days=o)
                                 if np.isfinite(o) else pd.NaT
                                 for o in offsets])
    raise ValueError(f"Padova {year}: {n_values} values, expected {expected}.")


def read_padova_poleni(path: Path | str | None = None) -> pd.DataFrame:
    """Read the Padova (Poleni format) file into the canonical schema.

    Values are already in millimetres. Years absent from the file become
    NaN rows over their Gregorian calendar.
    """
    path = Path(path) if path else config.DATA_RAW / config.RAW_FILES["padova"]
    _require_file(path)
    with open(path, "r", encoding="utf-8-sig") as fh:
        blocks = parse_poleni_blocks(fh.readlines())

    frames = []
    for year, values in sorted(blocks.items()):
        if (values < 0).any():
            raise ValueError(f"Padova {year}: negative rainfall values.")
        frames.append(pd.DataFrame({
            "station": "padova", "year": year,
            "day_of_year": np.arange(1, len(values) + 1),
            "date": _poleni_dates(year, len(values)),
            "precip_mm": values,
        }))
    daily = pd.concat(frames, ignore_index=True)
    daily = complete_calendar(daily, "padova", min(blocks), max(blocks))
    _check_units(daily, "padova")
    return daily


READERS = {"bologna": read_bologna_ghcnd, "padova": read_padova_poleni}


def load_station(station: str, path: Path | str | None = None) -> pd.DataFrame:
    """Read any supported station into the canonical daily schema."""
    try:
        reader = READERS[station]
    except KeyError as err:
        raise KeyError(f"Unknown station {station!r}; "
                       f"choose from {sorted(READERS)}.") from err
    return reader(path)

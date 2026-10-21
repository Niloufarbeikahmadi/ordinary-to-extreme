"""
Moving windows over the years of a record.

Two window definitions are supported, so that the effect of the definition
itself can be tested:

* ``calendar`` (main, task specification): windows of ``length`` CALENDAR
  years, advanced by ``step`` years. A window is retained only if it
  contains at least ``min_valid`` valid years, so missing or invalid years
  simply reduce the sample. Each window is indexed by its FINAL calendar
  year.
* ``consecutive`` (variant): windows of exactly ``length`` consecutive
  VALID years (missing years are skipped). Every sample therefore has the
  same size, but a window can span more than ``length`` calendar years
  across a gap. It is indexed by its last valid year.

Marani & Zanetti (2015) used 30-year windows with a 29-year overlap,
indexed by the START year. Both indices are stored so results can be
compared with their Figure 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from . import config


@dataclass(frozen=True)
class Window:
    start: int                 # first calendar year of the window
    end: int                   # final calendar year (index used in plots)
    years: tuple[int, ...]     # valid years inside the window
    mode: str

    @property
    def n_valid(self) -> int:
        return len(self.years)

    @property
    def n_calendar(self) -> int:
        return self.end - self.start + 1


def calendar_windows(valid_years: Iterable[int], first_year: int,
                     last_year: int, length: int = config.WINDOW_YEARS,
                     step: int = config.WINDOW_STEP,
                     min_valid: int = config.MIN_VALID_YEARS,
                     keep_all: bool = False) -> list[Window]:
    """Calendar-year windows; by default only those with >= ``min_valid``."""
    valid = set(int(y) for y in valid_years)
    out = []
    for start in range(first_year, last_year - length + 2, step):
        end = start + length - 1
        years = tuple(y for y in range(start, end + 1) if y in valid)
        if keep_all or len(years) >= min_valid:
            out.append(Window(start, end, years, "calendar"))
    return out


def consecutive_windows(valid_years: Iterable[int],
                        length: int = config.WINDOW_YEARS,
                        step: int = config.WINDOW_STEP) -> list[Window]:
    """Windows of exactly ``length`` consecutive valid years."""
    years = sorted(set(int(y) for y in valid_years))
    out = []
    for i in range(0, len(years) - length + 1, step):
        chunk = tuple(years[i:i + length])
        out.append(Window(chunk[0], chunk[-1], chunk, "consecutive"))
    return out


def build_windows(annual: pd.DataFrame, mode: str = "calendar",
                  **kwargs) -> list[Window]:
    """Windows for one station/scenario annual table (column ``valid``)."""
    valid_years = annual.loc[annual["valid"], "year"]
    if mode == "calendar":
        return calendar_windows(valid_years, int(annual["year"].min()),
                                int(annual["year"].max()), **kwargs)
    if mode == "consecutive":
        return consecutive_windows(valid_years, **kwargs)
    raise ValueError(f"Unknown window mode {mode!r}.")


def windows_table(windows: list[Window]) -> pd.DataFrame:
    """One row per window: start, end, number of valid years."""
    return pd.DataFrame({
        "window_start": [w.start for w in windows],
        "window_end": [w.end for w in windows],
        "n_valid": [w.n_valid for w in windows],
        "n_calendar": [w.n_calendar for w in windows],
        "mode": [w.mode for w in windows],
    })

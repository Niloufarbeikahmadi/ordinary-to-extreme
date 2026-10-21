"""
Project-wide configuration: paths, analysis constants and QC thresholds.

Every number that influences a result is defined here, once, with the
reason it was chosen. Change a value here and every script picks it up,
which is what makes the sensitivity analyses in later tasks honest.

Priority of sources when a choice has to be made
-------------------------------------------------
1. The task specification (``task description.txt``, Prof. M. Marani).
2. Marani & Zanetti (2015, WRR) - the original moving-window GEV study
   of the Padova series - for data handling and GEV choices.
3. Zorzetto, Botter & Marani (2016, GRL) and the ``mevpy`` reference
   implementation (E. Zorzetto) for MEVD choices.
4. Our own documented choices, always accompanied by a sensitivity run.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
#: Repository root (this file lives in <root>/src/rainfall_extremes/).
ROOT: Path = Path(__file__).resolve().parents[2]
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
OUTPUTS: Path = ROOT / "outputs"
FIGURES: Path = OUTPUTS / "figures"
TABLES: Path = OUTPUTS / "tables"

#: Raw file names exactly as distributed with the assignment.
#: Copy the two files into ``data/raw/`` without renaming them.
RAW_FILES: dict[str, str] = {
    "bologna": "Bologna Daily Rainfall_assignment_19 october 2026.csv",
    "padova": "polenidata_Assignment_19 october 2026.txt",
}

# ----------------------------------------------------------------------
# Analysis constants (task specification)
# ----------------------------------------------------------------------
#: A wet day is a day with rainfall STRICTLY greater than this depth [mm].
WET_THRESHOLD_MM: float = 1.0
#: Moving-window length [calendar years] and step [years].
WINDOW_YEARS: int = 30
WINDOW_STEP: int = 1
#: A window is retained only if it holds at least this many valid years.
MIN_VALID_YEARS: int = 20
#: Return period of interest [years].
RETURN_PERIOD: float = 200.0
#: Bootstrap repetitions and the pointwise interval percentiles.
N_BOOTSTRAP: int = 500
CI_PERCENTILES: tuple[float, float] = (5.0, 95.0)
#: Fixed seed so every figure in the report is exactly reproducible.
RANDOM_SEED: int = 20261020
#: Monte Carlo check of the bootstrap intervals (Task 4): number of synthetic
#: 30-year records per station and per synthetic "world". The Monte Carlo
#: standard error of a coverage near 0.90 is sqrt(0.9 * 0.1 / R), i.e. about
#: 0.013 for R = 500.
COVERAGE_RECORDS: int = 500
COVERAGE_YEARS: int = WINDOW_YEARS

# ----------------------------------------------------------------------
# Data-validity rule
# ----------------------------------------------------------------------
#: A year is "valid" when fewer than this fraction of its days are missing.
#: Source: Zorzetto et al. (2016, GRL), "only years with less than 10% of
#: missing daily data were considered". Sensitivity: 0.05 and 0.20.
MAX_MISSING_FRACTION: float = 0.10

# ----------------------------------------------------------------------
# Unit handling
# ----------------------------------------------------------------------
#: GHCN-Daily stores PRCP in tenths of a millimetre -> multiply by 0.1.
GHCND_PRCP_SCALE: float = 0.1
#: Sanity bounds on the largest daily value AFTER conversion [mm].
#: If the converted maximum falls outside them the unit is probably wrong
#: (e.g. the 1/10 conversion was applied twice or forgotten) and we stop.
PLAUSIBLE_DAILY_MAX_MM: tuple[float, float] = (30.0, 600.0)

# ----------------------------------------------------------------------
# Quality-control thresholds (screening flags, never automatic deletion)
# ----------------------------------------------------------------------
#: Robust z-scores are computed against the whole record (median / MAD).
#: "Accumulation" signature: few wet days AND heavy mean wet-day depth,
#: i.e. multi-day amounts reported on a single day (annual total preserved).
QC_Z_NWET_LOW: float = -1.5
QC_Z_INTENSITY_HIGH: float = 1.5
#: "Under-reporting" signature: few wet days AND a low annual total,
#: i.e. rain days possibly recorded as zero.
QC_Z_TOTAL_LOW: float = -2.0
QC_Z_NWET_VERY_LOW: float = -2.0
#: A suspicious pattern repeated in at least this many consecutive
#: available years is treated as a change in recording practice
#: (systematic), not as weather.
QC_MIN_SYSTEMATIC_RUN: int = 3
#: Informational: share of rain days (P > 0) with P <= 1 mm below which a
#: year is said to show a "small-rain deficit" (small amounts not recorded).
QC_SMALL_RAIN_SHARE: float = 0.05
#: Duplicate-block search: identical runs of this many days containing at
#: least ``DUP_MIN_RAIN_DAYS`` non-zero values (zeros alone are not evidence).
DUP_WINDOW_DAYS: int = 10
DUP_MIN_RAIN_DAYS: int = 6

#: "Isolated winter spike": a large daily value in winter preceded by a long
#: completely dry spell and followed by a dry day. This is the signature of
#: snowfall accumulated over several days and measured as water on a single
#: day. Flagged for review only (never removed).
SPIKE_MIN_MM: float = 50.0
SPIKE_DRY_DAYS_BEFORE: int = 7
SPIKE_MONTHS: tuple[int, ...] = (12, 1, 2)

#: Candidate recording resolutions [mm] used to fingerprint unit eras.
#: Values are the *rounded* conversion factors that appear in the files:
#: 0.0254 = 1/1000 English inch; 0.0188 and 0.188 = Paris "point" (1/12
#: ligne) and its tenth; 0.0226 = 1/100 Paris ligne; 0.094 = 1/24 ligne;
#: 0.1, 0.2, 0.5, 1.0 = metric gauge readings.
RESOLUTION_CANDIDATES_MM: tuple[float, ...] = (
    0.01, 0.0188, 0.0226, 0.0254, 0.094, 0.1, 0.188, 0.2, 0.5, 1.0,
)
#: Share of non-zero values that must be integer multiples of a candidate.
RESOLUTION_MIN_SHARE: float = 0.95


def ensure_output_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for folder in (DATA_PROCESSED, FIGURES, TABLES):
        folder.mkdir(parents=True, exist_ok=True)

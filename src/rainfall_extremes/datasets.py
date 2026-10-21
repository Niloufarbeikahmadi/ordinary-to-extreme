"""
Access to the processed tables produced by Task 1.

Later tasks read ``data/processed/{station}_annual.csv``. If that file is
missing (Task 1 has not been run), the QC pipeline is run in memory, so every
script also works on its own.
"""
from __future__ import annotations

import warnings

import pandas as pd

from . import config


def load_daily(station: str) -> pd.DataFrame:
    """Canonical daily table (with ``precip_mm`` and ``precip_mm_qc``)."""
    path = config.DATA_PROCESSED / f"{station}_daily.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    warnings.warn(f"{path.name} not found: running Task 1 QC in memory.")
    from . import ingest, qc
    return qc.run_qc(ingest.load_station(station)).daily


#: Daily column holding the values of each scenario.
SCENARIO_COLUMN = {"asis": "precip_mm", "qc": "precip_mm_qc"}


def load_annual(station: str, scenario: str | None = None) -> pd.DataFrame:
    """Annual table of one station; optionally a single scenario."""
    path = config.DATA_PROCESSED / f"{station}_annual.csv"
    if path.exists():
        annual = pd.read_csv(path, parse_dates=["date_of_max"])
    else:
        warnings.warn(f"{path.name} not found: running Task 1 QC in memory. "
                      f"Run scripts/01_ingest_and_qc.py to cache it.")
        from . import ingest, qc
        annual = qc.run_qc(ingest.load_station(station)).annual
    if scenario is not None:
        annual = annual.loc[annual["scenario"] == scenario].reset_index(drop=True)
        if annual.empty:
            raise ValueError(f"Unknown scenario {scenario!r}.")
    return annual

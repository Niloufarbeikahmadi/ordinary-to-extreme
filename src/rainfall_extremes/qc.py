"""
Quality control (QC) of daily rainfall records.

Philosophy
----------
QC here *screens and documents*; it does not silently edit data.
Each check produces a flag with a reason. Two analysis **scenarios** are
then derived, and every later task is run on both:

* ``asis`` (main analysis): the record exactly as provided. Years are valid
  when fewer than 10% of their days are missing (Zorzetto et al., 2016).
  This mirrors Marani & Zanetti (2015), who kept the full Padova record
  apart from documented gaps.
* ``qc`` (sensitivity): additionally (i) duplicated blocks of days are set
  to missing, both copies, because the data cannot tell which one is the
  original; and (ii) years belonging to a *systematic* suspicious pattern
  (a recording-practice change spanning at least 3 consecutive available
  years) are marked invalid.

Isolated suspicious years are flagged for review but retained in both
scenarios: a single odd year is more likely weather than a protocol change,
and extremes are exactly the values we must not discard on suspicion alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import config
from .ingest import _days_in_year


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------
def robust_z(values: pd.Series, reference: pd.Series | None = None) -> pd.Series:
    """Robust z-score: (x - median) / (1.4826 * MAD).

    The median and MAD (median absolute deviation) resist the very outliers
    we are trying to detect, unlike the mean and standard deviation. The
    factor 1.4826 makes the MAD comparable to a standard deviation under a
    normal distribution. ``reference`` (default: ``values``) sets the
    baseline.
    """
    ref = (values if reference is None else reference).dropna()
    med = float(np.median(ref))
    mad = 1.4826 * float(np.median(np.abs(ref - med)))
    if mad == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - med) / mad


def compress_years(years: Iterable[int]) -> str:
    """[1765, 1766, 1767, 1815] -> '1765-1767, 1815'."""
    years = sorted(set(int(y) for y in years))
    if not years:
        return "none"
    out, start, prev = [], years[0], years[0]
    for y in years[1:] + [None]:
        if y is not None and y == prev + 1:
            prev = y
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        if y is not None:
            start = prev = y
    return ", ".join(out)


def detect_resolution(values: np.ndarray,
                      candidates: Iterable[float] = config.RESOLUTION_CANDIDATES_MM,
                      min_share: float = config.RESOLUTION_MIN_SHARE,
                      tol: float = 0.02, min_n: int = 10) -> float:
    """Recording resolution of a set of positive rainfall values [mm].

    Returns the LARGEST candidate step ``q`` such that at least
    ``min_share`` of the values are integer multiples of ``q`` (within
    ``tol`` of an integer, in units of ``q``). Returns NaN when no candidate
    fits ("mixed" or irregular conversions) or when there are too few values.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < min_n:
        return np.nan
    for q in sorted(candidates, reverse=True):
        ratio = values / q
        share = np.mean(np.abs(ratio - np.round(ratio)) < tol)
        if share >= min_share:
            return float(q)
    return np.nan


# ----------------------------------------------------------------------
# Duplicated blocks
# ----------------------------------------------------------------------
def find_duplicate_blocks(daily: pd.DataFrame,
                          value_col: str = "precip_mm",
                          window: int = config.DUP_WINDOW_DAYS,
                          min_rain_days: int = config.DUP_MIN_RAIN_DAYS
                          ) -> pd.DataFrame:
    """Find runs of days that appear identically twice in the record.

    A sliding window of ``window`` days is hashed. Only windows with at
    least ``min_rain_days`` non-zero values are considered, because
    repeated zeros (dry spells) are not evidence of copying. Matching
    windows with a constant offset are merged into blocks, and each block
    is extended day by day while the two runs remain identical.
    """
    p = np.round(daily[value_col].to_numpy(float), 4)
    seen: dict[tuple, list[int]] = {}
    for i in range(len(p) - window + 1):
        seg = p[i:i + window]
        if np.isnan(seg).any() or (seg > 0).sum() < min_rain_days:
            continue
        seen.setdefault(tuple(seg), []).append(i)

    pairs = sorted({(v[0], v[j]) for v in seen.values() if len(v) > 1
                    for j in range(1, len(v))})
    blocks: list[list[int]] = []          # [first_start, repeat_start, n]
    for a, b in pairs:
        if blocks and b - a == blocks[-1][1] - blocks[-1][0] \
                and a <= blocks[-1][0] + blocks[-1][2]:
            blocks[-1][2] = a - blocks[-1][0] + 1
        else:
            blocks.append([a, b, 1])

    # Extend every piece to the full identical span, then merge pieces that
    # overlap with the same offset (a copied block can be found in several
    # pieces when some of its windows are too dry to be hashed).
    spans: list[tuple[int, int, int]] = []           # (start, end, offset)
    for a, b, n in blocks:
        length = n + window - 1
        while a > 0 and p[a - 1] == p[b - 1]:          # extend backwards
            a, b, length = a - 1, b - 1, length + 1
        while b + length < len(p) and p[a + length] == p[b + length]:
            length += 1                                  # extend forwards
        spans.append((a, a + length, b - a))
    merged: list[list[int]] = []
    for start, end, off in sorted(spans, key=lambda s: (s[2], s[0])):
        if merged and off == merged[-1][2] and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, off])

    rows = []
    for a, end, off in sorted(merged):
        b, length = a + off, end - a
        seg = p[a:a + length]
        rows.append({
            "first_index": a, "repeat_index": b, "length_days": length,
            "first_year": int(daily["year"].iat[a]),
            "first_day_of_year": int(daily["day_of_year"].iat[a]),
            "repeat_year": int(daily["year"].iat[b]),
            "repeat_day_of_year": int(daily["day_of_year"].iat[b]),
            "offset_days": b - a,
            "rain_days_in_block": int((seg > 0).sum()),
            "block_total_mm": float(np.nansum(seg)),
            "block_max_mm": float(np.nanmax(seg)),
        })
    return pd.DataFrame(rows)


def mask_blocks(daily: pd.DataFrame, blocks: pd.DataFrame,
                value_col: str = "precip_mm") -> pd.DataFrame:
    """Add ``is_duplicate`` and ``precip_mm_qc`` (both copies set to NaN)."""
    out = daily.copy()
    dup = np.zeros(len(out), dtype=bool)
    for _, blk in blocks.iterrows():
        for start in (blk.first_index, blk.repeat_index):
            dup[int(start):int(start) + int(blk.length_days)] = True
    out["is_duplicate"] = dup
    out["precip_mm_qc"] = out[value_col].where(~dup)
    return out


# ----------------------------------------------------------------------
# Isolated winter spikes (possible snow accumulation)
# ----------------------------------------------------------------------
def find_isolated_winter_spikes(daily: pd.DataFrame,
                                value_col: str = "precip_mm",
                                min_mm: float = config.SPIKE_MIN_MM,
                                dry_before: int = config.SPIKE_DRY_DAYS_BEFORE,
                                months: Iterable[int] = config.SPIKE_MONTHS
                                ) -> pd.DataFrame:
    """Large winter values surrounded by completely dry days.

    A real convective or frontal storm usually has rain on neighbouring days.
    A large value in winter, preceded by ``dry_before`` days of exactly zero
    and followed by a zero, is more consistent with snow that fell over
    several days being melted and measured on one day. That is a reporting
    artefact which would inflate a block maximum. The output is a review
    list, not a deletion list.
    """
    p = daily[value_col].to_numpy(float)
    month = daily["date"].dt.month.to_numpy()
    rows = []
    for i in np.flatnonzero(np.nan_to_num(p) >= min_mm):
        if i < dry_before or i + 1 >= len(p) or month[i] not in set(months):
            continue
        before, after = p[i - dry_before:i], p[i + 1]
        if np.all(before == 0) and after == 0:
            rows.append({"date": daily["date"].iat[i],
                         "year": int(daily["year"].iat[i]),
                         "precip_mm": float(p[i])})
    return pd.DataFrame(rows, columns=["date", "year", "precip_mm"])


# ----------------------------------------------------------------------
# Annual summary
# ----------------------------------------------------------------------
def annual_summary(daily: pd.DataFrame, value_col: str = "precip_mm",
                   wet_threshold: float = config.WET_THRESHOLD_MM
                   ) -> pd.DataFrame:
    """One row per calendar year with the statistics used by QC and EVA.

    ``n_wet`` counts days with P > ``wet_threshold`` (strictly greater, as
    the task specifies). It is the "N" of the MEVD, so its homogeneity
    matters directly for the MEVD estimates. ``annual_max_mm`` is the block
    maximum used by the GEV. Statistics are NaN for years with no
    observations.
    """
    rows = []
    for year, grp in daily.groupby("year", sort=True):
        p = grp[value_col].to_numpy(float)
        ok = np.isfinite(p)
        pv = p[ok]
        n_days, n_valid = len(p), int(ok.sum())
        rain, wet = pv[pv > 0], pv[pv > wet_threshold]
        has = n_valid > 0
        imax = int(np.nanargmax(p)) if has else None
        rows.append({
            "year": int(year),
            "n_days": n_days,
            "calendar_ok": n_days == _days_in_year(int(year)),
            "n_valid": n_valid,
            "frac_missing": 1.0 - n_valid / n_days,
            "total_mm": float(pv.sum()) if has else np.nan,
            "annual_max_mm": float(pv.max()) if has else np.nan,
            "date_of_max": grp["date"].iat[imax] if has else pd.NaT,
            "doy_of_max": int(grp["day_of_year"].iat[imax]) if has else np.nan,
            "n_rain": int(rain.size) if has else np.nan,
            "n_wet": int(wet.size) if has else np.nan,
            "n_small": int(rain.size - wet.size) if has else np.nan,
            "frac_small": (rain.size - wet.size) / rain.size
                          if rain.size else np.nan,
            "mean_wet_mm": float(wet.mean()) if wet.size else np.nan,
            "min_wet_value_mm": float(wet.min()) if wet.size else np.nan,
            "resolution_mm": detect_resolution(rain),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Year-level flags
# ----------------------------------------------------------------------
def flag_years(annual: pd.DataFrame, duplicate_years: Iterable[int] = (),
               spike_years: Iterable[int] = ()) -> pd.DataFrame:
    """Add QC flags, robust z-scores and a status to the ``asis`` table.

    Flags (all screening signals; none deletes data by itself):

    * ``flag_incomplete``  - >= 10% of days missing (Zorzetto et al., 2016).
    * ``flag_calendar``    - day count differs from the Gregorian calendar.
    * ``flag_duplicate``   - year contains a duplicated block of days.
    * ``flag_accumulation``- few wet days AND large mean wet-day depth:
      the signature of multi-day amounts reported on a single day. The
      annual total is preserved but the wet-day count N falls, which biases
      the MEVD directly.
    * ``flag_underreport`` - few wet days AND low annual total: rain days
      possibly recorded as zero.
    * ``flag_small_rain_deficit`` (informational) - almost no amounts
      <= 1 mm recorded (e.g. coarse gauge or large funnel). The 1 mm wet-day
      threshold already shields the analysis from most of this effect.
    * ``flag_systematic``  - accumulation/under-report flags repeated over at
      least ``QC_MIN_SYSTEMATIC_RUN`` consecutive *available* years (missing
      years do not break a run): treated as a recording-practice change.
    * ``flag_winter_spike`` - the year's ANNUAL MAXIMUM is an isolated winter
      spike (possible snow accumulation); it matters for the GEV.
    """
    a = annual.copy()
    observed = a["n_valid"] > 0
    reference = a.loc[observed & (a["frac_missing"] < config.MAX_MISSING_FRACTION)]

    a["z_nwet"] = robust_z(a["n_wet"], reference["n_wet"])
    a["z_log_mean_wet"] = robust_z(np.log(a["mean_wet_mm"]),
                                   np.log(reference["mean_wet_mm"]))
    a["z_total"] = robust_z(a["total_mm"], reference["total_mm"])

    a["flag_incomplete"] = observed & (a["frac_missing"] >= config.MAX_MISSING_FRACTION)
    a["flag_calendar"] = observed & ~a["calendar_ok"]
    a["flag_duplicate"] = a["year"].isin(set(duplicate_years))
    a["flag_accumulation"] = ((a["z_nwet"] <= config.QC_Z_NWET_LOW)
                              & (a["z_log_mean_wet"] >= config.QC_Z_INTENSITY_HIGH))
    a["flag_underreport"] = ((a["z_nwet"] <= config.QC_Z_NWET_VERY_LOW)
                             & (a["z_total"] <= config.QC_Z_TOTAL_LOW))
    a["flag_small_rain_deficit"] = observed & (a["frac_small"]
                                               < config.QC_SMALL_RAIN_SHARE)
    a["flag_winter_spike"] = a["year"].isin(set(spike_years))

    # Run length of suspicious years along the sequence of AVAILABLE years.
    suspect = (a["flag_accumulation"] | a["flag_underreport"]) & observed
    seq = a.loc[observed, ["year"]].assign(s=suspect[observed].to_numpy())
    run_id = (seq["s"] != seq["s"].shift()).cumsum()
    run_len = seq.groupby(run_id)["s"].transform("size")
    systematic = seq["s"] & (run_len >= config.QC_MIN_SYSTEMATIC_RUN)
    a["flag_systematic"] = False
    a.loc[seq.index, "flag_systematic"] = systematic.to_numpy()

    # Status used for figures and reports (one per year).
    status = np.where(~observed | a["flag_incomplete"], "missing",
             np.where(a["flag_systematic"], "systematic",
             np.where(suspect | a["flag_duplicate"] | a["flag_winter_spike"],
                      "review", "valid")))
    a["status"] = status

    reasons = []
    for _, r in a.iterrows():
        why = []
        if r["n_valid"] == 0:
            why.append("year absent from record")
        if r["flag_incomplete"]:
            why.append(f"{r['frac_missing']:.0%} days missing")
        if r["flag_calendar"]:
            why.append(f"{r['n_days']} days in a non-leap year")
        if r["flag_duplicate"]:
            why.append("contains duplicated block")
        if r["flag_accumulation"]:
            why.append("few heavy wet days (accumulation pattern)")
        if r["flag_underreport"]:
            why.append("few wet days and low total (under-reporting?)")
        if r["flag_systematic"]:
            why.append("part of a systematic multi-year pattern")
        if r["flag_winter_spike"]:
            why.append("annual max is an isolated winter spike "
                       "(snow accumulation?)")
        if r["flag_small_rain_deficit"]:
            why.append("small amounts <=1 mm almost absent")
        reasons.append("; ".join(why))
    a["qc_reason"] = reasons
    return a


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
FLAG_COLUMNS = ["z_nwet", "z_log_mean_wet", "z_total", "flag_incomplete",
                "flag_calendar", "flag_duplicate", "flag_accumulation",
                "flag_underreport", "flag_small_rain_deficit",
                "flag_winter_spike", "flag_systematic", "status",
                "qc_reason"]


@dataclass
class QCResult:
    """Everything produced by the QC of one station."""
    station: str
    daily: pd.DataFrame          # canonical daily + is_duplicate, precip_mm_qc
    annual: pd.DataFrame         # long table: one row per (scenario, year)
    duplicates: pd.DataFrame     # one row per duplicated block
    spikes: pd.DataFrame         # isolated winter spikes (review list)

    def scenario(self, name: str) -> pd.DataFrame:
        """Annual table for one scenario ('asis' or 'qc')."""
        return self.annual.loc[self.annual["scenario"] == name].reset_index(drop=True)


def run_qc(daily: pd.DataFrame) -> QCResult:
    """Run every QC step on one canonical daily table."""
    station = str(daily["station"].iat[0])
    blocks = find_duplicate_blocks(daily)
    daily = mask_blocks(daily, blocks)

    dup_years: set[int] = set()
    for b in blocks.itertuples():
        dup_years |= {int(b.first_year), int(b.repeat_year)}

    asis = annual_summary(daily, "precip_mm")
    spikes = find_isolated_winter_spikes(daily)
    # Only spikes that ARE the annual maximum change a block-maxima fit.
    spike_max_years = set(spikes.merge(
        asis[["year", "annual_max_mm"]],
        left_on=["year", "precip_mm"], right_on=["year", "annual_max_mm"]
    )["year"])
    spikes["is_annual_max"] = spikes["year"].isin(spike_max_years)
    asis = flag_years(asis, dup_years, spike_max_years)
    asis["valid"] = (asis["n_valid"] > 0) & ~asis["flag_incomplete"]

    qc = annual_summary(daily, "precip_mm_qc")
    qc = qc.merge(asis[["year", *FLAG_COLUMNS]], on="year", how="left")
    qc["valid"] = ((qc["n_valid"] > 0)
                   & (qc["frac_missing"] < config.MAX_MISSING_FRACTION)
                   & ~qc["flag_systematic"])

    annual = pd.concat([asis.assign(scenario="asis"),
                        qc.assign(scenario="qc")], ignore_index=True)
    annual.insert(0, "station", station)
    first = ["station", "scenario", "year", "valid"]
    annual = annual[first + [c for c in annual.columns if c not in first]]
    return QCResult(station, daily, annual, blocks, spikes)


# ----------------------------------------------------------------------
# Plain-text report (feeds the "Data engineering" section of the report)
# ----------------------------------------------------------------------
def _resolution_eras(asis: pd.DataFrame) -> list[str]:
    obs = asis.loc[asis["n_valid"] > 0, ["year", "resolution_mm"]].copy()
    obs["label"] = obs["resolution_mm"].map(
        lambda q: "mixed / irregular" if pd.isna(q) else f"{q:g} mm")
    run = (obs["label"] != obs["label"].shift()).cumsum()
    lines = []
    for _, g in obs.groupby(run):
        lines.append(f"{compress_years(g['year'])}: {g['label'].iat[0]}")
    return lines


def qc_report_markdown(res: QCResult) -> str:
    """Human-readable QC summary for one station (Markdown)."""
    a = res.scenario("asis")
    q = res.scenario("qc")
    obs = a[a["n_valid"] > 0]
    lines = [f"## {res.station.capitalize()}", ""]
    lines += [
        f"- Period on file: {int(obs['year'].min())}-{int(obs['year'].max())} "
        f"({len(obs)} years with data, {int((a['n_valid'] == 0).sum())} "
        f"years absent).",
        f"- Absent years: {compress_years(a.loc[a['n_valid'] == 0, 'year'])}.",
        f"- Missing days inside observed years: "
        f"{int((obs['n_days'] - obs['n_valid']).sum())}.",
        f"- Calendar anomalies: "
        f"{compress_years(a.loc[a['flag_calendar'], 'year'])}.",
        f"- Wet-day threshold: P > {config.WET_THRESHOLD_MM:g} mm; "
        f"median wet days/yr = {obs['n_wet'].median():.0f}; "
        f"median annual total = {obs['total_mm'].median():.0f} mm.",
        "",
        "### Duplicated blocks",
    ]
    if res.duplicates.empty:
        lines.append("- None found.")
    for b in res.duplicates.itertuples():
        lines.append(
            f"- {b.length_days} identical days: {b.first_year} (day "
            f"{b.first_day_of_year}) repeated in {b.repeat_year} (day "
            f"{b.repeat_day_of_year}); {b.rain_days_in_block} rain days, "
            f"block max {b.block_max_mm:.1f} mm.")
    lines += ["", f"### Isolated winter spikes (>= {config.SPIKE_MIN_MM:g} mm, "
              f"{config.SPIKE_DRY_DAYS_BEFORE} dry days before, dry day after)"]
    if res.spikes.empty:
        lines.append("- None found.")
    for s in res.spikes.itertuples():
        tag = " - annual maximum" if s.is_annual_max else ""
        lines.append(f"- {s.date.date()}: {s.precip_mm:.1f} mm{tag}")
    lines += ["", "### Recording resolution (detected)"]
    lines += [f"- {s}" for s in _resolution_eras(a)]
    lines += ["", "### Flags"]
    for col, label in [("flag_accumulation", "Accumulation pattern"),
                       ("flag_underreport", "Possible under-reporting"),
                       ("flag_systematic", "Systematic (excluded in QC scenario)"),
                       ("flag_duplicate", "Contains duplicated block"),
                       ("flag_winter_spike", "Annual max is a winter spike"),
                       ("flag_small_rain_deficit", "Small-rain deficit (info)")]:
        lines.append(f"- {label}: {compress_years(a.loc[a[col], 'year'])}")
    lines += [
        "",
        "### Valid years per scenario",
        f"- asis: {int(a['valid'].sum())} | qc: {int(q['valid'].sum())}",
        "",
        "### Five largest daily values (asis)",
    ]
    top = (res.daily.nlargest(5, "precip_mm")
           [["date", "year", "day_of_year", "precip_mm"]])
    for _, r in top.iterrows():
        when = r["date"].date() if pd.notna(r["date"]) else f"{r['year']} d{r['day_of_year']}"
        lines.append(f"- {when}: {r['precip_mm']:.1f} mm")
    return "\n".join(lines) + "\n"

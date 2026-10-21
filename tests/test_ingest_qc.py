"""Unit tests for ingestion and QC (run with: pytest)."""
import numpy as np
import pandas as pd
import pytest

from rainfall_extremes import ingest, qc


def _poleni_text(years: dict[int, list[float]]) -> list[str]:
    lines = []
    for y, vals in years.items():
        lines.append(f"{y}\r\n")
        lines += [f"{v}\r\n" for v in vals]
    return lines


def test_poleni_parser_separates_years_and_values():
    data = {1901: [0.0] * 364 + [12.5], 1902: [3.0] * 365}
    blocks = ingest.parse_poleni_blocks(_poleni_text(data))
    assert sorted(blocks) == [1901, 1902]
    assert len(blocks[1901]) == 365 and blocks[1901][-1] == 12.5


def test_poleni_value_never_mistaken_for_year():
    # a value "1999" inside a block must not open a new year (fewer than
    # 365 values so far) - realistic rainfall never reaches this, but the
    # parser must be safe by construction
    vals = [0.0] * 100 + [1999.0] + [0.0] * 264
    blocks = ingest.parse_poleni_blocks(_poleni_text({1901: vals}))
    assert list(blocks) == [1901] and len(blocks[1901]) == 365


def test_poleni_dates_handle_spurious_feb29():
    dates = ingest._poleni_dates(1900, 366)          # 1900 is not leap
    assert pd.isna(dates[59])                        # inserted 29 Feb
    assert dates[60] == pd.Timestamp("1900-03-01")
    assert dates[-1] == pd.Timestamp("1900-12-31")
    with pytest.raises(ValueError):
        ingest._poleni_dates(1901, 300)


def test_detect_resolution():
    rng = np.random.default_rng(0)
    vals = rng.integers(1, 400, 200) * 0.188
    assert qc.detect_resolution(vals) == pytest.approx(0.188)
    vals = rng.integers(1, 400, 200) * 0.1
    assert qc.detect_resolution(vals) == pytest.approx(0.1)


def test_duplicate_block_is_found_and_masked():
    rng = np.random.default_rng(1)
    p = np.where(rng.random(800) < 0.4, rng.gamma(0.8, 10, 800), 0.0)
    p[500:530] = p[100:130]                          # plant a 30-day copy
    p[100] = p[500] = 7.7
    daily = pd.DataFrame({"station": "x", "year": 2000 + np.arange(800) // 400,
                          "day_of_year": np.arange(800) % 400 + 1,
                          "date": pd.NaT, "precip_mm": p})
    blocks = qc.find_duplicate_blocks(daily)
    assert len(blocks) == 1
    b = blocks.iloc[0]
    assert b.first_index <= 100 and b.repeat_index - b.first_index == 400
    assert b.length_days >= 30
    masked = qc.mask_blocks(daily, blocks)
    assert masked["precip_mm_qc"].isna().sum() == 2 * b.length_days


def test_robust_z_and_compress_years():
    s = pd.Series([10, 11, 9, 10, 50.0])
    z = qc.robust_z(s)
    assert z.iloc[-1] > 10 and abs(z.iloc[0]) < 1e-12
    assert qc.compress_years([1765, 1766, 1767, 1815, 1817]) == \
        "1765-1767, 1815, 1817"

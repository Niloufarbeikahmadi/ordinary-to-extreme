"""Unit tests for the Task 5 modules (run with: pytest)."""
import numpy as np
import pandas as pd
import pytest

from rainfall_extremes import estimation_window as ewm
from rainfall_extremes import mev, sensitivity


@pytest.fixture
def excesses():
    rng = np.random.default_rng(11)
    return {y: np.sort(9.0 * rng.weibull(0.85, size=rng.integers(50, 90)))
            for y in range(1901, 1931)}


def test_blocks_keep_every_year_once():
    years = list(range(1901, 1928))                    # 27 valid years
    for ew in (1, 3, 5, "all"):
        b = ewm.blocks(years, ew)
        assert sum(b, []) == years
    assert [len(x) for x in ewm.blocks(years, 5)] == [5, 5, 5, 5, 7]


def test_ew1_reproduces_yearly_fits(excesses):
    years = sorted(excesses)
    N, C, w = ewm.window_parameters(years, excesses, 1)
    for k, y in enumerate(years):
        f = mev.weibull_fit_pwm(excesses[y])
        assert (N[k], C[k], w[k]) == pytest.approx((excesses[y].size, f.C, f.w))


def test_smev_uses_mean_number_of_wet_days(excesses):
    years = sorted(excesses)
    N, C, w = ewm.window_parameters(years, excesses, "smev")
    assert N.size == 1
    assert N[0] == pytest.approx(np.mean([excesses[y].size for y in years]))


def test_weighted_pwm_equals_pwm_of_concatenated_resample(excesses):
    years = sorted(excesses)
    xs, lab, N = ewm.pooled_window_data(years, excesses)
    idx = np.random.default_rng(2).integers(0, len(years), size=(6, len(years)))
    C, w = ewm.pooled_pwm_weighted(xs, lab, ewm.counts_from_indices(idx, len(years)))
    for b in range(6):
        f = mev.weibull_fit_pwm(np.concatenate([excesses[years[i]] for i in idx[b]]))
        assert C[b] == pytest.approx(f.C, rel=1e-12)
        assert w[b] == pytest.approx(f.w, rel=1e-12)


def test_one_at_a_time_is_zero_for_an_identical_variant():
    rows = [{"station": "s", "model": "MEV", "variant": v, "window_end": e,
             "rl_200": 100.0 + e % 7} for v in ("main", "copy")
            for e in range(1950, 1960)]
    oat = sensitivity.one_at_a_time(pd.DataFrame(rows))
    assert oat["max_abs_change_mm"].iat[0] == 0.0
    assert oat["n_windows"].iat[0] == 10

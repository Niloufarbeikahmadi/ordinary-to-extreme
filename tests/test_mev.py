"""Unit tests for the MEVD-Weibull module (run with: pytest)."""
import numpy as np
import pandas as pd
import pytest

from rainfall_extremes import mev


@pytest.fixture
def weibull_sample():
    rng = np.random.default_rng(7)
    return 8.0 * rng.weibull(0.8, size=20000)        # C = 8 mm, w = 0.8


def test_pwm_recovers_weibull_parameters(weibull_sample):
    fit = mev.weibull_fit_pwm(weibull_sample)
    assert fit.C == pytest.approx(8.0, rel=0.03)
    assert fit.w == pytest.approx(0.8, rel=0.03)


def test_ml_recovers_weibull_parameters(weibull_sample):
    fit = mev.weibull_fit_ml(weibull_sample)
    assert fit.C == pytest.approx(8.0, rel=0.03)
    assert fit.w == pytest.approx(0.8, rel=0.03)


def test_single_year_mev_is_weibull_power():
    # one year: zeta(x) = F(x - theta)^N exactly, so the quantile is analytic
    N, C, w, theta = 80, 9.0, 0.85, 1.0
    F = 1 - 1 / 200
    expected = theta + mev.weibull_quantile(F ** (1 / N), C, w)
    got = mev.mev_quantile(F, [N], [C], [w], threshold=theta)
    assert got == pytest.approx(expected, rel=1e-9)


def test_mev_cdf_is_monotonic_and_bounded():
    N, C, w = [60, 90, 75], [7.0, 10.0, 8.5], [0.8, 0.9, 0.75]
    x = np.linspace(0, 500, 400)
    z = mev.mev_cdf(x, N, C, w)
    assert np.all(np.diff(z) >= 0) and z[0] == 0 and z[-1] <= 1


def test_raw_formulation_has_no_shift():
    N, C, w = [70], [10.0], [0.9]
    a = mev.mev_return_level(100, N, C, w, formulation="excess")
    b = mev.mev_return_level(100, N, C, w, formulation="raw")
    assert a - b == pytest.approx(1.0, abs=1e-8)


def test_yearly_parameters_counts_strictly_above_threshold():
    days = pd.DataFrame({"year": [2000] * 15,
                         "precip_mm": [0, 1.0, 1.0, 2, 3, 4, 5, 6, 7, 8, 9,
                                       10, 12, 15, 30]})
    yp = mev.yearly_parameters(days)
    assert yp.loc[0, "N"] == 12            # 1.0 mm is NOT a wet day
    assert yp.loc[0, "annual_max_mm"] == 30

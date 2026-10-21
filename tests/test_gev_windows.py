"""Unit tests for the GEV module and the window builder (run with: pytest)."""
import numpy as np
import pytest
from scipy.stats import genextreme

from rainfall_extremes import gev, windows


@pytest.fixture
def sample():
    # 30 "annual maxima" from a known heavy-tailed GEV (xi = +0.15)
    return genextreme.rvs(c=-0.15, loc=45, scale=14, size=30,
                          random_state=np.random.default_rng(42))


def test_quantile_and_cdf_are_inverse():
    for xi in (-0.3, 0.0, 0.2):
        F = np.array([0.1, 0.5, 0.9, 0.995])
        q = gev.quantile(F, xi, 12.0, 40.0)
        assert np.allclose(gev.cdf(q, xi, 12.0, 40.0), F)


def test_sign_convention_matches_scipy():
    # scipy uses c = -xi
    xi, s, m = 0.2, 10.0, 50.0
    assert gev.quantile(0.995, xi, s, m) == pytest.approx(
        genextreme.ppf(0.995, c=-xi, loc=m, scale=s))


def test_return_level_definition():
    p = gev.GEVParams(0.1, 15.0, 45.0)
    assert gev.return_level(200, p) == pytest.approx(
        gev.quantile(1 - 1 / 200, 0.1, 15.0, 45.0))


def test_ml_is_never_worse_than_scipy(sample):
    ours = gev.fit_ml(sample)
    c, loc, scale = genextreme.fit(sample)
    assert ours.converged
    assert ours.nll <= gev.negloglik((-c, scale, loc), sample) + 1e-6


def test_lmom_known_values():
    # Gumbel-like sample: L-skewness of a Gumbel is 0.1699 -> xi ~ 0
    x = genextreme.rvs(c=0.0, loc=40, scale=10, size=20000,
                       random_state=np.random.default_rng(0))
    p = gev.fit_lmom(x)
    assert abs(p.xi) < 0.02
    assert p.sigma == pytest.approx(10, rel=0.03)
    assert p.mu == pytest.approx(40, rel=0.01)


def test_calendar_windows_rule():
    valid = [y for y in range(1900, 1960) if not 1910 <= y <= 1921]
    w = windows.calendar_windows(valid, 1900, 1959, length=30, min_valid=20)
    ends = [x.end for x in w]
    # windows containing the full 12-year gap have 18 valid years: dropped
    assert 1929 not in ends and 1940 not in ends
    assert all(x.n_valid >= 20 for x in w)
    assert all(x.end - x.start == 29 for x in w)


def test_consecutive_windows_have_exact_size():
    valid = [y for y in range(1900, 1960) if y % 7]
    w = windows.consecutive_windows(valid, length=30)
    assert all(x.n_valid == 30 for x in w)
    assert w[0].years == tuple(sorted(valid)[:30])

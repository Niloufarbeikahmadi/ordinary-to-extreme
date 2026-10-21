"""Unit tests for the batched fitters and the common bootstrap (run with: pytest)."""
import numpy as np
import pytest

from rainfall_extremes import bootstrap, config, gev, gev_batch, mev


@pytest.fixture
def gev_samples():
    rng = np.random.default_rng(42)
    # 20 samples of 30 annual maxima from a rainfall-like GEV
    u = rng.uniform(size=(20, 30))
    return gev.quantile(u, 0.1, 15.0, 55.0)


def test_analytic_hessian_matches_finite_differences(gev_samples):
    x = gev_samples[:3]
    theta = np.array([[55.0, np.log(15.0), 0.1],
                      [54.0, np.log(16.0), -0.2],
                      [56.0, np.log(14.0), 0.3]])
    _, H = gev_batch._grad_hess(theta, x)
    Hn = np.empty_like(H)
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1e-6 * (15.0 if k == 0 else 1.0)
        Hn[:, :, k] = (gev_batch._grad(theta + e, x)
                       - gev_batch._grad(theta - e, x)) / (2 * e[k])
    assert np.allclose(H, Hn, rtol=1e-5, atol=1e-6)


def test_batched_gev_matches_reference_fitter(gev_samples):
    fast = gev_batch.fit_ml(gev_samples)
    for b, x in enumerate(gev_samples):
        ref = gev.fit_ml(x)
        if not (-0.5 < ref.xi < 1.0):            # compare regular fits only
            continue
        assert fast.nll[b] <= ref.nll + 1e-6
        assert fast.return_level(200.0)[b] == pytest.approx(
            float(gev.return_level(200.0, ref)), abs=1e-3)


def test_degenerate_resample_stays_within_shape_bounds():
    # many ties at the minimum: the classic route to xi -> infinity
    x = np.array([33.0] * 4 + [33.4] * 4 + [33.8] * 3 + [36, 37.3, 40.3, 42.2,
                  43.3, 44.3, 44.3, 44.9, 49.3, 55.4, 63.3, 63.3, 63.3, 65.8,
                  68.1, 68.1, 68.1, 78.8, 78.8])
    fit = gev_batch.fit_ml(x[None, :])
    assert gev_batch.XI_MIN <= fit.xi[0] <= gev_batch.XI_MAX
    assert np.isfinite(fit.return_level(200.0)[0])


def test_batched_mev_quantile_matches_scalar_solver():
    rng = np.random.default_rng(3)
    N = rng.integers(40, 110, size=(25, 30)).astype(float)
    C = rng.uniform(6, 14, size=(25, 30))
    w = rng.uniform(0.7, 1.1, size=(25, 30))
    F = 1 - 1 / 200
    fast = mev.mev_quantile_batch(F, N, C, w)
    slow = np.array([mev.mev_quantile(F, N[i], C[i], w[i]) for i in range(25)])
    assert np.max(np.abs(fast - slow)) < 1e-6


def test_resampling_is_reproducible_and_window_specific():
    a = bootstrap.resample_indices(30, 50, "padova", 1900)
    b = bootstrap.resample_indices(30, 50, "padova", 1900)
    c = bootstrap.resample_indices(30, 50, "padova", 1901)
    d = bootstrap.resample_indices(30, 50, "bologna", 1900)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c) and not np.array_equal(a, d)
    assert bootstrap.station_code("padova") == bootstrap.station_code("padova")


def test_both_models_see_the_same_resampled_years():
    rng = np.random.default_rng(9)
    n = 25
    ws = bootstrap.WindowSample(
        station="test", start=1901, end=1925, years=np.arange(1901, 1926),
        amax=gev.quantile(rng.uniform(size=n), 0.1, 15.0, 55.0),
        N=rng.integers(50, 100, size=n).astype(float),
        C=rng.uniform(7, 11, size=n), w=rng.uniform(0.75, 1.0, size=n))
    B = 40
    bw = bootstrap.bootstrap_window(ws, B=B)
    idx = bootstrap.resample_indices(n, B, "test", 1925)
    F = 1 - 1 / config.RETURN_PERIOD
    # MEVD replicates are exactly the MEVD of the drawn (N, C, w) triplets ...
    expected_mev = mev.mev_quantile_batch(F, ws.N[idx], ws.C[idx], ws.w[idx])
    assert np.allclose(bw.rl_mev, expected_mev)
    # ... and GEV replicates are fits to the maxima of the SAME drawn years
    p0 = gev.fit_ml(ws.amax)
    expected_gev = gev_batch.fit_ml(ws.amax[idx], start=p0.as_tuple()
                                    ).return_level(config.RETURN_PERIOD)
    assert np.allclose(bw.rl_gev, expected_gev)


def test_wettest_year_groups_cover_all_resamples():
    rng = np.random.default_rng(5)
    n, B = 30, 2000
    ws = bootstrap.WindowSample(
        station="test", start=1901, end=1930, years=np.arange(1901, 1931),
        amax=rng.uniform(40, 80, size=n), N=np.full(n, 70.0),
        C=np.full(n, 9.0), w=np.full(n, 0.85))
    rl = rng.normal(150, 10, size=B)
    eff = bootstrap.largest_year_effect(ws, rl, rl)
    assert eff["n_resamples"].sum() == B
    # a given year is missing from about (1 - 1/n)^n = 36% of the resamples
    share0 = eff.loc[eff["copies"] == "0", "n_resamples"].iat[0] / B
    assert share0 == pytest.approx((1 - 1 / n) ** n, abs=0.03)

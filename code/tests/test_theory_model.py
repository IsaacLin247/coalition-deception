"""Analytical model of credibility collapse (theory/simulate_model.py): closed forms vs Monte Carlo."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))
import simulate_model as sm  # noqa: E402

TRIALS = 3000


def test_mean_is_unbiased_with_variance_sigma2_over_N_without_attack():
    n, m = 5, 2
    r = sm.simulate(n=n, m=m, rho=0.0, b=0.0, sigma=1.0, sigma_c=1.0, trials=TRIALS, seed=1)
    assert abs(r["bias_mean"]) < 0.05
    assert r["mse_mean"] == pytest.approx(1.0 / (n + m), rel=0.12)
    # the mean is BLUE for i.i.d. Gaussian reports: no reweighting can beat it in expectation
    assert r["mse_credibility"] >= r["mse_mean"] * 0.97
    assert r["mse_median"] >= r["mse_mean"] * 0.97


def test_rho_zero_reduces_coalition_to_independent_biased_reporters():
    """At rho = 0 the coalition reports are i.i.d.: its members corroborate each other no more
    than two honest reporters do (p_CC = p_HH when sigma_c = sigma)."""
    assert sm.p_cc(0.0, 1.0, 0.4) == pytest.approx(sm.p_hh(1.0, 0.4))
    r = sm.simulate(n=5, m=2, rho=0.0, b=4.0, trials=TRIALS, seed=2)
    assert r["p_cc_empirical"] == pytest.approx(sm.p_cc(0.0, 1.0, 0.4), abs=0.03)
    assert r["p_hh_empirical"] == pytest.approx(sm.p_hh(1.0, 0.4), abs=0.03)


def test_rho_one_makes_coalition_reports_identical():
    assert sm.p_cc(1.0, 1.0, 0.4) == 1.0
    r = sm.simulate(n=5, m=2, rho=1.0, b=4.0, trials=500, seed=3)
    assert r["p_cc_empirical"] == pytest.approx(1.0)


def test_closed_form_agreement_probabilities_match_monte_carlo():
    rng = np.random.default_rng(0)
    sigma, sigma_c, h, b, rho = 1.0, 1.3, 0.7, 2.0, 0.6
    eps = sigma * rng.standard_normal((200000, 2))
    assert np.mean(np.abs(eps[:, 0] - eps[:, 1]) <= h) == pytest.approx(sm.p_hh(sigma, h), abs=0.01)
    common = rng.standard_normal((200000, 1))
    idio = rng.standard_normal((200000, 2))
    eta = sigma_c * (math.sqrt(rho) * common + math.sqrt(1 - rho) * idio)
    assert np.mean(np.abs(eta[:, 0] - eta[:, 1]) <= h) == pytest.approx(sm.p_cc(rho, sigma_c, h), abs=0.01)
    y = b + eta[:, 0]
    assert np.mean(np.abs(eps[:, 0] - y) <= h) == pytest.approx(sm.p_hc(b, sigma, sigma_c, h), abs=0.01)


def test_mean_field_weight_is_monotone_in_gain_and_correlation():
    args = dict(n=5, m=2, b=4.0, sigma=1.0, sigma_c=1.0, h=0.4)
    w = [sm.mean_field_coalition_weight(rho=1.0, gamma=g, **args) for g in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert all(a < b for a, b in zip(w, w[1:]))
    assert w[0] == pytest.approx(2 / 7)  # zero gain = uniform weights = the mean
    w = [sm.mean_field_coalition_weight(rho=r, gamma=2.0, **args) for r in (0.0, 0.5, 0.9, 1.0)]
    assert all(a <= b + 1e-12 for a, b in zip(w, w[1:]))


def test_median_bias_bound_holds_in_simulation():
    n, m = 5, 2
    f = m / (n + m)
    bound = sm.median_bias_bound(f, 1.0)
    for b in (4.0, 8.0, 20.0):
        r = sm.simulate(n=n, m=m, rho=1.0, b=b, trials=TRIALS, seed=4)
        assert abs(r["bias_median"]) <= bound + 0.05  # bounded, independent of b
    assert sm.median_bias_bound(0.5, 1.0) == math.inf


def test_credibility_amplifies_attack_exactly_when_coalition_out_corroborates():
    """dc > 0 <=> the credibility estimate puts more than the uniform share f on the coalition."""
    for h in (0.25, 0.4, 1.0):
        n, m, rho, b = 5, 2, 1.0, 6.0
        ch, cc = sm.expected_corroboration(n, m, rho, b, 1.0, 1.0, h)
        r = sm.simulate(n=n, m=m, rho=rho, b=b, h=h, gamma=1.5, trials=TRIALS, seed=5)
        f = m / (n + m)
        if cc - ch > 0.15:
            assert r["coalition_weight_credibility"] > f
            assert r["mse_credibility"] > r["mse_mean"]
        elif cc - ch < -0.15:
            assert r["coalition_weight_credibility"] < f
            assert r["mse_credibility"] < r["mse_mean"]


def test_dependence_penalty_discounts_the_coordinated_coalition_with_a_long_window():
    r = sm.simulate(n=5, m=2, rho=1.0, b=4.0, gamma=2.0, lam=4.0, T=16, trials=TRIALS, seed=6)
    assert r["mean_penalty_coalition"] < r["mean_penalty_honest"]
    assert r["coalition_weight_credibility_dependence"] < r["coalition_weight_credibility"]
    assert r["mse_credibility_dependence"] < r["mse_credibility"]


def test_dependence_penalty_is_neutral_at_zero_strength():
    a = sm.simulate(n=5, m=2, rho=1.0, b=4.0, lam=0.0, trials=800, seed=7)
    assert a["mse_credibility_dependence"] == pytest.approx(a["mse_credibility"])
    assert a["coalition_weight_credibility_dependence"] == pytest.approx(a["coalition_weight_credibility"])


def test_expected_excess_matches_long_window_simulation():
    n, m, rho, b, h = 5, 2, 1.0, 4.0, 0.4
    eh, ec = sm.expected_excess(n, m, rho, b, 1.0, 1.0, h)
    rng = np.random.default_rng(8)
    T = 4000
    eps = rng.standard_normal((1, T, n))
    common = rng.standard_normal((1, T, 1))
    r = np.concatenate([eps, b + common * np.ones((1, T, m))], axis=-1)
    pen = sm.dependence_penalty(r, h, lam=1.0)
    excess = -np.log(pen[0])
    assert excess[n:].mean() == pytest.approx(ec, abs=0.03)
    assert excess[:n].mean() == pytest.approx(eh, abs=0.05)


def test_critical_gain_is_finite_only_when_collapse_is_possible():
    # b = 1.5 sigma: the uniform weight f = 2/7 is below W* = 0.35, so the mean still beats the
    # median and a finite gain is needed before credibility loses to it
    g = sm.critical_gain(n=5, m=2, rho=1.0, b=1.5, sigma=1.0, sigma_c=1.0, h=0.4)
    assert np.isfinite(g) and g > 0
    # no collapse when honest reporters out-corroborate the coalition (dc < 0 at h = sigma)
    assert not np.isfinite(sm.critical_gain(n=5, m=2, rho=1.0, b=1.5, sigma=1.0, sigma_c=1.0, h=1.0))
    # b = 4 sigma: f already exceeds W*, i.e. even the mean is worse than the median -> NaN
    assert not np.isfinite(sm.critical_gain(n=5, m=2, rho=1.0, b=4.0, sigma=1.0, sigma_c=1.0, h=0.4))
    # a strong dependence penalty pushes the critical gain up (or removes the collapse)
    gap = sm.defended_penalty_gap(5, 2, 1.0, 1.5, 4.0, 1.0, 1.0, 0.4)
    gd = sm.critical_gain(n=5, m=2, rho=1.0, b=1.5, sigma=1.0, sigma_c=1.0, h=0.4, penalty_gap=gap)
    assert gap > 0 and (not np.isfinite(gd) or gd > g)


def test_expected_excess_when_cross_group_agreement_is_larger():
    """A coalition reporter's closest partner can be honest, even at nonzero attack bias."""
    n, m, b, sigma_c, h = 5, 2, 4.0, 10.0, 0.4
    assert sm.p_hc(b, 1.0, sigma_c, h) > sm.p_cc(0.0, sigma_c, h)
    rng = np.random.default_rng(83)
    reports = np.concatenate([
        rng.normal(size=(1, 200000, n)),
        b + sigma_c * rng.normal(size=(1, 200000, m)),
    ], axis=-1)
    observed = -np.log(sm.dependence_penalty(reports, h, lam=1.0)[0])
    eh, ec = sm.expected_excess(n, m, 0.0, b, 1.0, sigma_c, h)
    assert ec > 0.0
    assert observed[n:].mean() == pytest.approx(ec, abs=0.001)
    assert observed[:n].mean() == pytest.approx(eh, abs=0.004)


def test_mean_field_can_reverse_realized_weight_amplification():
    """Regression against treating softmax(E counts) as E softmax(counts)."""
    args = dict(n=5, m=2, rho=1.0, b=20.0, gamma=1.0, sigma=1.0, sigma_c=1.0, h=0.43)
    surrogate = sm.mean_field_coalition_weight(**args)
    realized = sm.simulate(**args, T=1, trials=100000, seed=82)
    assert surrogate > 2 / 7
    assert realized["coalition_weight_credibility"] < 2 / 7 - 0.005


def test_zero_gain_and_singleton_have_no_majority_capture_threshold():
    assert math.isnan(sm.critical_rho(5, 2, 0.0, 1.0, 1.0, 0.4))
    assert math.isnan(sm.critical_rho(5, 1, 1.0, 1.0, 1.0, 0.4))


def test_large_gain_weight_is_numerically_stable():
    assert sm.mean_field_coalition_weight(5, 2, 0.0, 4.0, 1e6, 1.0, 1.0, 0.4) == 0.0
    assert sm.mean_field_coalition_weight(5, 2, 1.0, 4.0, 1e6, 1.0, 1.0, 0.4) == 1.0


def test_dependence_discount_can_restore_positive_bias_threshold_crossing():
    """Although f exceeds W*, a strong penalty can put the zero-gain weight below W*."""
    args = dict(n=5, m=2, rho=1.0, b=4.0, sigma=1.0, sigma_c=1.0, h=0.4)
    gap = 2.0
    threshold = sm.critical_gain(**args, penalty_gap=gap)
    wstar = sm.robustness_weight_threshold(2 / 7, 1.0, 4.0)
    assert np.isfinite(threshold) and threshold > 0.0
    assert sm.mean_field_coalition_weight(**args, gamma=0.0, penalty_gap=gap) < wstar
    assert sm.mean_field_coalition_weight(**args, gamma=threshold, penalty_gap=gap) == pytest.approx(wstar)

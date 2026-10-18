"""0.5.0 anchors for `fit_xy`, the fit with errors in the settings too.

- Straight line, Pearson's data with York's weights (as tabulated by
  C. A. Cantrell, Atmos. Chem. Phys. 8, 5477-5487 (2008), Table 1):
  the slope and intercept equal the "Williamson-York" row of Cantrell's
  Table 2 (-0.48053 and 5.4799, to the digits printed),
  and equal the iteration of D. York et al., Am. J. Phys. 72, 367
  (2004) (its equations as reproduced by Cantrell, eqs. 5 and 6),
  written out in this test, to 1e-6 of an error bar; the error bars
  equal York's standard errors evaluated at the adjusted points. The
  ordinary weighted fit of the same data (`fit`, x exact) equals
  Cantrell's Table 2 row "Std Lst-Sqrs w/wgts" (-0.61081 and 6.1001).
- The minimum chi2 of a straight line equals the closed form
  sum (y - a - b x)^2 / (sigma_y^2 + b^2 sigma_x^2), the minimum over
  each true setting worked out by hand (the derivation is below).
- A nonlinear model (exponential decay, errors in the times): the
  gradient of S, computed here with exact derivatives, vanishes at the
  solution, and the covariance equals the effective-variance formula
  built here from exact derivatives at the fitted settings.
- Seeded simulation: the scatter of the fitted slope over replicate
  data sets matches the reported error bar, and chi2 averages n - p;
  `fit` on the same data (x taken as exact) is biased.
- Exact settings (x_sigmas 0) give `fit`; a second, exact setting
  column is left untouched; the result is the same in seconds and in
  nanoseconds; the result works with goodness_of_fit, inverse_predict
  and audit_record; every refusal fires.
"""
import json
import math

import numpy as np
import pytest

from labplan import (Model, audit_record, fit, fit_xy, goodness_of_fit,
                     inverse_predict, report_text)
from labplan.fit import data_digest

# Pearson (1901) data with York (1966) weights, Cantrell (2008) Table 1
PX = np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
PY = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
WX = np.array([1000.0, 1000.0, 500.0, 800.0, 200.0, 80.0, 60.0, 20.0,
               1.8, 1.0])
WY = np.array([1.0, 1.8, 4.0, 8.0, 20.0, 20.0, 70.0, 70.0, 100.0,
               500.0])


def _line():
    return Model("line", lambda th, x: th[0] + th[1] * x[:, 0],
                 ("a", "b"), "straight line, test suite")


def _york(x, y, wx, wy, m=-0.5, iters=500):
    """York et al. (2004) without x-y error correlation (r_i = 0), as
    given by Cantrell (2008), eqs. 5 and 6: returns slope, intercept,
    sigma_slope, sigma_intercept (at the adjusted points) and
    S = sum W_i (y_i - b - m x_i)^2."""
    for _ in range(iters):
        w = wx * wy / (wx + m * m * wy)
        xb = np.sum(w * x) / np.sum(w)
        yb = np.sum(w * y) / np.sum(w)
        u, v = x - xb, y - yb
        beta = w * (u / wy + m * v / wx)
        m_new = np.sum(w * beta * v) / np.sum(w * beta * u)
        if m_new == m:
            break
        m = m_new
    w = wx * wy / (wx + m * m * wy)
    xb = np.sum(w * x) / np.sum(w)
    yb = np.sum(w * y) / np.sum(w)
    b = yb - m * xb
    beta = w * ((x - xb) / wy + m * (y - yb) / wx)
    xadj = xb + beta                      # adjusted (least-squares) x
    xab = np.sum(w * xadj) / np.sum(w)
    s_m = math.sqrt(1.0 / np.sum(w * (xadj - xab) ** 2))
    s_b = math.sqrt(1.0 / np.sum(w) + xab ** 2 * s_m ** 2)
    s = float(np.sum(w * (y - b - m * x) ** 2))
    return m, b, s_m, s_b, s


def test_pearson_york_benchmark():
    res = fit_xy(_line(), PX, PY, [5.0, -0.5], sigmas=1 / np.sqrt(WY),
                 x_sigmas=1 / np.sqrt(WX))
    a, b = res.theta
    # Cantrell (2008) Table 2, "Williamson-York" row, as printed
    assert round(b, 5) == -0.48053 and round(a, 4) == 5.4799
    m, c, s_m, s_c, s = _york(PX, PY, WX, WY)
    # tolerances: the search stops when a step moves no parameter by
    # more than 1e-10 (1 + max |theta|) and changes chi2 by at most
    # 1e-12 (1 + chi2); measured here: 4e-10 of an error bar
    assert abs(b - m) < 1e-6 * res.sigma["b"]
    assert abs(a - c) < 1e-6 * res.sigma["a"]
    assert res.sigma["b"] == pytest.approx(s_m, rel=1e-7)
    assert res.sigma["a"] == pytest.approx(s_c, rel=1e-7)
    assert res.chi2 == pytest.approx(s, rel=1e-10)
    assert res.chi2_dof == 8
    # the same numbers rounded as printed by the R package bfsl
    # (README of bfsl on CRAN: 5.47991, 0.29497, -0.48053, 0.05799,
    # chi-squared 11.87 on 8 degrees of freedom, p-value 0.1573)
    assert round(res.sigma["a"], 5) == 0.29497
    assert round(res.sigma["b"], 5) == 0.05799
    g = goodness_of_fit(res)
    assert round(g["chi2"], 2) == 11.87 and round(g["p_value"], 4) == 0.1573
    # the ordinary weighted fit (x exact): Cantrell Table 2 row
    # "Std Lst-Sqrs w/wgts"
    ols = fit(_line(), PX, PY, [5.0, -0.5], sigmas=1 / np.sqrt(WY))
    assert round(ols.theta[1], 5) == -0.61081
    assert round(ols.theta[0], 4) == 6.1001


def test_line_chi2_is_the_effective_variance_closed_form():
    """For y = a + b x, minimizing ((y - a - b xi)/sy)^2 + ((x - xi)/sx)^2
    over xi (set the derivative to zero: xi = x + b sx^2 r / (sy^2 +
    b^2 sx^2) with r = y - a - b x) leaves r^2 / (sy^2 + b^2 sx^2). So
    the minimum S is that sum at the fitted a, b, and the fitted
    settings are those xi."""
    rng = np.random.default_rng(4)
    xt = np.linspace(0.0, 10.0, 15)
    sx = rng.uniform(0.05, 0.3, 15)
    sy = rng.uniform(0.05, 0.3, 15)
    x = xt + sx * rng.standard_normal(15)
    y = 1.0 + 0.7 * xt + sy * rng.standard_normal(15)
    res = fit_xy(_line(), x, y, [0.0, 1.0], sy, sx)
    a, b = res.theta
    r = y - a - b * x
    assert res.chi2 == pytest.approx(np.sum(r ** 2 / (sy ** 2 + b ** 2
                                                      * sx ** 2)),
                                     rel=1e-10)
    xi = x + b * sx ** 2 * r / (sy ** 2 + b ** 2 * sx ** 2)
    assert np.allclose(res.x_adjusted[:, 0], xi, rtol=0,
                       atol=1e-6 * sx.min())
    # York's iteration gives the same line
    m, c, s_m, s_c, _ = _york(x, y, 1 / sx ** 2, 1 / sy ** 2, m=1.0)
    assert abs(b - m) < 1e-6 * s_m and abs(a - c) < 1e-6 * s_c
    assert res.sigma["b"] == pytest.approx(s_m, rel=1e-6)


def _decay():
    return Model("decay", lambda th, x: th[0] * np.exp(-x[:, 0] / th[1])
                 + th[2], ("A", "tau", "B"), "exponential decay, tests")


def _decay_data(seed=5):
    rng = np.random.default_rng(seed)
    xt = np.linspace(0.0, 5.0, 25)
    x = xt + 0.05 * rng.standard_normal(25)
    y = _decay().predict([1.0, 1.5, 0.2], xt) \
        + 0.01 * rng.standard_normal(25)
    return x, y


def test_nonlinear_solution_is_stationary_and_cov_is_effective_variance():
    x, y = _decay_data()
    sx, sy = 0.05, 0.01
    res = fit_xy(_decay(), x, y, [0.8, 1.2, 0.1], sy, sx)
    A, tau, B = res.theta
    xi = res.x_adjusted[:, 0]
    e = np.exp(-xi / tau)
    r = (A * e + B - y) / sy                       # reading residuals
    d_theta = np.column_stack([e, A * xi * e / tau ** 2,
                               np.ones_like(xi)])  # df/dtheta, exact
    d_x = -A * e / tau                             # df/dx, exact
    # gradient of S/2 with respect to theta and to each true setting,
    # in units of one error bar (dimensionless)
    g_theta = d_theta.T @ (r / sy)
    g_x = r * d_x / sy + (xi - x) / sx ** 2
    sig = np.array([res.sigma[k] for k in ("A", "tau", "B")])
    assert np.max(np.abs(g_theta) * sig) < 1e-5
    assert np.max(np.abs(g_x) * sx) < 1e-5
    w = 1.0 / (sy ** 2 + d_x ** 2 * sx ** 2)
    cov = np.linalg.inv((d_theta * w[:, None]).T @ d_theta)
    scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    assert np.all(np.abs(res.cov - cov) < 1e-6 * scale)
    # the chi-squared has n - p degrees of freedom
    assert res.chi2_dof == 22


def test_simulation_unbiased_and_error_bars_honest():
    """300 seeded replicate straight-line data sets with errors in both
    coordinates (true slope 2, x error 0.3, y error 0.2): the fitted
    slopes scatter as the reported error bar says (within 15 %), their
    mean is the truth within 4.5 standard errors, and chi2 averages
    n - p = 10 within 4.5 standard errors (its variance is 2 (n - p)).
    `fit`, treating the settings as exact, is biased low by more
    than 10 standard errors of the mean (regression dilution)."""
    rng = np.random.default_rng(11)
    xt = np.linspace(0.0, 4.0, 12)
    slopes, sigmas, chis, ols = [], [], [], []
    for _ in range(300):
        x = xt + 0.3 * rng.standard_normal(12)
        y = 0.5 + 2.0 * xt + 0.2 * rng.standard_normal(12)
        res = fit_xy(_line(), x, y, [0.0, 1.0], 0.2, 0.3)
        slopes.append(res.theta[1])
        sigmas.append(res.sigma["b"])
        chis.append(res.chi2)
        ols.append(fit(_line(), x, y, [0.0, 1.0], sigmas=0.2).theta[1])
    slopes, chis, ols = map(np.array, (slopes, chis, ols))
    se = np.std(slopes) / math.sqrt(300)
    assert abs(np.std(slopes) / np.mean(sigmas) - 1.0) < 0.15
    assert abs(np.mean(slopes) - 2.0) < 4.5 * se
    assert abs(np.mean(chis) - 10.0) < 4.5 * math.sqrt(20.0 / 300)
    assert 2.0 - np.mean(ols) > 10 * np.std(ols) / math.sqrt(300)


def test_exact_settings_give_fit_and_exact_columns_stay():
    x, y = _decay_data()
    ref = fit(_decay(), x, y, [0.8, 1.2, 0.1], sigmas=0.01)
    res = fit_xy(_decay(), x, y, [0.8, 1.2, 0.1], 0.01, 0.0)
    sig = np.sqrt(np.diag(ref.cov))
    assert np.all(np.abs(res.theta - ref.theta) < 1e-6 * sig)
    assert np.allclose(np.sqrt(np.diag(res.cov)), sig, rtol=1e-6)
    assert res.chi2 == pytest.approx(ref.chi2, rel=1e-9)
    assert np.array_equal(res.x_adjusted, x[:, None])
    # a plane y = a x1 + b x2 + c with x1 uncertain and x2 exact: the
    # exact column is not moved, and the minimum S is the effective
    # variance sum with only x1's error (a linear model: closed form)
    plane = Model("plane", lambda th, x: th[0] * x[:, 0] + th[1] * x[:, 1]
                  + th[2], ("a", "b", "c"), "plane model, test suite")
    rng = np.random.default_rng(8)
    xt = rng.uniform(-1.0, 1.0, (20, 2))
    xo = xt.copy()
    xo[:, 0] += 0.05 * rng.standard_normal(20)
    yo = plane.predict([1.5, -0.7, 0.3], xt) + 0.02 * rng.standard_normal(20)
    xs = np.column_stack([np.full(20, 0.05), np.zeros(20)])
    res = fit_xy(plane, xo, yo, [1.0, 0.0, 0.0], 0.02, xs)
    assert np.array_equal(res.x_adjusted[:, 1], xo[:, 1])
    a, b, c = res.theta
    r = yo - plane.predict(res.theta, xo)
    assert res.chi2 == pytest.approx(
        np.sum(r ** 2 / (0.02 ** 2 + a ** 2 * 0.05 ** 2)), rel=1e-10)


def test_units_do_not_matter():
    x, y = _decay_data()
    s = fit_xy(_decay(), x, y, [0.8, 1.2, 0.1], 0.01, 0.05)
    ns = fit_xy(_decay(), x * 1e-9, y, [0.8, 1.2e-9, 0.1], 0.01, 0.05e-9)
    assert abs(ns.theta[1] * 1e9 - s.theta[1]) < 1e-6 * s.sigma["tau"]
    assert ns.sigma["tau"] * 1e9 == pytest.approx(s.sigma["tau"], rel=1e-6)
    assert ns.chi2 == pytest.approx(s.chi2, rel=1e-9)


def test_result_works_downstream_and_digest():
    res = fit_xy(_line(), PX, PY, [5.0, -0.5], 1 / np.sqrt(WY),
                 1 / np.sqrt(WX))
    rec = audit_record(res, operator="test")
    assert json.loads(json.dumps(rec)) == rec
    assert rec["fit_method"].startswith("fit_xy")
    assert "fit_xy" in report_text(rec)
    assert "fit_method" not in audit_record(
        fit(_line(), PX, PY, [5.0, -0.5], sigmas=1 / np.sqrt(WY)))
    assert res.data_digest == data_digest(
        PX[:, None], PY, 1 / np.sqrt(WY), (1 / np.sqrt(WX))[:, None])
    assert res.data_digest != data_digest(PX[:, None], PY, 1 / np.sqrt(WY))
    out = inverse_predict(_line(), res, 4.0, (0.0, 7.4))
    assert out["value"] == pytest.approx((4.0 - res.theta[0])
                                         / res.theta[1], rel=1e-12)


def test_refusals():
    m = _line()
    with pytest.raises(ValueError, match="sigmas"):
        fit_xy(m, PX, PY, [5.0, -0.5], None, 0.1)
    for bad in (-0.1, np.nan, np.ones(3)):
        with pytest.raises(ValueError, match="x_sigmas"):
            fit_xy(m, PX, PY, [5.0, -0.5], 0.1, bad)
    coupled = Model("centered line",
                    lambda th, x: th[0] + th[1] * (x[:, 0]
                                                   - x[:, 0].mean()),
                    ("a", "b"), "line about the mean setting, tests")
    with pytest.raises(ValueError, match="other rows"):
        fit_xy(coupled, PX, PY, [3.0, -0.5], 0.1, 0.1)
    with pytest.raises(ValueError, match="cannot tell"):
        fit_xy(Model("sum", lambda th, x: (th[0] + th[1]) * x[:, 0],
                     ("a", "b"), "degenerate test model"),
               PX, PY, [1.0, 1.0], 0.1, 0.1)
    with pytest.raises(ValueError, match="max_iter"):
        fit_xy(m, PX, PY, [5.0, -0.5], 0.1, 0.1, max_iter=0)
    with pytest.raises(RuntimeError, match="converge"):
        fit_xy(m, PX, PY, [50.0, 5.0], 0.1, 0.1, max_iter=1)
    with pytest.raises(ValueError, match="2 parameters"):
        fit_xy(m, [1.0], [2.0], [5.0, -0.5], 0.1, 0.1)

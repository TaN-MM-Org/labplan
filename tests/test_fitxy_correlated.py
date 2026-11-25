"""0.7.0 anchors for `fit_xy(..., xy_correlation=...)`: errors of a
setting and of its reading that are correlated.

- Straight line: equal to York's iteration with correlation r_i,
  written out in this test from the algorithm of D. York et al.,
  Am. J. Phys. 72, 367 (2004), eqs. (13), as listed by C. M. Wehr and
  S. R. Saleska, Biogeosciences 14, 17-29 (2017), Appendix A:
  W_i = alpha_i^2 / (b^2 omega_Y,i + omega_X,i - 2 b r_i alpha_i),
  beta_i = W_i (U_i / omega_Y,i + b V_i / omega_X,i
  - (b U_i + V_i) r_i / alpha_i), b = sum W beta V / sum W beta U, with
  alpha_i = sqrt(omega_X,i omega_Y,i), omega = 1 / sigma^2. The error
  bars are York's at the adjusted points x = X_bar + beta.
- The minimum chi2 equals the closed form worked out by hand: for a
  model that is a straight line in its settings, minimizing the
  correlated quadratic form over the true settings leaves
  r^2 / var(e_y - sum_k b_k e_xk), with
  var = sigma_y^2 + sum_k b_k^2 sigma_xk^2
  - 2 sum_k b_k r_k sigma_xk sigma_y (one and two setting columns).
- A nonlinear model: the gradient of the correlated sum S, computed
  here with exact derivatives and the inverse 2 x 2 covariance of each
  measurement, is zero at the solution, and the covariance equals the
  effective-variance formula with the same correlation term.
- Seeded simulation with correlated errors: the scatter of the slope
  matches its error bar and chi2 averages n - p; leaving the
  correlation out makes chi2 far too small.
- Correlation 0 gives exactly the 0.6.0 fit (digest included); every
  refusal fires.
"""
import json
import math

import numpy as np
import pytest

from labplan import Model, audit_record, fit_xy, report_text
from labplan.fit import data_digest

PX = np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
PY = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
WX = np.array([1000.0, 1000.0, 500.0, 800.0, 200.0, 80.0, 60.0, 20.0,
               1.8, 1.0])
WY = np.array([1.0, 1.8, 4.0, 8.0, 20.0, 20.0, 70.0, 70.0, 100.0,
               500.0])


def _line():
    return Model("line", lambda th, x: th[0] + th[1] * x[:, 0],
                 ("a", "b"), "straight line, test suite")


def _york(x, y, wx, wy, r, m=-0.5, iters=1000):
    """York's iteration with correlation r_i (see the module
    docstring). Returns slope, intercept, their standard errors at the
    adjusted points, S and the adjusted x."""
    al = np.sqrt(wx * wy)

    def parts(m):
        w = al ** 2 / (m * m * wy + wx - 2 * m * r * al)
        xb = np.sum(w * x) / np.sum(w)
        yb = np.sum(w * y) / np.sum(w)
        u, v = x - xb, y - yb
        beta = w * (u / wy + m * v / wx - (m * u + v) * r / al)
        return w, xb, yb, u, v, beta

    for _ in range(iters):
        w, xb, yb, u, v, beta = parts(m)
        m_new = np.sum(w * beta * v) / np.sum(w * beta * u)
        if m_new == m:
            break
        m = m_new
    w, xb, yb, u, v, beta = parts(m)
    b = yb - m * xb
    xadj = xb + beta
    xab = np.sum(w * xadj) / np.sum(w)
    s_m = math.sqrt(1.0 / np.sum(w * (xadj - xab) ** 2))
    s_b = math.sqrt(1.0 / np.sum(w) + xab ** 2 * s_m ** 2)
    s = float(np.sum(w * (y - b - m * x) ** 2))
    return m, b, s_m, s_b, s, xadj


@pytest.mark.parametrize("r", [0.5, -0.7, "spread"])
def test_york_with_correlation(r):
    rr = np.linspace(-0.9, 0.9, PX.size) if r == "spread" \
        else np.full(PX.size, r)
    sx, sy = 1 / np.sqrt(WX), 1 / np.sqrt(WY)
    res = fit_xy(_line(), PX, PY, [5.0, -0.5], sy, sx, xy_correlation=rr)
    m, c, s_m, s_c, s, xadj = _york(PX, PY, WX, WY, rr)
    a, b = res.theta
    assert abs(b - m) < 1e-6 * res.sigma["b"]          # measured 1.8e-9
    assert abs(a - c) < 1e-6 * res.sigma["a"]          # measured 1.8e-9
    assert res.sigma["b"] == pytest.approx(s_m, rel=1e-7)  # meas. 6.1e-10
    assert res.sigma["a"] == pytest.approx(s_c, rel=1e-7)  # meas. 3.8e-10
    assert res.chi2 == pytest.approx(s, rel=1e-10)     # measured 4.7e-15
    assert np.all(np.abs(res.x_adjusted[:, 0] - xadj)
                  < 1e-6 * sx)                         # measured 1.5e-8
    # and the closed form, with the fitted line
    var = sy ** 2 + b ** 2 * sx ** 2 - 2 * b * rr * sx * sy
    assert res.chi2 == pytest.approx(np.sum((PY - a - b * PX) ** 2 / var),
                                     rel=1e-10)        # measured 4.3e-15
    # the correlation moves the line (measured: by 0.20, 1.13 and 0.099
    # error bars)
    plain = fit_xy(_line(), PX, PY, [5.0, -0.5], sy, sx)
    assert abs(plain.theta[1] - b) > 0.05 * res.sigma["b"]


def test_two_setting_columns_closed_form():
    """y = a + b x1 + c x2 with errors in both settings, each
    correlated with the reading error. The minimum over the true
    settings of the quadratic form is r^2 / var with var as in the
    module docstring (with resid = y - a - b x1 - c x2 in place of r),
    and the true settings are the point of the plane closest to the
    measured one in that form: xi_k = x_k - cov(e_xk, e) resid / var
    with e = e_y - b e_x1 - c e_x2, that is
    xi_k = x_k + (theta_k sigma_xk^2 - r_k sigma_xk sigma_y) resid / var
    (theta_1 = b, theta_2 = c)."""
    rng = np.random.default_rng(11)
    n = 20
    x1t = np.linspace(0.0, 10.0, n)
    x2t = rng.uniform(-3.0, 3.0, n)
    s1 = rng.uniform(0.05, 0.2, n)
    s2 = rng.uniform(0.05, 0.2, n)
    sy = rng.uniform(0.05, 0.2, n)
    r1 = rng.uniform(-0.6, 0.6, n)
    r2 = rng.uniform(-0.6, 0.6, n)
    z1, z2, z3 = rng.standard_normal((3, n))
    # e_y = sy (r1 z1 + r2 z2 + sqrt(1 - r1^2 - r2^2) z3) has
    # correlation r1 with e_x1 = s1 z1 and r2 with e_x2 = s2 z2
    ex1, ex2 = s1 * z1, s2 * z2
    ey = sy * (r1 * z1 + r2 * z2 + np.sqrt(1 - r1 ** 2 - r2 ** 2) * z3)
    x = np.column_stack([x1t + ex1, x2t + ex2])
    y = 1.0 + 0.7 * x1t - 0.4 * x2t + ey
    plane = Model("plane", lambda th, x: th[0] + th[1] * x[:, 0]
                  + th[2] * x[:, 1], ("a", "b", "c"), "plane, test suite")
    res = fit_xy(plane, x, y, [0.0, 1.0, 0.0], sy,
                 np.column_stack([s1, s2]),
                 xy_correlation=np.column_stack([r1, r2]))
    a, b, c = res.theta
    var = sy ** 2 + b ** 2 * s1 ** 2 + c ** 2 * s2 ** 2 \
        - 2 * b * r1 * s1 * sy - 2 * c * r2 * s2 * sy
    resid = y - a - b * x[:, 0] - c * x[:, 1]
    assert res.chi2 == pytest.approx(np.sum(resid ** 2 / var),
                                     rel=1e-10)        # measured 4.2e-15
    xi1 = x[:, 0] + (b * s1 ** 2 - r1 * s1 * sy) * resid / var
    xi2 = x[:, 1] + (c * s2 ** 2 - r2 * s2 * sy) * resid / var
    assert np.all(np.abs(res.x_adjusted[:, 0] - xi1) < 1e-6 * s1)
    assert np.all(np.abs(res.x_adjusted[:, 1] - xi2) < 1e-6 * s2)
    # measured 1.7e-10 and 3.3e-10


def test_nonlinear_model_is_stationary_with_exact_derivatives():
    rng = np.random.default_rng(5)
    t_true = np.linspace(0.0, 4.0, 25)
    st = 0.03
    sy = 0.01
    rho = np.where(np.arange(25) % 2 == 0, 0.6, -0.3)
    z1, z2 = rng.standard_normal((2, 25))
    t = t_true + st * z1
    y = 2.0 * np.exp(-t_true / 1.3) + 0.1 \
        + sy * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)
    decay = Model("decay", lambda th, x: th[0] * np.exp(-x[:, 0] / th[1])
                  + th[2], ("A", "tau", "B"), "exponential decay, tests")
    res = fit_xy(decay, t, y, [1.5, 1.0, 0.0], sy, st, xy_correlation=rho)
    A, tau, B = res.theta
    xi = res.x_adjusted[:, 0]
    e = np.exp(-xi / tau)
    ex = t - xi                        # setting residual
    ey = y - (A * e + B)               # reading residual
    # S_i = (ex, ey) V^-1 (ex, ey)^T with V = [[st^2, c], [c, sy^2]]
    c = rho * st * sy
    det = st ** 2 * sy ** 2 - c ** 2
    vx = (sy ** 2 * ex - c * ey) / det       # (V^-1 e)_x
    vy = (st ** 2 * ey - c * ex) / det       # (V^-1 e)_y
    d_theta = np.column_stack([e, A * e * xi / tau ** 2, np.ones(25)])
    d_x = -A * e / tau
    # half gradients: d(ex)/dxi = -1, d(ey)/dxi = -d_x,
    # d(ey)/dtheta = -d_theta
    g_theta = -(d_theta.T @ vy)
    g_x = -vx - vy * d_x
    sig = np.sqrt(np.diag(res.cov))
    assert np.max(np.abs(g_theta) * sig) < 1e-5     # measured 6.4e-10
    assert np.max(np.abs(g_x) * st) < 1e-5          # measured 5.6e-10
    w = 1.0 / (sy ** 2 + d_x ** 2 * st ** 2 - 2 * d_x * rho * st * sy)
    cov = np.linalg.inv((d_theta * w[:, None]).T @ d_theta)
    scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    assert np.all(np.abs(res.cov - cov) < 1e-6 * scale)   # meas. 1.7e-9


def test_simulation_with_correlated_errors():
    """200 seeded data sets of a line, 15 points, sx = sy = 0.2,
    correlation 0.6. The scatter of the fitted slope matches its mean
    error bar within 15 %, the mean slope is the truth within 4.5
    standard errors, and chi2 averages 13 within 4.5 standard errors
    (sqrt(2 * 13 / 200)). Fitted without the correlation, chi2 averages
    about 13 * (1 + 0.49 - 0.84) / 1.49 = 5.7 instead."""
    rng = np.random.default_rng(21)
    xt = np.linspace(0.0, 10.0, 15)
    s, r = 0.2, 0.6
    slopes, errs, chi2, chi2_plain = [], [], [], []
    for _ in range(200):
        z1, z2 = rng.standard_normal((2, 15))
        x = xt + s * z1
        y = 1.0 + 0.7 * xt + s * (r * z1 + math.sqrt(1 - r * r) * z2)
        res = fit_xy(_line(), x, y, [1.0, 0.7], s, s, xy_correlation=r)
        slopes.append(res.theta[1])
        errs.append(res.sigma["b"])
        chi2.append(res.chi2)
        chi2_plain.append(fit_xy(_line(), x, y, [1.0, 0.7], s, s).chi2)
    slopes = np.array(slopes)
    ratio = np.std(slopes, ddof=1) / np.mean(errs)
    assert abs(ratio - 1.0) < 0.15                  # measured 1.046
    # measured: 1.9 standard errors
    assert abs(np.mean(slopes) - 0.7) < 4.5 * np.std(slopes) / math.sqrt(200)
    assert abs(np.mean(chi2) - 13.0) < 4.5 * math.sqrt(2 * 13 / 200)
    # measured mean chi2 12.83
    assert np.mean(chi2_plain) < 8.0                # measured 5.59


def test_zero_correlation_is_the_plain_fit():
    sx, sy = 1 / np.sqrt(WX), 1 / np.sqrt(WY)
    plain = fit_xy(_line(), PX, PY, [5.0, -0.5], sy, sx)
    for zero in (0.0, np.zeros(PX.size), np.zeros((PX.size, 1))):
        res = fit_xy(_line(), PX, PY, [5.0, -0.5], sy, sx,
                     xy_correlation=zero)
        assert np.array_equal(res.theta, plain.theta)
        assert np.array_equal(res.cov, plain.cov)
        assert res.chi2 == plain.chi2
        assert res.data_digest == plain.data_digest \
            == data_digest(PX[:, None], PY, sy * np.ones(10), sx[:, None])
        assert res.xy_correlation is None
    rec = audit_record(plain)
    assert rec["fit_method"] == ("fit_xy: errors in the settings and in "
                                 "the readings")


def test_correlation_reaches_the_record():
    sx, sy = 1 / np.sqrt(WX), 1 / np.sqrt(WY)
    res = fit_xy(_line(), PX, PY, [5.0, -0.5], sy, sx, xy_correlation=0.5)
    plain = fit_xy(_line(), PX, PY, [5.0, -0.5], sy, sx)
    assert res.data_digest != plain.data_digest
    assert res.data_digest == data_digest(
        PX[:, None], PY, sy, sx[:, None],
        xy_correlation=np.full((10, 1), 0.5))
    assert np.array_equal(res.xy_correlation, np.full((10, 1), 0.5))
    rec = audit_record(res)
    assert rec["fit_method"].endswith(", correlated with each other")
    assert json.loads(json.dumps(rec)) == rec
    assert "correlated with each other" in report_text(rec)
    # the result names the user's model, as for any fit
    assert res.model_name == "line"


def test_correlation_refusals():
    sx, sy = 1 / np.sqrt(WX), 1 / np.sqrt(WY)
    args = (_line(), PX, PY, [5.0, -0.5], sy, sx)
    for bad in (1.0, -1.0, 1.5):
        with pytest.raises(ValueError, match="add up to"):
            fit_xy(*args, xy_correlation=bad)
    with pytest.raises(ValueError, match="finite"):
        fit_xy(*args, xy_correlation=np.nan)
    with pytest.raises(ValueError, match="shape"):
        fit_xy(*args, xy_correlation=[0.1, 0.2])
    exact_first = sx.copy()
    exact_first[0] = 0.0
    with pytest.raises(ValueError, match="exact"):
        fit_xy(_line(), PX, PY, [5.0, -0.5], sy, exact_first,
               xy_correlation=0.3)
    # with the correlation 0 where the setting is exact it is accepted
    rr = np.full(10, 0.3)
    rr[0] = 0.0
    fit_xy(_line(), PX, PY, [5.0, -0.5], sy, exact_first, xy_correlation=rr)
    # two columns whose squares add up to 1 or more
    x2 = np.column_stack([PX, PX ** 2])
    quad = Model("quad", lambda th, x: th[0] + th[1] * x[:, 0]
                 + th[2] * x[:, 1], ("a", "b", "c"), "two-column model")
    with pytest.raises(ValueError, match="add up to 1"):
        fit_xy(quad, x2, PY, [5.0, -0.5, 0.0], sy, 0.1,
               xy_correlation=np.tile([0.8, 0.6], (10, 1)))

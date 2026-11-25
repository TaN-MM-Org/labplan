"""0.7.0 anchors for three wrong answers. Each test fails on 0.6.0,
except `test_zero_first_derivative_is_not_enlarged`, which checks that
a case 0.6.0 got right is unchanged.

- The conformal rank of an alpha that is not a decimal of up to 8
  significant digits used a tolerance of 1e-9 (relative). It read
  alpha = 0.099999999999 as 0.1 and gave, for n = 9, k = 9 and a
  coverage of 0.9, below 1 - alpha = 0.900000000001; for alpha = 1/3
  it gave a rank one too small from n of about 5e8. Every alpha is
  now read as an exact number (a short decimal, a fraction such as
  1/3, or the decimal as typed) and the rank is exact.
- `Model.numeric_jacobian` used the fallback step 1e-9 for a parameter
  at zero even when that step changed the predictions by no more than
  rounding: for a detuning of 0 Hz in a fringe with times in seconds
  the slopes were 1.2 % wrong over 10 us, and pure rounding over 0.1
  us (the plan was refused). The step is now enlarged in that case.
- `fit_xy` used the fallback step 1e-6 of the largest setting of a
  column, unchecked, for settings at or near zero. For a saturating
  response x / (x + 0.01) with settings up to 1000 that step is a
  tenth of the bend, and the fit stopped where the gradient of the
  minimized sum is 0.003 of an error bar. The step is now checked
  against smaller ones, as for parameters.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from labplan import (Model, conformal_quantile, coverage_exact, fit,
                     fit_xy, information)
from labplan.conformal import _rank


def _rank_060(n, a):
    """The 0.6.0 rule for an alpha that is not a short decimal."""
    t = (n + 1) * (1.0 - a)
    return int(np.ceil(t - 1e-9 * max(1.0, t)))


# ------------------------------------------------------------------
# conformal rank


def test_long_decimal_alpha_is_read_as_typed():
    a = 0.099999999999
    fa = Fraction(repr(a))
    # 0.6.0: k = 9 of 9 scores, coverage 0.9 < 1 - alpha
    assert _rank_060(9, a) == 9 and Fraction(9, 10) < 1 - fa
    with pytest.raises(ValueError, match="rank 10 exceeds n"):
        conformal_quantile(np.arange(1.0, 10.0), a)
    # the smallest n named in the refusal is the exact one:
    # n >= (1 - alpha) / alpha = 9.0000000001..., so 10
    assert math.ceil((1 - fa) / fa) == 10
    with pytest.raises(ValueError, match="at least 10 calibration"):
        conformal_quantile(np.arange(1.0, 10.0), a)
    assert coverage_exact(10, a) == 10 / 11
    assert coverage_exact(19, a) == 19 / 20 >= 1 - a
    assert conformal_quantile(np.arange(1.0, 20.0), a) == 19.0


def test_random_long_decimals_get_the_exact_rank():
    """Decimals of 9 to 15 significant digits at n up to 1e7: the rank
    is ceil((n + 1)(1 - alpha)) in exact rational arithmetic, and the
    coverage is never below 1 - alpha. The 0.6.0 rule gives a smaller
    rank in some of these cases (counted, so the test shows the
    difference)."""
    rng = np.random.default_rng(7)
    below_060 = 0
    for _ in range(400):
        digits = int(rng.integers(9, 16))
        a = float(f"{rng.uniform(1e-4, 0.9999):.{digits}g}")
        fa = Fraction(repr(a))
        for n in rng.integers(1, 10 ** 7, 10).tolist():
            k = math.ceil((n + 1) * (1 - fa))
            assert _rank(n, a) == k
            if k <= n:
                assert Fraction(k, n + 1) >= 1 - fa
            old = _rank_060(n, a)
            below_060 += old <= n and Fraction(old, n + 1) < 1 - fa
    assert below_060 > 0            # measured: 7 of 4000


def test_fractions_are_read_exactly_at_large_n():
    """1/3, 2/7 and 1/3000 are read as those fractions. For 1/3 the
    0.6.0 rule took (n + 1) * 2/3 = an integer + 1/3 to be that integer
    once 1e-9 (n + 1) * 2/3 >= 1/3, so from n of about 5e8."""
    n = 600000001                    # n + 1 = 3 m + 2
    assert (n + 1) % 3 == 2
    k = math.ceil(Fraction(2 * (n + 1), 3))
    assert _rank(n, 1 / 3) == k == _rank_060(n, 1 / 3) + 1
    assert coverage_exact(n, 1 / 3) == k / (n + 1)
    assert Fraction(k, n + 1) >= Fraction(2, 3)
    for a, frac in ((2 / 7, Fraction(2, 7)), (1 / 3000, Fraction(1, 3000))):
        for n in (10 ** 6, 10 ** 9 + 7, 123456789012):
            assert _rank(n, a) == math.ceil((n + 1) * (1 - frac))


# ------------------------------------------------------------------
# slopes for a parameter at zero whose natural size is large


def _fringe(th, x):
    return th[0] * np.cos(2 * np.pi * th[1] * x[:, 0] + 0.3) + 0.5


def _fringe_jac(th, x):
    t = x[:, 0]
    phase = 2 * np.pi * th[1] * t + 0.3
    return np.column_stack([np.cos(phase),
                            -th[0] * np.sin(phase) * 2 * np.pi * t])


@pytest.mark.parametrize("span", [1e-3, 1e-5, 1e-7, 1e-9])
def test_zero_detuning_in_hertz_with_times_in_seconds(span):
    """A fringe A cos(2 pi df t + 0.3) + 0.5 at df = 0 Hz, sampled from
    0 to `span` seconds. 0.6.0: slopes 1.1e-4 (1 ms) and 1.2e-2 (10 us)
    wrong, and pure rounding (plan not identifiable) at 0.1 us and
    1 ns."""
    m = Model("fringe", _fringe, ("A", "df"), "fringe with a detuning, "
              "test suite")
    exact_m = Model("fringe", _fringe, ("A", "df"), "fringe with a "
                    "detuning, test suite", jac=_fringe_jac)
    t = np.linspace(0.0, span, 41)
    th = [0.4, 0.0]
    exact = _fringe_jac(np.array(th), t[:, None])
    jac = m.jacobian(th, t)
    col_err = np.max(np.abs(jac - exact), axis=0) \
        / np.max(np.abs(exact), axis=0)
    assert np.all(col_err < 1e-6)          # measured 9.7e-9
    plan = information(m, th, t, sigmas=0.01)
    want = information(exact_m, th, t, sigmas=0.01)["cov"]
    scale = np.sqrt(np.outer(np.diag(want), np.diag(want)))
    assert plan["identifiable"]
    assert np.all(np.abs(plan["cov"] - want) < 1e-6 * scale)  # m. 4.3e-9
    # a noiseless fit from a start 0.1 / span away ends at df = 0 and
    # reports the exact covariance
    y = _fringe(np.array(th), t[:, None])
    res = fit(m, t, y, [0.35, 0.1 / span], sigmas=0.01)
    assert abs(res.values["df"]) * span < 1e-6        # measured 9.4e-18
    assert abs(res.values["A"] - 0.4) < 1e-9           # measured 0
    assert np.all(np.abs(res.cov - want) < 1e-6 * scale)  # meas. 6.4e-9


def test_zero_first_derivative_is_not_enlarged():
    """A parameter whose first derivative really is zero (it enters
    as th^3) keeps the slopes of 0.6.0, here exactly zero: a larger
    step would give h^2 x instead, and the check against a 10 times
    larger step catches that. The model stays refused."""
    m = Model("cube", lambda th, x: th[0] ** 3 * x[:, 0] + th[1],
              ("a", "b"), "cubic parameter, test suite")
    x = np.linspace(0.5, 5.0, 12)
    jac = m.jacobian([0.0, 1.0], x)
    assert np.array_equal(jac[:, 0], np.zeros(12))
    assert not information(m, [0.0, 1.0], x)["identifiable"]


# ------------------------------------------------------------------
# fit_xy: slopes with respect to settings near zero

K = 0.01


def _saturating():
    return Model("saturating",
                 lambda th, x: th[0] + th[1] * x[:, 0] / (x[:, 0] + K),
                 ("a", "b"), "saturating response, test suite")


def test_fit_xy_slopes_near_zero_settings():
    """Settings 0 (three times) and 2e-3 to 1e3, each with an error
    0.02 x + 2e-4, for a + b x / (x + 0.01). The gradient of S/2 at the
    solution, with exact derivatives (df/dxi = b K / (xi + K)^2), is
    zero, and the covariance is the effective-variance formula built
    from exact derivatives. 0.6.0 stopped at a gradient of 0.003 of an
    error bar, with a covariance 0.3 % off."""
    x = np.concatenate([[0.0, 0.0, 0.0], np.geomspace(0.002, 1000, 12)])
    rng = np.random.default_rng(3)
    sx = 0.02 * x + 2e-4
    xt = np.abs(x + sx * rng.standard_normal(x.size))
    sy = 0.01
    y = 1.0 + 2.0 * xt / (xt + K) + sy * rng.standard_normal(x.size)
    res = fit_xy(_saturating(), x, y, [1.0, 2.0], sy, sx)
    a, b = res.theta
    xi = res.x_adjusted[:, 0]
    r = (a + b * xi / (xi + K) - y) / sy
    d_theta = np.column_stack([np.ones_like(xi), xi / (xi + K)])
    d_x = b * K / (xi + K) ** 2
    g_theta = d_theta.T @ (r / sy)
    g_x = r * d_x / sy + (xi - x) / sx ** 2
    sig = np.sqrt(np.diag(res.cov))
    assert np.max(np.abs(g_theta) * sig) < 1e-5     # measured 2e-10
    assert np.max(np.abs(g_x) * sx) < 1e-5          # measured 3.4e-9
    w = 1.0 / (sy ** 2 + d_x ** 2 * sx ** 2)
    cov = np.linalg.inv((d_theta * w[:, None]).T @ d_theta)
    scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    assert np.all(np.abs(res.cov - cov) < 1e-6 * scale)   # meas. 2.7e-9

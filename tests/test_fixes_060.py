"""0.6.0 anchors for one wrong answer and two smaller fixes. Each test
fails on 0.5.0.

- `fit_xy` took the slope df/dx of every setting with a step of 1e-6
  of the largest setting in its column. For settings spread over many
  decades that step is as large as the smallest settings: for a model
  a + b log(x) with x from 2e-3 to 1e3 the slopes at the small settings
  were wrong, and the fit stopped at a point where the gradient of half
  the minimized sum (as in test_fitxy.py), worked out here with exact
  derivatives, is 0.009 of an error bar instead of 0. The parameters
  were off by up to 0.005 of their error bar, the fitted settings by up
  to 0.006 of theirs, and the error bars were 0.6 % and 1.3 % too large.
  With x from 1e-3 the step reached zero and the fit was refused as
  "non-finite". The step is now 1e-6 of each setting where that
  resolves the change.
- The refusal of `conformal_quantile` printed the level to 3 digits, so
  alpha = 0.0001 said "cannot certify level 1".
- `load_columns(skip_rows=2.7)` skipped 2 lines without a word; a
  number of lines that is not whole is now refused, as `design` refuses
  n_pick = 2.5 since 0.3.0.
"""
import numpy as np
import pytest

from labplan import Model, conformal_quantile, fit_xy, load_columns


def _log_model():
    return Model("log response",
                 lambda th, x: th[0] + th[1] * np.log(x[:, 0]),
                 ("a", "b"), "logarithmic response, test suite")


def _log_data(low, seed=0):
    """13 settings from `low` to 1e3, each with a 5 % error, and
    readings a + b log(true setting) with error 0.02."""
    x = np.geomspace(low, 1e3, 13)
    rng = np.random.default_rng(seed)
    xt = x * (1 + 0.05 * rng.standard_normal(x.size))
    y = 1.0 + 0.5 * np.log(xt) + 0.02 * rng.standard_normal(x.size)
    return x, y, 0.05 * x, 0.02


def _check_stationary(res, x, y, sx, sy):
    """The gradient of S/2 with exact derivatives (df/da = 1,
    df/db = log(xi), df/dxi = b / xi) is zero at the solution, in units
    of one error bar, and the covariance is the effective-variance
    formula built from the same exact derivatives."""
    a, b = res.theta
    xi = res.x_adjusted[:, 0]
    r = (a + b * np.log(xi) - y) / sy
    d_theta = np.column_stack([np.ones_like(xi), np.log(xi)])
    d_x = b / xi
    g_theta = d_theta.T @ (r / sy)
    g_x = r * d_x / sy + (xi - x) / sx ** 2
    sig = np.array([res.sigma["a"], res.sigma["b"]])
    assert np.max(np.abs(g_theta) * sig) < 1e-5
    assert np.max(np.abs(g_x) * sx) < 1e-5
    w = 1.0 / (sy ** 2 + d_x ** 2 * sx ** 2)
    cov = np.linalg.inv((d_theta * w[:, None]).T @ d_theta)
    scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    assert np.all(np.abs(res.cov - cov) < 1e-6 * scale)


def test_fit_xy_settings_over_many_decades():
    # 0.5.0: the gradient with respect to the settings was 0.009 of an
    # error bar here
    x, y, sx, sy = _log_data(2e-3)
    res = fit_xy(_log_model(), x, y, [1.0, 0.5], sy, sx)
    _check_stationary(res, x, y, sx, sy)
    # 0.5.0 refused this one: its step for df/dx was 1e-3, the size of
    # the smallest setting, so log(x - h) = log(0)
    x, y, sx, sy = _log_data(1e-3)
    res = fit_xy(_log_model(), x, y, [1.0, 0.5], sy, sx)
    _check_stationary(res, x, y, sx, sy)


def test_conformal_refusal_names_the_level():
    scores = np.linspace(0.0, 1.0, 50)
    with pytest.raises(ValueError, match=r"certify level 0\.9999:"):
        conformal_quantile(scores, alpha=0.0001)
    with pytest.raises(ValueError, match=r"certify level 0\.9995:"):
        conformal_quantile(scores, alpha=0.0005)
    # README example 6 is unchanged
    with pytest.raises(ValueError, match=r"certify level 0\.9: the "
                       r"required rank 6 exceeds n\. Collect at least 9 "):
        conformal_quantile(scores[:5], alpha=0.1)


def test_skip_rows_must_be_a_whole_number(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("preamble\nmore\nx,y\n1,2\n3,4\n")
    ref = load_columns(p, "x", "y", skip_rows=2)
    for ok in (2.0, np.int64(2), "2"):
        x, y, _ = load_columns(p, "x", "y", skip_rows=ok)
        assert np.array_equal(x, ref[0]) and np.array_equal(y, ref[1])
    for bad in (2.7, -1, True, np.nan, "two"):
        with pytest.raises(ValueError, match="skip_rows"):
            load_columns(p, "x", "y", skip_rows=bad)

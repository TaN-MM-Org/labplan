"""0.3.0 regression anchors for two wrong answers, and for the
clearer refusals added in 0.3.0.

- `fit` without `sigmas` stopped after its first, still damped,
  Levenberg-Marquardt step when the readings were small numbers in
  their own units (a current of picoamperes written in amperes): the
  stopping test compared the change of the cost, which then has the
  readings' units squared, with a fixed 1e-12. A straight line fitted
  to such readings must equal ordinary least squares from
  `np.linalg.lstsq`, computed independently in the test, and must not
  depend on the units of the readings. In 0.2.0 the fitted values
  were 1.3 standard errors away from it.
- `propagate` returned an error bar of 0 (and a negative variance in
  `cov`) for a covariance that is not positive semidefinite, for
  example a strongly correlated covariance copied to 3 significant
  digits. It now refuses such a matrix.
"""
import numpy as np
import pytest

from labplan import Model, design, fit, information, propagate, repeats_for

X = np.linspace(0.5, 5.0, 12)
A = np.column_stack([X, np.ones(X.size)])


def _line():
    return Model("line", lambda th, x: th[0] * x[:, 0] + th[1],
                 ("gain", "offset"), "illustrative straight-line model")


def _ols(y):
    """Ordinary least squares, independent of labplan: values and the
    covariance s^2 (A^T A)^-1 with s^2 = RSS / (n - p)."""
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    s2 = np.sum((y - A @ beta) ** 2) / (X.size - 2)
    return beta, s2 * np.linalg.inv(A.T @ A)


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-6, 1e-9, 1e-12,
                                   1e-15])
def test_unweighted_fit_does_not_depend_on_reading_units(scale):
    rng = np.random.default_rng(1)
    y = scale * (2.0 * X + 0.1 + 0.05 * rng.standard_normal(X.size))
    beta, cov = _ols(y)
    res = fit(_line(), X, y, [0.0, 0.0])
    err = np.sqrt(np.diag(cov))
    # 0.2.0 at scale 1e-12: 1.30 and 1.15 standard errors off
    assert np.all(np.abs(res.theta - beta) < 1e-6 * err)
    assert np.allclose(np.sqrt(np.diag(res.cov)), err, rtol=1e-6)


def test_unweighted_fit_of_small_readings_with_tiny_noise():
    """Readings of order 1e-9 with relative noise 1e-5: 0.2.0 stopped
    after two steps, 0.60 standard errors away from least squares."""
    rng = np.random.default_rng(1)
    y = 1e-9 * (2.0 * X + 0.1 + 1e-4 * rng.standard_normal(X.size))
    beta, cov = _ols(y)
    res = fit(_line(), X, y, [0.0, 0.0])
    assert np.all(np.abs(res.theta - beta) < 1e-6 * np.sqrt(np.diag(cov)))


def test_weighted_fits_are_unchanged_by_the_stopping_fix():
    """With `sigmas` the cost is chi-squared and the stopping test is
    the 0.2.0 one; the small-unit fit already worked there."""
    rng = np.random.default_rng(1)
    y = 1e-12 * (2.0 * X + 0.1 + 0.05 * rng.standard_normal(X.size))
    res = fit(_line(), X, y, [0.0, 0.0], sigmas=0.05e-12)
    w = 1.0 / 0.05e-12
    beta = np.linalg.lstsq(A * w, y * w, rcond=None)[0]
    cov = np.linalg.inv((A * w).T @ (A * w))
    assert np.all(np.abs(res.theta - beta) < 1e-6 * np.sqrt(np.diag(cov)))


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-12])
def test_unweighted_noiseless_fits_still_converge(scale):
    """Noiseless readings leave a residual at rounding level; the
    stricter stopping test must still end (no RuntimeError) on the
    truth."""
    y = scale * (2.0 * X + 0.1)
    res = fit(_line(), X, y, [0.0, 0.0])
    assert np.allclose(res.theta, [2.0 * scale, 0.1 * scale],
                       rtol=1e-12, atol=0)
    decay = Model("decay", lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]),
                  ("A", "tau"), "exponential decay, test suite")
    yd = scale * 3.0 * np.exp(-X / 1.7)
    rd = fit(decay, X, yd, [scale, 3.0])
    assert abs(rd.values["A"] / scale - 3.0) < 1e-9
    assert abs(rd.values["tau"] - 1.7) < 1e-9


def _rounded(cov, digits):
    return np.array([[float(f"{v:.{digits}g}") for v in row]
                     for row in cov])


def test_propagate_refuses_a_covariance_that_is_not_positive():
    # a line measured far from x = 0: slope and offset correlated at
    # -0.999995
    m = _line()
    x = np.linspace(100.0, 101.0, 12)
    rng = np.random.default_rng(1)
    y = 2.0 * x + 0.1 + 0.05 * rng.standard_normal(12)
    res = fit(m, x, y, [1.0, 0.0], sigmas=0.05)
    f = lambda th: th[0] * 100.5 + th[1]
    # the full-precision covariance is accepted and gives the closed
    # form a C a^T
    a = np.array([100.5, 1.0])
    good = propagate(f, res.theta, res.cov)
    assert abs(good["sigma"] - np.sqrt(a @ res.cov @ a)) \
        < 1e-6 * good["sigma"]
    # copied to 3 digits it has a negative eigenvalue; 0.2.0 returned
    # sigma = 0.0 here (the right value is 0.0144)
    bad = _rounded(res.cov, 3)
    assert np.linalg.eigvalsh(bad).min() < 0.0
    with pytest.raises(ValueError, match="not positive semidefinite"):
        propagate(f, res.theta, bad)
    # a correlation of 2 between two parameters: 0.2.0 returned
    # sigma 0.0 and cov [[-2.0]] for their difference
    with pytest.raises(ValueError, match="between -1 and 1"):
        propagate(lambda th: th[0] - th[1], [0.0, 0.0],
                  [[1.0, 2.0], [2.0, 1.0]])
    # a parameter held fixed (zero variance, zero covariances) is fine
    out = propagate(lambda th: th[0] + th[1], [1.0, 2.0],
                    [[0.0, 0.0], [0.0, 4.0]])
    assert out["sigma"] == pytest.approx(2.0, rel=1e-9)
    # ...but a zero variance with a non-zero covariance is not
    with pytest.raises(ValueError, match="not positive semidefinite"):
        propagate(lambda th: th[0], [1.0, 2.0], [[0.0, 0.1], [0.1, 4.0]])


def test_planned_covariances_near_the_identifiability_limit_pass():
    """The positivity check must accept every covariance labplan
    itself produces, including a nearly degenerate design (condition
    number 4e9, just inside the 1e10 limit)."""
    x = np.linspace(1e4, 1e4 + 1.0, 12)
    plan = information(_line(), [2.0, 0.1], x, sigmas=0.05)
    assert plan["identifiable"] and plan["condition_number"] > 1e9
    for j in range(2):
        out = propagate(lambda th: th[j], [2.0, 0.1], plan["cov"])
        assert abs(out["sigma"] - np.sqrt(plan["cov"][j, j])) \
            < 1e-6 * out["sigma"]


def test_clearer_refusals():
    m = _line()
    plan = information(m, [2.0, 0.0], X, sigmas=0.05)
    with pytest.raises(ValueError, match="empty dictionary"):
        repeats_for({}, plan)
    with pytest.raises(ValueError, match="whole number"):
        design(m, [2.0, 0.0], X, 2.5, sigmas=0.05)
    # a whole number given as a float is still accepted
    assert design(m, [2.0, 0.0], X, 2.0, sigmas=0.05)["indices"] == \
        design(m, [2.0, 0.0], X, 2, sigmas=0.05)["indices"]
    with pytest.raises(ValueError, match="max_iter"):
        fit(m, X, 2 * X, [1.0, 0.0], sigmas=0.1, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        fit(m, X, 2 * X, [1.0, 0.0], sigmas=0.1, tol=-1.0)

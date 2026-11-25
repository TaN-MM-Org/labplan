"""0.7.0 anchors for `propagate_mc`, propagation by random draws (the
Monte Carlo method of JCGM 101:2008).

- Linear functions: the result for A theta + c is A times the result
  for theta itself plus c, with the same seed (an identity, to
  rounding), and agrees with `propagate` within the random error.
- A product a b of correlated Gaussian parameters: the exact mean
  mu_a mu_b + C_ab and the exact variance
  mu_a^2 C_bb + mu_b^2 C_aa + 2 mu_a mu_b C_ab + C_aa C_bb + C_ab^2
  (from the moments of a Gaussian, worked out in the docstring of the
  test), within 4.5 standard errors of the draws.
- exp(a) with a Gaussian: the exact mean exp(mu + s^2 / 2), standard
  deviation and the quantiles exp(mu -/+ z s), with z the 0.975
  quantile of the standard Gaussian found here by bisection on
  math.erf, within 4.5 standard errors of the draws. The first-order
  error bar of `propagate` is visibly smaller there.
- A parameter with zero variance stays fixed; the same seed repeats
  the result exactly; every refusal fires.
"""
import math

import numpy as np
import pytest

from labplan import propagate, propagate_mc


def _z(p):
    """The p quantile of the standard Gaussian, by bisection on
    Phi(z) = (1 + erf(z / sqrt 2)) / 2."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


THETA = np.array([2.0, 3.0, -1.0])
COV = np.array([[0.04, 0.03, 0.0],
                [0.03, 0.09, -0.01],
                [0.0, -0.01, 0.25]])


def test_linear_functions():
    a = np.array([[1.0, 2.0, 0.5], [-3.0, 0.0, 1.0]])
    c = np.array([0.1, -0.2])
    base = propagate_mc(lambda t: t, THETA, COV, n_draws=20000, seed=4)
    lin = propagate_mc(lambda t: a @ t + c, THETA, COV, n_draws=20000,
                       seed=4)
    assert np.allclose(lin["value"], a @ base["value"] + c, rtol=0,
                       atol=1e-12)                       # measured 3e-14
    assert np.allclose(lin["cov"], a @ base["cov"] @ a.T, rtol=1e-12,
                       atol=0)                           # measured 8e-16
    first = propagate(lambda t: a @ t + c, THETA, COV)
    se = first["sigma"] / math.sqrt(20000)
    assert np.all(np.abs(lin["value"] - first["value"]) < 4.5 * se)
    # measured 0.16 and 1.2 standard errors
    assert np.all(np.abs(lin["sigma"] / first["sigma"] - 1.0)
                  < 4.5 / math.sqrt(2 * 20000))           # measured 0.008


def test_product_of_correlated_gaussians():
    """For jointly Gaussian a, b with means m_a, m_b and covariance C:
    a b = (m_a + u)(m_b + v) with E[u v] = C_ab, so
    E[a b] = m_a m_b + C_ab; and E[u^2 v^2] = C_aa C_bb + 2 C_ab^2
    (the fourth moment of a Gaussian), which gives
    Var(a b) = m_a^2 C_bb + m_b^2 C_aa + 2 m_a m_b C_ab
    + C_aa C_bb + C_ab^2."""
    n = 200000
    out = propagate_mc(lambda t: t[0] * t[1], THETA, COV, n_draws=n)
    ma, mb = THETA[:2]
    caa, cbb, cab = COV[0, 0], COV[1, 1], COV[0, 1]
    mean = ma * mb + cab
    var = (ma ** 2 * cbb + mb ** 2 * caa + 2 * ma * mb * cab
           + caa * cbb + cab ** 2)
    assert abs(out["value"] - mean) < 4.5 * math.sqrt(var / n)
    # measured 0.37 standard errors
    # the standard error of a variance estimate is
    # sqrt((m4 - var^2) / n), with m4 the fourth central moment of a b,
    # estimated here from independent draws of NumPy's own generator
    ab = np.random.default_rng(99).multivariate_normal(
        THETA[:2], COV[:2, :2], 400000)
    prod = ab[:, 0] * ab[:, 1]
    m4 = np.mean((prod - mean) ** 4)
    se_var = math.sqrt((m4 - var ** 2) / n)
    assert abs(out["sigma"] ** 2 - var) < 4.5 * se_var
    # measured 0.15 standard errors
    # first order misses the C_aa C_bb + C_ab^2 term (0.0045)
    first = propagate(lambda t: t[0] * t[1], THETA, COV)
    assert first["sigma"] ** 2 == pytest.approx(var - caa * cbb
                                                - cab ** 2, rel=1e-6)


def test_exponential_of_a_gaussian():
    mu, s = 0.5, 0.3
    n = 200000
    out = propagate_mc(lambda t: math.exp(t[0]), [mu], [[s * s]],
                       n_draws=n, seed=1)
    mean = math.exp(mu + s * s / 2)
    sd = mean * math.sqrt(math.expm1(s * s))
    assert abs(out["value"] - mean) < 4.5 * sd / math.sqrt(n)
    # measured 1.25 standard errors
    assert abs(out["sigma"] / sd - 1) < 4.5 * 2 / math.sqrt(2 * n)
    # measured 0.0018
    z = _z(0.975)
    assert abs(z - 1.959963984540054) < 1e-12
    for q, p in ((out["low"], 0.025), (out["high"], 0.975)):
        exact = math.exp(mu + _z(p) * s)
        # standard error of a sample quantile: sqrt(p (1 - p) / n)
        # divided by the density of exp(a) there
        dens = math.exp(-0.5 * _z(p) ** 2) / (math.sqrt(2 * math.pi)
                                              * s * exact)
        se = math.sqrt(p * (1 - p) / n) / dens
        assert abs(q - exact) < 4.5 * se    # measured 0.006 and 0.6 se
    # the first-order error bar exp(mu) s is smaller than the true
    # one by a factor 1.070 (exact: sd / (exp(mu) s))
    first = propagate(lambda t: math.exp(t[0]), [mu], [[s * s]])
    assert first["sigma"] == pytest.approx(math.exp(mu) * s, rel=1e-6)
    assert sd / first["sigma"] == pytest.approx(1.0700, abs=1e-4)
    assert out["sigma"] / first["sigma"] > 1.05     # measured 1.068


def test_fixed_parameter_and_seed():
    cov = np.array([[0.04, 0.0], [0.0, 0.0]])
    out = propagate_mc(lambda t: t, [1.0, 5.0], cov, n_draws=1000)
    assert out["sigma"][1] == 0.0 and out["value"][1] == 5.0
    again = propagate_mc(lambda t: t, [1.0, 5.0], cov, n_draws=1000)
    assert np.array_equal(out["value"], again["value"])
    other = propagate_mc(lambda t: t, [1.0, 5.0], cov, n_draws=1000,
                         seed=2)
    assert other["value"][0] != out["value"][0]
    fresh = propagate_mc(lambda t: t[0], [1.0, 5.0], cov, n_draws=1000,
                         seed=None)
    assert fresh["seed"] is None and isinstance(fresh["value"], float)
    assert out["n_draws"] == 1000 and out["level"] == 0.95


def test_refusals():
    f = lambda t: t[0]                                      # noqa: E731
    for bad in (0, 2.5, True, "many", np.nan):
        with pytest.raises(ValueError, match="n_draws"):
            propagate_mc(f, [1.0], [[1.0]], n_draws=bad)
    with pytest.raises(ValueError, match="at least 400"):
        propagate_mc(f, [1.0], [[1.0]], n_draws=399)
    propagate_mc(f, [1.0], [[1.0]], n_draws=400)
    for bad in (0.0, 1.0, 1.5, True, "high"):
        with pytest.raises(ValueError, match="level"):
            propagate_mc(f, [1.0], [[1.0]], level=bad)
    for bad in (-1, 1.5, "x", True):
        with pytest.raises(ValueError, match="seed"):
            propagate_mc(f, [1.0], [[1.0]], seed=bad)
    # the checks of `propagate`
    with pytest.raises(ValueError, match="positive semidefinite"):
        propagate_mc(lambda t: t[0] + t[1], [1.0, 1.0],
                     [[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="cov is None"):
        propagate_mc(f, [1.0], None)
    with pytest.raises(ValueError, match="1-D array"):
        propagate_mc(lambda t: np.ones((2, 2)), [1.0], [[1.0]])
    # a quantity that is not finite within the error bars
    with pytest.raises(ValueError, match="not finite at draw"):
        propagate_mc(lambda t: math.log(t[0]) if t[0] > 0 else math.nan,
                     [1.0], [[1.0]], n_draws=1000)
    with pytest.raises(ValueError, match="numbers at draw"):
        propagate_mc(lambda t: np.ones(2 if t[0] > 1.0 else 1), [1.0],
                     [[1.0]], n_draws=1000)

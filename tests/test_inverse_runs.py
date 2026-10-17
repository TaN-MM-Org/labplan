"""0.4.0 anchors for the two new tools.

`inverse_predict`: on a straight line it equals, value and error bar,
the propagation-of-error formula of the NIST/SEMATECH e-Handbook of
Statistical Methods, section 2.3.6.7.3, typed into the test; on an
exponential decay it equals the closed-form inverse and its partial
derivatives; the covariance of several readings equals the closed
form; seeded replicate calibrations (ordinary least squares done in
the test with NumPy) scatter as predicted; and the slope-change
refusal fires exactly where seeded simulation shows the first-order
error bar going wrong.

`runs_test`: the distribution of the number of runs, its mean and its
variance equal the counts over every arrangement of n1 plus and n2
minus signs for n1, n2 <= 7; the tail probabilities equal exact
integer arithmetic (`math.comb`, `fractions`) for larger n; the
false-alarm rate on random signs and on residuals of correct
straight-line fits is at most the nominal level; and it detects the
curvature of README example 11 when chi-squared, with overstated
errors, does not.
"""
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from labplan import (Model, fit, goodness_of_fit, information,
                     inverse_predict, runs_test)

X = np.linspace(0.5, 5.0, 12)


def _line():
    return Model("sensor line", lambda th, x: th[0] + th[1] * x[:, 0],
                 ("a", "b"), "illustrative straight-line sensor model")


def _nist_u2(a, b, cov, y, s_y):
    """NIST/SEMATECH e-Handbook 2.3.6.7.3, X' = (Y' - a) / b."""
    return ((1 / b) ** 2 * s_y ** 2 + (-1 / b) ** 2 * cov[0, 0]
            + (-(y - a) / b ** 2) ** 2 * cov[1, 1]
            + 2 * (-1 / b) * (-(y - a) / b ** 2) * cov[0, 1])


def _calibration():
    rng = np.random.default_rng(1)
    y = 0.1 + 2.0 * X + 0.05 * rng.standard_normal(X.size)
    return fit(_line(), X, y, [0.0, 1.0], sigmas=0.05)


def test_line_equals_the_nist_handbook_formula():
    res = _calibration()
    a, b = res.theta
    for y0, s_y in ((1.5, 0.0), (5.0, 0.05), (9.7, 0.2)):
        out = inverse_predict(_line(), res, y0, (0.5, 5.0),
                              reading_sigma=s_y)
        assert abs(out["value"] - (y0 - a) / b) < 1e-12 * abs(out["value"])
        want = math.sqrt(_nist_u2(a, b, res.cov, y0, s_y))
        assert abs(out["sigma"] - want) < 1e-7 * want
        assert out["sigma_from_reading"] == pytest.approx(s_y / b,
                                                          rel=1e-9)
        assert out["sigma"] ** 2 == pytest.approx(
            out["sigma_from_reading"] ** 2
            + out["sigma_from_parameters"] ** 2, rel=1e-12)
        assert out["slope"] == pytest.approx(b, rel=1e-9)
    # several readings: the closed-form covariance
    ys = np.array([1.5, 5.0, 9.7])
    out = inverse_predict(_line(), res, ys, (0.5, 5.0), reading_sigma=0.05)
    g = np.column_stack([-np.ones(3) / b, -(ys - a) / b ** 2])
    want = g @ res.cov @ g.T + np.diag(np.full(3, 0.05 ** 2 / b ** 2))
    d = np.sqrt(np.diag(want))
    assert np.all(np.abs(out["cov"] - want) < 1e-7 * np.outer(d, d))
    assert np.allclose(out["value"], (ys - a) / b, rtol=1e-12)
    # theta= and cov= give the same as the FitResult
    same = inverse_predict(_line(), reading=ys, bracket=(0.5, 5.0),
                           reading_sigma=0.05, theta=res.theta,
                           cov=res.cov)
    assert np.array_equal(same["value"], out["value"])
    assert np.array_equal(same["cov"], out["cov"])


def test_exponential_decay_closed_form():
    m = Model("decay", lambda th, x: th[0] * np.exp(-x[:, 0] / th[1])
              + th[2], ("A", "tau", "B"), "exponential decay, test suite")
    th = np.array([3.0, 1.7, 0.4])
    cov = np.array([[4e-4, 1e-5, -2e-5],
                    [1e-5, 9e-4, -3e-5],
                    [-2e-5, -3e-5, 1e-4]])
    for y0 in (3.1, 1.0, 0.7):
        out = inverse_predict(m, reading=y0, bracket=(0.0, 8.0),
                              reading_sigma=0.01, theta=th, cov=cov)
        A, tau, B = th
        x0 = -tau * math.log((y0 - B) / A)
        g = np.array([tau / A, x0 / tau, tau / (y0 - B)])
        u2 = g @ cov @ g + (tau / (y0 - B)) ** 2 * 0.01 ** 2
        assert abs(out["value"] - x0) < 1e-12 * max(1.0, x0)
        assert abs(out["sigma"] - math.sqrt(u2)) < 1e-6 * math.sqrt(u2)


def test_replicated_calibrations_scatter_as_predicted():
    """20000 seeded replicate calibrations of the 12-point line, each
    followed by one new reading at x = 3.3: ordinary least squares and
    the inverse (y - a) / b computed in the test. The standard
    deviation of the 20000 inverse values matches the planned error
    bar (standard error of a standard deviation 0.5 %; tolerance 3 %)."""
    m = _line()
    truth = np.array([0.1, 2.0])
    plan = information(m, truth, X, sigmas=0.05)
    y_true = truth[0] + truth[1] * 3.3
    out = inverse_predict(m, reading=y_true, bracket=(0.5, 5.0),
                          reading_sigma=0.03, theta=truth,
                          cov=plan["cov"])
    rng = np.random.default_rng(17)
    A = np.column_stack([np.ones(X.size), X])
    ys = (A @ truth) + 0.05 * rng.standard_normal((20000, X.size))
    beta = ys @ np.linalg.pinv(A).T
    y_new = y_true + 0.03 * rng.standard_normal(20000)
    x0 = (y_new - beta[:, 0]) / beta[:, 1]
    assert abs(np.std(x0) / out["sigma"] - 1.0) < 0.03
    assert abs(np.mean(x0) - 3.3) < 4.5 * out["sigma"] / math.sqrt(20000)


def test_slope_change_refusal_matches_simulation():
    """A saturating response A (1 - exp(-x / c)): where it accepts, the
    first-order error bar equals the scatter of the exact inverse
    -c ln(1 - y / A) over seeded readings within 2 % (slope change
    1 %) and is at most 3 % small just above the 10 % limit; where it
    refuses
    (the slope changes by 65 % within one error bar) that scatter is
    more than 25 % larger than the first-order value."""
    m = Model("saturating", lambda th, x: th[0] * (1 - np.exp(-x[:, 0]
                                                             / th[1])),
              ("A", "c"), "saturating response, test suite")
    th = np.array([10.0, 1.0])
    cov = np.diag([1e-6, 1e-8])
    rng = np.random.default_rng(0)

    def scatter(y0, u):
        y = y0 + u * rng.standard_normal(200000)
        return np.std(-np.log(1 - y[y < 10.0] / 10.0))

    ok = inverse_predict(m, reading=5.0, bracket=(0.0, 8.0),
                         reading_sigma=0.05, theta=th, cov=cov)
    assert ok["slope_change"] < 0.1
    assert abs(scatter(5.0, 0.05) / ok["sigma"] - 1.0) < 0.02
    # just above the default limit (10.5 %), let through: the
    # first-order error bar is 1.4 % below the simulated scatter
    edge = inverse_predict(m, reading=9.5, bracket=(0.0, 8.0),
                           reading_sigma=0.05, theta=th, cov=cov,
                           max_slope_change=0.2)
    assert 0.1 < edge["slope_change"] < 0.11
    assert 1.0 < scatter(9.5, 0.05) / edge["sigma"] < 1.03
    with pytest.raises(ValueError, match="slope of the model changes"):
        inverse_predict(m, reading=9.9, bracket=(0.0, 8.0),
                        reading_sigma=0.05, theta=th, cov=cov)
    loose = inverse_predict(m, reading=9.9, bracket=(0.0, 8.0),
                            reading_sigma=0.05, theta=th, cov=cov,
                            max_slope_change=1.0)
    assert loose["slope_change"] > 0.5
    assert scatter(9.9, 0.05) > 1.25 * loose["sigma"]


def test_inverse_predict_refusals():
    res = _calibration()
    line = _line()
    with pytest.raises(ValueError, match="does not reach"):
        inverse_predict(line, res, 50.0, (0.5, 5.0))
    with pytest.raises(ValueError, match="low < high"):
        inverse_predict(line, res, 5.0, (5.0, 0.5))
    with pytest.raises(ValueError, match="two numbers"):
        inverse_predict(line, res, 5.0, 3.0)
    with pytest.raises(ValueError, match="bracket="):
        inverse_predict(line, res, 5.0)
    with pytest.raises(ValueError, match="finite"):
        inverse_predict(line, res, np.nan, (0.5, 5.0))
    with pytest.raises(ValueError, match=">= 0"):
        inverse_predict(line, res, 5.0, (0.5, 5.0), reading_sigma=-1.0)
    with pytest.raises(ValueError, match="one per reading"):
        inverse_predict(line, res, [5.0, 6.0], (0.5, 5.0),
                        reading_sigma=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="not both"):
        inverse_predict(line, res, 5.0, (0.5, 5.0), theta=res.theta,
                        cov=res.cov)
    with pytest.raises(ValueError, match="not identifiable"):
        inverse_predict(line, reading=5.0, bracket=(0.5, 5.0),
                        theta=[0.0, 1.0], cov=None)
    with pytest.raises(ValueError, match="positive semidefinite"):
        inverse_predict(line, reading=5.0, bracket=(0.5, 5.0),
                        theta=[0.0, 1.0], cov=[[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="n_scan"):
        inverse_predict(line, res, 5.0, (0.5, 5.0), n_scan=2)
    with pytest.raises(ValueError, match="labplan Model"):
        inverse_predict("line", res, 5.0, (0.5, 5.0))
    with pytest.raises(ValueError, match="reading="):
        inverse_predict(line, res, bracket=(0.5, 5.0))
    with pytest.raises(ValueError, match="1-D"):
        inverse_predict(line, res, [[5.0]], (0.5, 5.0))
    with pytest.raises(ValueError, match="FitResult from"):
        inverse_predict(line, {"theta": res.theta}, 5.0, (0.5, 5.0))
    with pytest.raises(ValueError, match="theta= and cov="):
        inverse_predict(line, reading=5.0, bracket=(0.5, 5.0))
    with pytest.raises(ValueError, match="2 finite numbers"):
        inverse_predict(line, reading=5.0, bracket=(0.5, 5.0),
                        theta=[1.0], cov=np.eye(2))
    with pytest.raises(ValueError, match=r"\(2, 2\)"):
        inverse_predict(line, reading=5.0, bracket=(0.5, 5.0),
                        theta=[0.0, 1.0], cov=np.eye(3))
    with pytest.raises(ValueError, match="symmetric"):
        inverse_predict(line, reading=5.0, bracket=(0.5, 5.0),
                        theta=[0.0, 1.0], cov=[[1.0, 0.5], [0.0, 1.0]])
    with pytest.raises(ValueError, match="max_slope_change"):
        inverse_predict(line, res, 5.0, (0.5, 5.0), max_slope_change=0.0)
    other = Model("other", lambda th, x: th[0] + th[1] * x[:, 0],
                  ("a", "b"), "another straight-line model")
    with pytest.raises(ValueError, match="not of 'other'"):
        inverse_predict(other, res, 5.0, (0.5, 5.0))
    bowl = Model("bowl", lambda th, x: th[0] * (x[:, 0] - th[1]) ** 2,
                 ("k", "x_min"), "parabola, test suite")
    cov = np.diag([1e-6, 1e-6])
    with pytest.raises(ValueError, match="at 2 settings"):
        inverse_predict(bowl, reading=0.5, bracket=(0.0, 2.0),
                        theta=[2.0, 1.0], cov=cov)
    with pytest.raises(ValueError, match="flat at the solution"):
        inverse_predict(bowl, reading=0.0, bracket=(0.0, 2.0),
                        theta=[2.0, 1.0], cov=cov)
    # one branch is fine
    one = inverse_predict(bowl, reading=0.5, bracket=(1.0, 2.0),
                          theta=[2.0, 1.0], cov=cov)
    assert one["value"] == pytest.approx(1.5, rel=1e-12)
    step = Model("clipped", lambda th, x: th[0] * np.clip(x[:, 0], 0, 1),
                 ("k",), "clipped response, test suite")
    with pytest.raises(ValueError, match="whole range"):
        inverse_predict(step, reading=2.0, bracket=(0.0, 2.0),
                        theta=[2.0], cov=[[1e-6]])


# ------------------------------------------------------------------
# runs_test


def _arrangements(n1, n2):
    for pos in itertools.combinations(range(n1 + n2), n1):
        s = -np.ones(n1 + n2)
        s[list(pos)] = 1.0
        yield s


def _runs(s):
    return 1 + int(np.sum(s[1:] != s[:-1]))


def test_runs_distribution_by_enumeration():
    for n1 in range(1, 8):
        for n2 in range(1, 8):
            counts = {}
            for s in _arrangements(n1, n2):
                r = _runs(s)
                counts[r] = counts.get(r, 0) + 1
            total = sum(counts.values())
            mean = sum(r * c for r, c in counts.items()) / total
            var = sum((r - mean) ** 2 * c for r, c in counts.items()) \
                / total
            for r in counts:
                # an arrangement with exactly r runs
                s = next(a for a in _arrangements(n1, n2)
                         if _runs(a) == r)
                out = runs_test(s)
                low = sum(c for k, c in counts.items() if k <= r) / total
                high = sum(c for k, c in counts.items() if k >= r) / total
                assert out["runs"] == r
                assert abs(out["p_value"] - low) < 1e-12 * low
                assert abs(out["p_value_high"] - high) < 1e-12 * high
                assert out["expected_runs"] == pytest.approx(mean,
                                                             rel=1e-12)
                assert out["sigma_runs"] ** 2 == pytest.approx(
                    var, rel=1e-12, abs=1e-15)


def _exact_low(r, n1, n2):
    """P(R <= r) in exact rational arithmetic (independent formula:
    count of arrangements with exactly r runs)."""
    def count(rr):
        if rr % 2 == 0:
            k = rr // 2
            return 2 * math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k - 1)
        k = (rr - 1) // 2
        return (math.comb(n1 - 1, k) * math.comb(n2 - 1, k - 1)
                + math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k))
    return Fraction(sum(count(rr) for rr in range(2, r + 1)),
                    math.comb(n1 + n2, n1))


@pytest.mark.parametrize("n1,n2", [(25, 40), (200, 200), (900, 700)])
def test_runs_tails_against_exact_integers(n1, n2):
    mean = 1 + 2 * n1 * n2 / (n1 + n2)
    sd = math.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
                   / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
    rng = np.random.default_rng(n1)
    for r in sorted({int(mean + k * sd) for k in (-4, -2, -1, 0, 1, 3)}):
        # an arrangement with exactly r runs: alternate blocks
        k_plus = (r + 1) // 2
        k_minus = r // 2
        cut_p = np.sort(rng.choice(np.arange(1, n1), k_plus - 1,
                                   replace=False))
        cut_m = np.sort(rng.choice(np.arange(1, n2), k_minus - 1,
                                   replace=False))
        bp = np.diff(np.concatenate([[0], cut_p, [n1]]))
        bm = np.diff(np.concatenate([[0], cut_m, [n2]]))
        seq = []
        for i in range(r):
            seq += [1.0] * bp[i // 2] if i % 2 == 0 else [-1.0] * bm[i // 2]
        out = runs_test(np.array(seq))
        assert out["runs"] == r
        want = float(_exact_low(r, n1, n2))
        assert abs(out["p_value"] - want) < 1e-9 * want
        want_high = float(1 - _exact_low(r - 1, n1, n2))
        assert abs(out["p_value_high"] - want_high) < 1e-9 * want_high


def test_false_alarm_rate_is_at_most_nominal():
    """4000 seeded sets of random signs (n = 30) and 4000 sets of
    residuals of correct straight-line fits (n = 12 and 30, least
    squares in NumPy): the fraction with p_value <= 0.05 is at most
    0.05 plus 4.5 binomial standard errors."""
    rng = np.random.default_rng(5)
    se = math.sqrt(0.05 * 0.95 / 4000)

    def p(r):
        try:
            return runs_test(r)["p_value"]
        except ValueError:             # all one sign: 2 in 2^n
            return 1.0

    hits = sum(p(rng.standard_normal(30)) <= 0.05 for _ in range(4000))
    assert hits / 4000 <= 0.05 + 4.5 * se
    for n in (12, 30):
        x = np.linspace(0.5, 5.0, n)
        A = np.column_stack([x, np.ones(n)])
        hits = 0
        for _ in range(4000):
            y = 2 * x + 0.1 + 0.05 * rng.standard_normal(n)
            hits += p(y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) <= 0.05
        assert hits / 4000 <= 0.05 + 4.5 * se


def test_runs_see_a_bend_that_chi2_misses():
    """Curved readings like README example 11's (0.05 x^2 added), fitted with
    a straight line and reading errors stated 3 times too large: chi2
    looks fine (p = 0.994), the runs of the residuals ordered by x do not
    (5 runs where 12.25 are expected, p = 0.001)."""
    line = Model("sensor line", lambda th, x: th[0] * x[:, 0] + th[1],
                 ("gain", "offset"), "illustrative straight-line model")
    x = np.linspace(0.5, 5.0, 24)
    rng = np.random.default_rng(2)
    y = 2.0 * x + 0.1 + 0.05 * x ** 2 + 0.05 * rng.standard_normal(x.size)
    res = fit(line, x, y, [1.0, 0.0], sigmas=0.15)
    assert goodness_of_fit(res)["p_value"] > 0.05
    resid = y - line.predict(res.theta, x)
    shuffled = rng.permutation(x.size)
    out = runs_test(resid[shuffled], order=x[shuffled])
    assert out["runs"] == runs_test(resid)["runs"]
    assert out["p_value"] < 0.01


def test_runs_test_refusals_and_zeros():
    with pytest.raises(ValueError, match="all 3 nonzero"):
        runs_test([1.0, 2.0, 0.5])
    with pytest.raises(ValueError, match="at least 2"):
        runs_test([1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        runs_test([1.0, np.nan, -1.0])
    with pytest.raises(ValueError, match="1-D"):
        runs_test(np.ones((2, 2)))
    with pytest.raises(ValueError, match="one value per residual"):
        runs_test([1.0, -1.0], order=[1.0])
    with pytest.raises(ValueError, match="order must be finite"):
        runs_test([1.0, -1.0], order=[1.0, np.inf])
    out = runs_test([1.0, 0.0, -1.0, 0.0, 1.0])
    assert out["n_zero"] == 2 and out["runs"] == 3
    assert out["n_positive"] == 2 and out["n_negative"] == 1

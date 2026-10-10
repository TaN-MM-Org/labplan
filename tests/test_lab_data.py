"""0.3.0 anchors for the two lab-data features.

`goodness_of_fit`: the chi-squared tail probabilities are checked
against closed forms computed independently in the test with the
standard library (exp for 2 degrees of freedom, the finite Poisson
sum for every even number, erfc plus the upward recurrence for every
odd number), against direct numerical integration of the
chi-squared density, against seeded chi-squared draws, and, for a
straight-line fit with Gaussian reading errors, against the uniform
distribution its p-values must have.

`load_columns`: known-answer files in several layouts are read back
to the numbers typed into them and agree with NumPy's own text
reader (`np.genfromtxt`) as an independent parser; every refusal
fires and names the line.
"""
import math

import numpy as np
import pytest

from labplan import (FitResult, Model, fit, goodness_of_fit,
                     load_columns, save_measurements_csv)
from labplan.stats import gamma_regularized


def _q_closed_form(k, t):
    """P(chi2_k >= t) from closed forms (independent of labplan)."""
    x = t / 2.0
    if k % 2 == 0:
        # Poisson sum: exp(-x) sum_{j < k/2} x^j / j!
        return sum(math.exp(-x + j * math.log(x) - math.lgamma(j + 1))
                   if x > 0 else float(j == 0) for j in range(k // 2))
    # Q(1/2, x) = erfc(sqrt x); Q(a + 1, x) = Q(a, x) + x^a e^-x / Gamma(a+1)
    q, a = math.erfc(math.sqrt(x)), 0.5
    while a < k / 2.0:
        q += math.exp(a * math.log(x) - x - math.lgamma(a + 1.0))
        a += 1.0
    return q


def test_tail_probabilities_against_closed_forms():
    worst = 0.0
    for k in range(1, 81):
        for t in (1e-8, 1e-3, 0.3, 1.0, 2.0, 5.0, 10.0, 33.0, 80.0,
                  150.0, 400.0, 1200.0):
            want = _q_closed_form(k, t)
            got = goodness_of_fit(chi2=t, dof=k)
            assert got["p_value"] + got["p_value_low"] == \
                pytest.approx(1.0, abs=1e-15)
            if want > 1e-290:
                worst = max(worst, abs(got["p_value"] - want) / want)
    assert worst < 1e-12
    # small lower tails, where 1 - Q loses digits: P(chi2_2 <= t) =
    # 1 - exp(-t/2) = -expm1(-t/2), and P(chi2_1 <= t) = erf(sqrt(t/2))
    for t in (1e-12, 1e-6, 1e-3):
        g2 = goodness_of_fit(chi2=t, dof=2)["p_value_low"]
        assert abs(g2 + math.expm1(-t / 2)) < 1e-13 * g2
        g1 = goodness_of_fit(chi2=t, dof=1)["p_value_low"]
        assert abs(g1 - math.erf(math.sqrt(t / 2))) < 1e-13 * g1


def test_tail_probabilities_against_numerical_integration():
    """P(chi2_k <= t) as the integral of the density
    s^(k/2-1) e^(-s/2) / (2^(k/2) Gamma(k/2)) from 0 to t, by
    Simpson's rule on a substitution s = u^2 that removes the
    square-root singularity at 0 for k = 1."""
    for k in (1, 3, 4, 7, 12):
        for t in (0.5, 3.0, 9.0, 20.0):
            u = np.linspace(0.0, math.sqrt(t), 20001)
            # density at s = u^2 times ds/du = 2 u
            f = (np.where(u > 0, u, 1.0) ** (k - 1.0) * 2.0
                 * np.exp(-u * u / 2.0)
                 / (2.0 ** (k / 2.0) * math.gamma(k / 2.0)))
            if k > 1:
                f[0] = 0.0
            h = u[1] - u[0]
            simpson = h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum()
                                 + 2.0 * f[2:-1:2].sum())
            got = goodness_of_fit(chi2=t, dof=k)["p_value_low"]
            assert abs(got - simpson) < 1e-10


def test_tail_probabilities_against_seeded_draws():
    rng = np.random.default_rng(4)
    for k in (1, 5, 30):
        draws = rng.chisquare(k, 20000)
        for t in np.quantile(draws, [0.05, 0.5, 0.95]):
            emp = float(np.mean(draws >= t))
            p = goodness_of_fit(chi2=t, dof=k)["p_value"]
            se = math.sqrt(p * (1 - p) / draws.size)
            assert abs(emp - p) < 4.5 * se


def test_p_values_of_correct_fits_are_uniform():
    """A straight line with Gaussian reading errors of exactly the
    stated size: chi2 follows the chi-squared distribution with
    n - 2 degrees of freedom, so the p-values are uniform on (0, 1).
    400 seeded fits: each decile holds 40 +/- 4.5 binomial standard
    deviations, and the mean p-value is 1/2 within 4.5 standard
    errors of the uniform mean (sqrt(1/12/400))."""
    m = Model("line", lambda th, x: th[0] * x[:, 0] + th[1],
              ("gain", "offset"), "illustrative straight-line model")
    x = np.linspace(0.5, 5.0, 10)
    y0 = m.predict([2.0, 0.1], x)
    rng = np.random.default_rng(12)
    ps = []
    for _ in range(400):
        res = fit(m, x, y0 + 0.05 * rng.standard_normal(x.size),
                  [1.0, 0.0], sigmas=0.05)
        g = goodness_of_fit(res)
        assert g["dof"] == 8 and g["chi2"] == res.chi2
        ps.append(g["p_value"])
    counts = np.histogram(ps, bins=10, range=(0.0, 1.0))[0]
    sd = math.sqrt(400 * 0.1 * 0.9)
    assert np.all(np.abs(counts - 40) < 4.5 * sd)
    assert abs(np.mean(ps) - 0.5) < 4.5 * math.sqrt(1 / 12 / 400)


def test_goodness_of_fit_fields_and_refusals():
    g = goodness_of_fit(chi2=18.0, dof=8)
    assert g["reduced_chi2"] == 18.0 / 8
    assert g["birge_ratio"] == pytest.approx(math.sqrt(18.0 / 8), rel=1e-15)
    # 8 degrees of freedom: Q = exp(-9) (1 + 9 + 81/2 + 729/6)
    assert g["p_value"] == pytest.approx(
        math.exp(-9.0) * (1 + 9 + 81 / 2 + 729 / 6), rel=1e-13)
    m = Model("line", lambda th, x: th[0] * x[:, 0] + th[1],
              ("gain", "offset"), "illustrative straight-line model")
    x = np.linspace(0.5, 5.0, 6)
    no_sigma = fit(m, x, 2 * x + 0.01 * np.sin(7 * x), [1.0, 0.0])
    with pytest.raises(ValueError, match="no chi-squared"):
        goodness_of_fit(no_sigma)
    exact = fit(m, x[:2], 2 * x[:2], [1.0, 0.0], sigmas=0.1)
    with pytest.raises(ValueError, match="dof = 0"):
        goodness_of_fit(exact)
    with pytest.raises(ValueError, match="not both"):
        goodness_of_fit(exact, chi2=1.0, dof=1)
    with pytest.raises(ValueError, match="both chi2"):
        goodness_of_fit(chi2=1.0)
    with pytest.raises(ValueError, match="FitResult"):
        goodness_of_fit({"chi2": 1.0})
    with pytest.raises(ValueError, match="finite"):
        goodness_of_fit(chi2=-1.0, dof=3)
    with pytest.raises(ValueError, match="finite"):
        goodness_of_fit(chi2=float("nan"), dof=3)
    with pytest.raises(ValueError, match="whole number"):
        goodness_of_fit(chi2=1.0, dof=2.5)
    with pytest.raises(ValueError):
        gamma_regularized(0.0, 1.0)
    with pytest.raises(ValueError):
        gamma_regularized(1.0, -1.0)
    assert isinstance(no_sigma, FitResult)


# ------------------------------------------------------------------
# load_columns

EXPORT = """\
# instrument: bench supply sweep (illustrative)
# operator: test
Time (s),Bias (V),Current (A),Error (A),Note
0.0,0.5,1.01e-12,2e-14,start
1.0,1.0,2.03e-12,2e-14,
2.0,1.5,2.98e-12,3e-14,"spike, ignored"

3.0,2.0,4.02e-12,3e-14,end
"""


def test_reads_named_columns_and_agrees_with_numpy(tmp_path):
    p = tmp_path / "sweep.csv"
    p.write_text(EXPORT)
    x, y, s = load_columns(p, x="Bias (V)", y="Current (A)",
                           sigma="Error (A)")
    assert x.shape == (4, 1)
    assert np.array_equal(x[:, 0], [0.5, 1.0, 1.5, 2.0])
    assert np.array_equal(y, [1.01e-12, 2.03e-12, 2.98e-12, 4.02e-12])
    assert np.array_equal(s, [2e-14, 2e-14, 3e-14, 3e-14])
    ref = np.genfromtxt(p, delimiter=",", comments="#", skip_header=3,
                        usecols=(1, 2, 3))
    assert np.array_equal(np.column_stack([x[:, 0], y, s]), ref)
    # by position, two setting columns, no sigma
    x2, y2, s2 = load_columns(p, x=[0, 1], y=2)
    assert s2 is None and x2.shape == (4, 2)
    assert np.array_equal(x2[:, 0], [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(y2, y)


@pytest.mark.parametrize("delim", ["\t", ";", " "])
def test_other_delimiters_and_decimal_comma(tmp_path, delim):
    rng = np.random.default_rng(3)
    data = rng.standard_normal((6, 3))
    data[:, 2] = np.abs(data[:, 2]) + 0.1          # positive errors
    sep = "   " if delim == " " else delim
    p = tmp_path / "t.txt"
    lines = [sep.join(["f_Hz", "gain", "err"])]
    lines += [sep.join(repr(float(v)) for v in row) for row in data]
    p.write_text("\n".join(lines) + "\n")
    x, y, s = load_columns(p, "f_Hz", "gain", "err")
    assert np.array_equal(np.column_stack([x[:, 0], y, s]), data)
    ref = np.genfromtxt(p, delimiter=None if delim == " " else delim,
                        skip_header=1)
    assert np.array_equal(ref, data)
    if delim == ";":
        q = tmp_path / "comma.csv"
        q.write_text("﻿U;I\n1,5;2,25\n-0,5;1E-3\n",
                     encoding="utf-8")
        x, y, s = load_columns(q, "U", "I", decimal=",")
        assert np.array_equal(x[:, 0], [1.5, -0.5])
        assert np.array_equal(y, [2.25, 1e-3])


def test_round_trip_with_the_labplan_format(tmp_path):
    rng = np.random.default_rng(9)
    x = rng.standard_normal((5, 2))
    y = rng.standard_normal(5)
    sig = np.abs(rng.standard_normal(5)) + 0.1
    p = tmp_path / "m.csv"
    save_measurements_csv(p, x, y, sig)
    x2, y2, s2 = load_columns(p, ["x1", "x2"], "y", "sigma")
    assert np.array_equal(x2, x) and np.array_equal(y2, y)
    assert np.array_equal(s2, sig)


def test_preamble_and_no_header(tmp_path):
    p = tmp_path / "scope.txt"
    p.write_text("Record Length,5\nSample Interval,1e-9\n"
                 "1e-9,0.25\n2e-9,0.5\n3e-9,0.75\n")
    x, y, s = load_columns(p, 0, 1, skip_rows=2, header=False)
    assert np.array_equal(x[:, 0], [1e-9, 2e-9, 3e-9])
    assert np.array_equal(y, [0.25, 0.5, 0.75])
    with pytest.raises(ValueError, match="header=False"):
        load_columns(p, "t", 1, skip_rows=2, header=False)


def test_load_columns_refusals(tmp_path):
    p = tmp_path / "bad.csv"

    def refuses(text, msg, *args, **kw):
        p.write_text(text)
        with pytest.raises(ValueError, match=msg):
            load_columns(p, *(args or ("a", "b")), **kw)

    refuses("a,b\n1,2\n", "no column named 'c'.*\\['a', 'b'\\]", "c", "b")
    refuses("a,b,a\n1,2,3\n", "appears 2 times", "a", "b")
    refuses("a,b\n1,2\n", "asked for twice", "a", "a")
    refuses("a,b\n1,2\n3\n", "line 3: 1 fields")
    refuses("a,b\n1,\n", "line 2, column 'b'.*not a number")
    refuses("a,b\n1,abc\n", "line 2, column 'b'.*not a number")
    refuses("a,b\n1,nan\n", "line 2.*finite")
    refuses("a,b\n1,inf\n", "line 2.*finite")
    refuses("a,b\n1,1_000\n", "not a number")
    refuses("a;b\n1,5;2\n", "decimal=','")
    refuses("a;b\n1.000,5;2\n", "thousands", decimal=",")
    refuses("a,b;c\n1,2;3\n", "ambiguous")
    refuses("a,b\n", "no data rows")
    refuses("# only a comment\n\n", "no header")
    refuses("a,b,s\n1,2,0\n", "line 2: reading errors", "a", "b", "s")
    refuses("a,b\n1,2\n", "outside", 0, 5)
    refuses("a,b\n1,2\n", "delimiter and the decimal", delimiter=",",
            decimal=",")
    refuses("a,b\n1,2\n", "str", 0.5, "b")

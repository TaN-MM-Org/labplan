"""0.6.0 anchors for `design_weights` and `design_efficiency`.

Independent references, each computed in this file:

- Polynomials of degree m = 1 to 5 on [-1, 1]: the classical best
  weights, 1 / (m + 1) at -1, 1 and the roots of the derivative of the
  Legendre polynomial P_m (the roots of (1 - x^2) P_m'(x)). The test
  does not take this on trust: it checks, with its own NumPy code on a
  grid of 20001 points, that this design satisfies the equivalence
  theorem (the largest d(x) on [-1, 1] equals p), which proves it best.
- Exponential decay A exp(-t / tau) on [0, T] with T > tau: weights
  1/2 at t = 0 and t = tau (worked out below, and checked by the
  equivalence theorem in the test).
- The identity sum_i w_i d_i = p at every result, and the efficiency
  bound p / max d_i against the exact optimum of the quadratic for 300
  random designs.
- Rounding to whole numbers: the best multiset of measurements found by
  checking every one, for 24 small problems.
"""
import itertools
import math

import numpy as np
import pytest

from labplan import (Model, design, design_efficiency, design_weights,
                     information, repeats_for)


def _poly(m):
    """c0 + c1 x + ... + cm x^m, with its exact slopes (the powers of x),
    so that comparisons with the closed forms are not limited by
    finite differences; the other models below use numerical slopes."""
    return Model(f"poly{m}",
                 lambda th, x: sum(th[k] * x[:, 0] ** k
                                   for k in range(m + 1)),
                 tuple(f"c{k}" for k in range(m + 1)),
                 "polynomial, test suite",
                 jac=lambda th, x: _vander(x[:, 0], m))


def _legendre_support(m):
    """-1, 1 and the m - 1 roots of P_m'(x), from NumPy's Legendre
    module (real for every m)."""
    from numpy.polynomial import legendre
    c = np.zeros(m + 1)
    c[m] = 1.0
    inner = np.real(legendre.legroots(legendre.legder(c)))
    inner = np.where(np.abs(inner) < 1e-12, 0.0, inner)
    return np.sort(np.concatenate([[-1.0, 1.0], inner]))


def _vander(x, m):
    return np.vander(np.asarray(x, dtype=float), m + 1, increasing=True)


def _candidates_with(points, grid):
    """The grid, with the exact points put in place of nearby grid
    values (so no candidate sits within 1e-9 of a support point)."""
    keep = [g for g in grid if np.min(np.abs(points - g)) > 1e-9]
    return np.sort(np.concatenate([points, keep]))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_polynomial_weights_are_the_classical_design(m):
    support = _legendre_support(m)
    p = m + 1
    # the closed form is best: equivalence theorem on a fine grid
    g = _vander(support, m)
    m_star = g.T @ g / p
    fine = np.linspace(-1.0, 1.0, 20001)
    gf = _vander(fine, m)
    d_fine = np.einsum("ij,jk,ik->i", gf, np.linalg.inv(m_star), gf)
    assert np.max(d_fine) <= p * (1 + 1e-9)
    d_support = np.einsum("ij,jk,ik->i", g, np.linalg.inv(m_star), g)
    assert np.allclose(d_support, p, rtol=1e-9)
    # design_weights on 101 candidates that include the support
    cand = _candidates_with(support, np.linspace(-1.0, 1.0, 101))
    out = design_weights(_poly(m), np.ones(p), cand, tol=1e-10)
    w = out["weights"]
    at = np.isin(cand, support)
    assert np.all(np.abs(w[at] - 1.0 / p) < 1e-6)
    assert np.sum(w[~at]) < 1e-6
    assert out["max_variance"] <= p * (1 + 1e-10)
    assert out["efficiency_bound"] >= 1 / (1 + 1e-10)
    # its information per measurement is the closed form's
    assert np.allclose(out["fisher"], m_star, rtol=0, atol=1e-6)


def test_line_and_quadratic_on_a_plain_grid():
    """Without inserted points: the grid -1, -0.99, ..., 1 contains the
    ends and 0, the whole support for degrees 1 and 2."""
    x = np.linspace(-1.0, 1.0, 201)
    w1 = design_weights(_poly(1), [1.0, 1.0], x)["weights"]
    assert w1[0] == pytest.approx(0.5, abs=1e-9)
    assert w1[-1] == pytest.approx(0.5, abs=1e-9)
    w2 = design_weights(_poly(2), [1.0, 1.0, 1.0], x)["weights"]
    for i in (0, 100, 200):
        assert w2[i] == pytest.approx(1 / 3, abs=1e-9)


def _decay():
    return Model("decay", lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]),
                 ("A", "tau"), "exponential decay, test suite")


def test_decay_puts_half_at_zero_and_half_at_tau():
    """f = A exp(-t / tau): slopes g(t) = (e, A t e / tau^2) with
    e = exp(-t / tau). For two settings t1 < t2 the determinant of
    [g(t1), g(t2)] is (A / tau^2) exp(-(t1 + t2) / tau) (t2 - t1), which
    falls as t1 grows (so t1 = 0) and then is largest at t2 = tau. A
    two-point design for two parameters has equal weights (det M is
    w1 w2 times that determinant squared). The test then checks with
    the equivalence theorem that no third setting helps."""
    tau = 1.3
    t = np.sort(np.concatenate([[tau], np.linspace(0.0, 5.0, 51)]))
    t = np.unique(t)
    out = design_weights(_decay(), [2.0, tau], t, tol=1e-10, n_total=10)
    w = out["weights"]
    i0, it = 0, int(np.flatnonzero(t == tau)[0])
    assert w[i0] == pytest.approx(0.5, abs=1e-6)
    assert w[it] == pytest.approx(0.5, abs=1e-6)
    assert out["counts"][i0] == 5 and out["counts"][it] == 5
    assert out["counts_efficiency_low"] == pytest.approx(1.0, abs=1e-9)
    # equivalence theorem for the closed form, on a fine grid of [0, 5]
    def g(tt):
        e = np.exp(-tt / tau)
        return np.column_stack([e, 2.0 * tt * e / tau ** 2])
    m_star = 0.5 * (g(np.array([0.0])).T @ g(np.array([0.0]))
                    + g(np.array([tau])).T @ g(np.array([tau])))
    gf = g(np.linspace(0.0, 5.0, 50001))
    d = np.einsum("ij,jk,ik->i", gf, np.linalg.inv(m_star), gf)
    assert np.max(d) <= 2 * (1 + 1e-9)


def test_decay_with_background_three_points():
    """f = A exp(-t / tau) + B on [0, T], tau = 1: with settings 0, t and
    T the determinant of the slopes (rows (e, A t e, 1), e = exp(-t))
    is A [t e^-t (1 - e^-T) - T e^-T (1 - e^-t)], largest where its
    derivative e^-t [(1 - e^-T)(1 - t) - T e^-T] vanishes:
    t* = 1 - T e^-T / (1 - e^-T) (0.8428 for T = 3). A three-point
    design for three parameters has equal weights. The test checks with
    the equivalence theorem that this design is the best of all, and
    that design_weights finds it (README example 16)."""
    big_t = 3.0
    t_star = 1.0 - big_t * math.exp(-big_t) / (1.0 - math.exp(-big_t))
    assert t_star == pytest.approx(0.84281, abs=1e-5)
    model = _models()[0][0]
    theta = [1.0, 1.0, 0.2]
    support = np.array([0.0, t_star, big_t])

    def g(tt):
        e = np.exp(-tt)
        return np.column_stack([e, tt * e, np.ones_like(tt)])
    m_star = g(support).T @ g(support) / 3
    gf = g(np.linspace(0.0, big_t, 30001))
    d = np.einsum("ij,jk,ik->i", gf, np.linalg.inv(m_star), gf)
    assert np.max(d) <= 3 * (1 + 1e-9)
    cand = _candidates_with(support, np.linspace(0.0, big_t, 301))
    out = design_weights(model, theta, cand, sigmas=0.01, tol=1e-10)
    w = out["weights"]
    at = np.isin(cand, support)
    assert np.all(np.abs(w[at] - 1 / 3) < 1e-6)
    assert np.sum(w[~at]) < 1e-6


def _models():
    dec_b = Model("decay with background",
                  lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]) + th[2],
                  ("A", "tau", "B"), "exponential decay, test suite")
    peak = Model("peak", lambda th, x: th[0] * np.exp(
        -0.5 * ((x[:, 0] - th[1]) / th[2]) ** 2) + th[3],
        ("A", "c", "w", "B"), "Gaussian peak, test suite")
    surf = Model("surface", lambda th, x: th[0] + th[1] * x[:, 0]
                 + th[2] * x[:, 1] + th[3] * x[:, 0] * x[:, 1]
                 + th[4] * x[:, 0] ** 2 + th[5] * x[:, 1] ** 2,
                 tuple("abcdef"), "quadratic surface, test suite")
    g = np.linspace(-1.0, 1.0, 15)
    grid = np.array([[u, v] for u in g for v in g])
    return [(dec_b, [1.0, 1.0, 0.2], np.linspace(0.0, 3.0, 301)),
            (peak, [1.0, 0.2, 0.5, 0.1], np.linspace(-3.0, 3.0, 241)),
            (surf, np.ones(6), grid)]


def test_identities_at_every_result():
    """sum_i w_i d_i = p (a trace identity), max d_i <= p (1 + tol),
    the bound is p / max d_i, and the weights are a distribution; d_i
    recomputed here from the model's slopes."""
    for model, theta, cand in _models():
        out = design_weights(model, theta, cand, sigmas=0.01, tol=1e-7)
        w = out["weights"]
        p = model.n_params
        assert np.all(w >= 0) and math.isclose(np.sum(w), 1.0,
                                                rel_tol=1e-12)
        g = model.jacobian(theta, cand) / 0.01
        m = g.T @ (w[:, None] * g)
        assert np.allclose(out["fisher"], m, rtol=1e-10,
                           atol=1e-12 * np.max(np.abs(m)))
        d = np.einsum("ij,jk,ik->i", g, np.linalg.inv(m), g)
        assert np.sum(w * d) == pytest.approx(p, rel=1e-8)
        assert np.max(d) <= p * (1 + 1e-7) * (1 + 1e-8)
        assert out["efficiency_bound"] == pytest.approx(p / np.max(d),
                                                        rel=1e-8)
        # sigma and cov are those of M^-1
        cov = np.linalg.inv(m)
        scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        assert np.all(np.abs(out["cov"] - cov) < 1e-8 * scale)


def test_bounds_hold_against_the_exact_optimum():
    """Quadratic on 21 candidates: the best weights are 1/3 at -1, 0, 1
    (see above), so the exact D-efficiency of any design is known. For
    300 random designs, and for the picks of `design`, the bounds of
    `design_efficiency` contain it."""
    x = np.linspace(-1.0, 1.0, 21)
    g = _vander(x, 2)
    star = g[[0, 10, 20]]
    ld_star = np.linalg.slogdet(star.T @ star / 3)[1]
    rng = np.random.default_rng(12)
    for trial in range(300):
        counts = rng.integers(0, 4, size=21) * (rng.random(21) < 0.3)
        if counts.sum() == 0:
            continue                    # refused; see test_refusals
        if np.sum(counts > 0) < 3:      # fewer settings than parameters
            out = design_efficiency(_poly(2), np.ones(3), x, counts)
            assert out["efficiency_low"] == 0.0
            assert out["efficiency_high"] == 0.0
            assert out["identifiable"] is False
            continue
        m = g.T @ ((counts / counts.sum())[:, None] * g)
        exact = math.exp((np.linalg.slogdet(m)[1] - ld_star) / 3)
        out = design_efficiency(_poly(2), np.ones(3), x, counts)
        assert out["efficiency_low"] <= exact * (1 + 1e-9)
        assert exact <= out["efficiency_high"] * (1 + 1e-9)
        assert out["efficiency_low"] >= 3 / out["max_variance"] - 1e-12
        assert out["n_total"] == counts.sum()
    pick = design(_poly(2), np.ones(3), x, 5)["indices"]
    counts = np.bincount(pick, minlength=21)
    m = g.T @ ((counts / 5)[:, None] * g)
    exact = math.exp((np.linalg.slogdet(m)[1] - ld_star) / 3)
    out = design_efficiency(_poly(2), np.ones(3), x, counts)
    assert out["efficiency_low"] <= exact * (1 + 1e-9) \
        <= out["efficiency_high"] * (1 + 1e-9) ** 2


def _best_multiset_logdet(g, n_total):
    best = -np.inf
    for comb in itertools.combinations_with_replacement(
            range(g.shape[0]), n_total):
        rows = g[list(comb)]
        sign, val = np.linalg.slogdet(rows.T @ rows)
        if sign > 0 and val > best:
            best = val
    return best


def test_counts_against_every_multiset():
    """24 small problems (4 models, n_total from p to p + 5): the counts
    sum to n_total, their efficiency bounds contain the efficiency of
    the best multiset found by checking every one, and in all 24 cases
    the counts are as good as that best multiset."""
    dec_b = _models()[0][0]
    cases = [(_poly(1), np.ones(2), np.linspace(-1.0, 1.0, 5)),
             (_poly(2), np.ones(3), np.linspace(-1.0, 1.0, 7)),
             (dec_b, [1.0, 1.0, 0.2], np.linspace(0.0, 3.0, 8)),
             (_poly(3), np.ones(4), np.linspace(-1.0, 1.0, 7))]
    for model, theta, cand in cases:
        g = model.jacobian(theta, cand)
        p = model.n_params
        for n_total in range(p, p + 6):
            out = design_weights(model, theta, cand, n_total=n_total)
            c = out["counts"]
            assert c.dtype.kind == "i" and c.sum() == n_total
            assert np.all(c >= 0)
            ld = np.linalg.slogdet(g.T @ (c[:, None] * g))[1]
            best = _best_multiset_logdet(g, n_total)
            assert ld >= best - 1e-9 * abs(best) - 1e-12
            assert out["counts_efficiency_low"] \
                <= out["counts_efficiency_high"] <= 1.0


def test_repeats_for_counts_all_measurements():
    """A line on [-1, 1] with reading error 0.05: the best weights are
    1/2 at each end, M = identity / 0.05^2 per measurement, so one
    measurement gives error bars 0.05 on both parameters and N give
    0.05 / sqrt(N). repeats_for then returns N = ceil((0.05/0.01)^2) =
    25 measurements in all; 26 of them, 13 at each end, give exactly
    0.05 / sqrt(26). (Exact slopes: the numerical slope of a parameter
    at 0 is good to about 1e-7 only.)"""
    x = np.linspace(-1.0, 1.0, 41)
    line = Model("line", lambda th, x: th[0] + th[1] * x[:, 0],
                 ("c0", "c1"), "polynomial, test suite",
                 jac=lambda th, x: np.column_stack([np.ones(len(x)),
                                                    x[:, 0]]))
    out = design_weights(line, [0.0, 2.0], x, sigmas=0.05)
    for k in ("c0", "c1"):
        assert out["sigma"][k] == pytest.approx(0.05, rel=1e-9)
    r, pred = repeats_for(0.01, out)
    assert r == 25
    c = design_weights(line, [0.0, 2.0], x, sigmas=0.05,
                       n_total=26)["counts"]
    assert c[0] == 13 and c[-1] == 13 and c.sum() == 26
    plan = information(line, [0.0, 2.0], np.repeat(x, c), sigmas=0.05)
    for k in ("c0", "c1"):
        assert plan["sigma"][k] == pytest.approx(0.05 / math.sqrt(26),
                                                 rel=1e-9)
    eff = design_efficiency(line, [0.0, 2.0], x, c, sigmas=0.05)
    assert eff["sigma"]["c1"] == pytest.approx(plan["sigma"]["c1"],
                                               rel=1e-12)


def test_units_and_reading_errors():
    """The weights do not depend on the units of the parameters or of
    the settings, nor on a common factor in the reading errors; with
    exactly p candidates the weights are 1/p whatever their reading
    errors (det M = w_1 ... w_p det(G)^2, largest at equal weights)."""
    dec_b = _models()[0][0]
    t = np.linspace(0.0, 3.0, 61)
    s = design_weights(dec_b, [1.0, 1.0, 0.2], t, sigmas=0.01)["weights"]
    ns = design_weights(dec_b, [1.0, 1e-9, 0.2], t * 1e-9,
                        sigmas=0.01)["weights"]
    big = design_weights(dec_b, [1.0, 1.0, 0.2], t, sigmas=7.0)["weights"]
    assert np.max(np.abs(s - ns)) < 1e-6
    assert np.max(np.abs(s - big)) < 1e-6
    w = design_weights(_poly(2), np.ones(3), [-1.0, 0.3, 1.0],
                       sigmas=[0.1, 3.0, 0.02])["weights"]
    assert np.allclose(w, 1 / 3, rtol=0, atol=1e-9)


def test_design_picks_are_rated():
    """README example 9: the exchange triple is rated at least as high
    as the greedy one (its determinant is larger), and both bounds are
    consistent with their own variance functions."""
    dec_b = _models()[0][0]
    cand = np.linspace(0.0, 3.0, 31)
    effs = []
    for exchange in (False, True):
        pick = design(dec_b, [1.0, 1.0, 0.2], cand, 3, sigmas=0.01,
                      exchange=exchange)
        counts = np.bincount(pick["indices"], minlength=31)
        out = design_efficiency(dec_b, [1.0, 1.0, 0.2], cand, counts,
                                sigmas=0.01)
        assert out["efficiency_low"] <= out["efficiency_high"] <= 1.0
        assert out["fisher"] == pytest.approx(pick["fisher"], rel=1e-9)
        effs.append(out["efficiency_high"])
    assert effs[1] >= effs[0]


def test_refusals():
    x = np.linspace(-1.0, 1.0, 11)
    m = _poly(1)
    deg = Model("sum only", lambda th, x: (th[0] + th[1]) * x[:, 0],
                ("a", "b"), "degenerate test model")
    with pytest.raises(ValueError, match="identifiable"):
        design_weights(deg, [1.0, 1.0], x)
    with pytest.raises(ValueError, match="identifiable"):
        design_weights(m, [1.0, 1.0], [0.5])
    for bad in (0.0, -1e-3, np.nan, True, "x"):
        with pytest.raises(ValueError, match="tol"):
            design_weights(m, [1.0, 1.0], x, tol=bad)
    for bad in (0, 2.5, True):
        with pytest.raises(ValueError, match="max_iter"):
            design_weights(m, [1.0, 1.0], x, max_iter=bad)
    for bad in (1, 2.5, True, np.inf):
        with pytest.raises(ValueError, match="n_total"):
            design_weights(m, [1.0, 1.0], x, n_total=bad)
    for bad in (np.ones(3), -np.ones(11), np.full(11, 0.5),
                np.zeros(11), np.ones(11, dtype=bool)):
        with pytest.raises(ValueError, match="counts"):
            design_efficiency(m, [1.0, 1.0], x, bad)
    cubic = _legendre_support(3)
    with pytest.raises(RuntimeError, match="at least"):
        design_weights(_poly(3), np.ones(4),
                       _candidates_with(cubic, np.linspace(-1, 1, 101)),
                       max_iter=1)

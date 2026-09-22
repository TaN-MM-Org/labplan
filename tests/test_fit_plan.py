"""Core anchors: on a linear model the fit covariance equals the
textbook closed form sigma^2 (X^T X)^-1 (to relative 1e-6) and the
planner promises the same matrix; noiseless fits recover the truth;
seeded Monte Carlo matches the reported error bars; an exactly
degenerate model (duplicated parameter) is refused by planner and fit
alike via the scale-invariant rank test; the rank-one determinant
identity holds, and the greedy design never loses to a random subset
and finds the best pair by exhaustion; the repeat law 1/sqrt(r) is
asserted by tiling the design; and the finite-difference slopes are
unit-invariant for tiny parameters and exact near zero."""
import numpy as np
import pytest

from labplan import (Model, design, fit, information, repeats_for)

REF = "line model, unit test suite (y = a x + b)"


def _line():
    return Model("line", lambda th, x: th[0] * x[:, 0] + th[1],
                 ("slope", "intercept"), REF)


X = np.linspace(0.5, 5.0, 12)[:, None]
TRUTH = np.array([2.5, -0.7])


def test_linear_closed_form_covariance_and_promise_kept():
    m = _line()
    y = m.predict(TRUTH, X)
    sigma = 0.05
    res = fit(m, X, y, [1.0, 0.0], sigmas=sigma)
    assert abs(res.values["slope"] - TRUTH[0]) < 1e-8
    assert abs(res.values["intercept"] - TRUTH[1]) < 1e-8
    assert res.chi2 < 1e-12 and res.chi2_dof == 10
    Xd = np.column_stack([X[:, 0], np.ones(X.shape[0])])
    want = sigma ** 2 * np.linalg.inv(Xd.T @ Xd)
    assert np.allclose(res.cov, want, rtol=1e-6, atol=1e-15)
    plan = information(m, TRUTH, X, sigmas=sigma)
    assert plan["identifiable"]
    for name in ("slope", "intercept"):
        assert abs(plan["sigma"][name] - res.sigma[name]) \
            < 1e-6 * res.sigma[name]


def test_error_bars_agree_with_monte_carlo():
    m = _line()
    y0 = m.predict(TRUTH, X)
    sigma = 0.2
    rng = np.random.default_rng(5)
    draws = []
    reported = None
    for _ in range(400):
        res = fit(m, X, y0 + sigma * rng.standard_normal(y0.size),
                  [1.0, 0.0], sigmas=sigma)
        draws.append(res.theta)
        reported = res.sigma
    emp = np.std(np.array(draws), axis=0, ddof=1)
    assert np.allclose(emp, [reported["slope"],
                             reported["intercept"]], rtol=0.15)


def test_nonlinear_model_recovered():
    """An exponential decay -- genuinely nonlinear -- from a detuned
    start."""
    m = Model("decay", lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]),
              ("amplitude", "tau"), "exponential decay, test suite")
    truth = np.array([3.0, 1.7])
    x = np.linspace(0.1, 6.0, 15)[:, None]
    y = m.predict(truth, x)
    res = fit(m, x, y, [1.0, 3.0], sigmas=0.01)
    assert abs(res.values["amplitude"] - 3.0) < 1e-6
    assert abs(res.values["tau"] - 1.7) < 1e-6


def test_exact_degeneracy_refused_everywhere():
    """A model where two parameters enter only through their sum:
    exactly rank-deficient, whatever the design -- reported by the
    planner, refused by the fit and the design tool."""
    m = Model("degenerate",
              lambda th, x: (th[0] + th[1]) * x[:, 0],
              ("a", "b"), "degenerate-by-construction test model")
    plan = information(m, [1.0, 2.0], X, sigmas=0.1)
    assert not plan["identifiable"]
    fi = plan["fisher"]
    d = np.sqrt(np.diag(fi))
    sv = np.linalg.svd(fi / np.outer(d, d), compute_uv=False)
    assert sv[-1] < 1e-10 * sv[0]
    y = 3.0 * X[:, 0]
    with pytest.raises(ValueError, match="cannot tell"):
        fit(m, X, y, [1.0, 2.0], sigmas=0.1)
    with pytest.raises(ValueError, match="identifiable"):
        design(m, [1.0, 2.0], X, 3, sigmas=0.1)


def test_rank_one_determinant_identity():
    rng = np.random.default_rng(3)
    for _ in range(20):
        a = rng.standard_normal((3, 3))
        a = a @ a.T + 0.5 * np.eye(3)
        g = rng.standard_normal(3)
        lhs = np.linalg.det(a + np.outer(g, g))
        rhs = np.linalg.det(a) * (1.0 + g @ np.linalg.solve(a, g))
        assert abs(lhs - rhs) < 1e-9 * abs(rhs)
        assert lhs >= np.linalg.det(a) - 1e-12 * abs(lhs)


def test_design_greedy_invariant_and_quality():
    m = _line()
    cand = np.linspace(0.2, 6.0, 15)[:, None]
    out = design(m, TRUTH, cand, 5, sigmas=0.1)
    idx = out["indices"]
    assert len(idx) == 5 and len(set(idx)) == 5
    assert out["sigma"] is not None

    def logdet_of(subset):
        info = information(m, TRUTH, cand[list(subset)], sigmas=0.1)
        s, d = np.linalg.slogdet(info["fisher"])
        return d if s > 0 else -np.inf

    best = logdet_of(idx)
    rng = np.random.default_rng(11)
    for _ in range(30):
        assert best >= logdet_of(rng.choice(15, 5, replace=False)) \
            - 1e-9
    # for the line model the most informative pair straddles the
    # candidate range: the two extreme settings maximize the
    # determinant (a classical two-point-design fact, checked here by
    # exhaustion over all pairs)
    pair = design(m, TRUTH, cand, 2, sigmas=0.1)["indices"]
    dets = {(i, j): logdet_of([i, j]) for i in range(15)
            for j in range(i + 1, 15)}
    assert max(dets, key=dets.get) == tuple(sorted(pair))


def test_repeats_law_exact():
    m = _line()
    plan = information(m, TRUTH, X, sigmas=0.1)
    r = 9
    tiled = information(m, TRUTH, np.tile(X, (r, 1)), sigmas=0.1)
    for name in ("slope", "intercept"):
        assert abs(tiled["sigma"][name]
                   - plan["sigma"][name] / 3.0) \
            < 1e-9 * plan["sigma"][name]
    target = 0.6 * plan["sigma"]["slope"]
    r_need, pred = repeats_for({"slope": target}, plan)
    assert pred["slope"] <= target
    if r_need > 1:
        prev = {k: v * np.sqrt(r_need) / np.sqrt(r_need - 1)
                for k, v in pred.items()}
        assert prev["slope"] > target


def test_small_parameters_unit_invariant():
    """The same RC charging curve written with tau in seconds (1e-9)
    and in nanoseconds (1.0) must give the same planned error bars
    and the same noiseless fit. Before 0.1.1 the finite-difference
    step had an absolute floor of 1e-9, as large as tau itself in
    seconds: the plan changed with the units, the noiseless fit
    missed the truth, and a model dividing by such a parameter was
    refused as non-finite."""
    f = lambda th, x: th[0] * (1.0 - np.exp(-x[:, 0] / th[1]))
    ms = Model("rc, seconds", f, ("V0", "tau"), "RC charging, test")
    mn = Model("rc, ns", f, ("V0", "tau"), "RC charging, test")
    t = np.linspace(0.2, 5.0, 12)
    ps = information(ms, [1.0, 1e-9], t * 1e-9, sigmas=0.01)["sigma"]
    pn = information(mn, [1.0, 1.0], t, sigmas=0.01)["sigma"]
    assert abs(ps["V0"] - pn["V0"]) < 1e-6 * pn["V0"]
    assert abs(ps["tau"] * 1e9 - pn["tau"]) < 1e-6 * pn["tau"]
    y = mn.predict([1.0, 1.0], t)
    res = fit(ms, t * 1e-9, y, [0.8, 1.3e-9], sigmas=0.01)
    assert abs(res.values["V0"] - 1.0) < 1e-6
    assert abs(res.values["tau"] - 1e-9) < 1e-6 * 1e-9
    inv = Model("ratio", lambda th, x: th[0] * x[:, 0] / th[1],
                ("a", "c"), "ratio model, test suite")
    jac = inv.jacobian([1.0, 1e-9], t)
    assert np.all(np.isfinite(jac))
    assert np.allclose(jac[:, 1], -t / 1e-18, rtol=1e-6)


def test_near_zero_parameter_keeps_a_resolvable_step():
    """A parameter close to zero on its natural scale -- a slope of
    1e-12 next to an intercept of 1 -- must still get exact slopes:
    a purely relative step (1e-18) is lost in floating-point
    rounding. The line's slopes are known exactly (x and 1), and a
    noiseless fit of a flat line (true slope 0) must return the
    textbook covariance sigma^2 (X^T X)^-1."""
    m = _line()
    exact = np.column_stack([X[:, 0], np.ones(X.shape[0])])
    for slope in (1e-12, 1e-300, 0.0):
        jac = m.jacobian([slope, 1.0], X)
        assert np.allclose(jac, exact, rtol=1e-6, atol=0.0)
    sigma = 0.05
    res = fit(m, X, m.predict([0.0, 0.5], X), [1.0, 0.0],
              sigmas=sigma)
    assert abs(res.values["slope"]) < 1e-8
    assert abs(res.values["intercept"] - 0.5) < 1e-8
    want = sigma ** 2 * np.linalg.inv(exact.T @ exact)
    assert np.allclose(res.cov, want, rtol=1e-6, atol=1e-15)


def test_model_and_input_refusals():
    with pytest.raises(ValueError, match="reference"):
        Model("m", lambda th, x: x[:, 0], ("a",), "short")
    with pytest.raises(ValueError, match="callable"):
        Model("m", None, ("a",), REF)
    with pytest.raises(ValueError, match="unique"):
        Model("m", lambda th, x: x[:, 0], ("a", "a"), REF)
    m = _line()
    with pytest.raises(ValueError, match="determine"):
        fit(m, X[:1], [1.0], [1.0, 0.0], sigmas=0.1)
    with pytest.raises(ValueError, match="positive"):
        fit(m, X, m.predict(TRUTH, X), [1.0, 0.0], sigmas=-1.0)
    with pytest.raises(ValueError, match="n_pick"):
        design(m, TRUTH, X, 1, sigmas=0.1)
    with pytest.raises(ValueError, match="finite"):
        m.predict([np.inf, 0.0], X)

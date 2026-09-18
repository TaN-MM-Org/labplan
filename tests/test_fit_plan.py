"""Core anchors: on a linear model the fit covariance equals the
textbook closed form sigma^2 (X^T X)^-1 exactly and the planner
promises the same matrix; noiseless fits recover the truth; seeded
Monte Carlo matches the reported error bars; an exactly degenerate
model (duplicated parameter) is refused by planner and fit alike via
the scale-invariant rank test; the greedy design obeys the exact
rank-one determinant identity, reproduces its own rule, and never
loses to a random subset; and the repeat law 1/sqrt(r) is asserted by
tiling the design."""
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

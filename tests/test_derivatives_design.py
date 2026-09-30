"""0.2.0 anchors for slopes and design.

- A parameter that is exactly zero no longer gets a blind absolute
  step: the Gaussian-pulse centre t0 = 0 written in seconds (width
  1 ns) gets slopes that match the exact derivative, planned error
  bars that match the exact information matrix, and the same answer
  as the model written in nanoseconds. In 0.1.1 the slope of t0 was
  31 % wrong and its planned error bar 26 % too large.
- Where the 0.1.0 fallback step was already accurate, the refined
  slopes are bit-identical to it, recomputed in the test by hand.
- An exact Jacobian supplied by the user is used as given, and agrees
  with the finite-difference one.
- design(exchange=True) reaches the best subset found by checking
  every subset on 24 seeded problems where the greedy picks miss it
  6 times, and no single swap improves its result.
- A `design` result is accepted by `repeats_for`; its error bars equal
  those `information` gives for the chosen settings.
"""
import itertools

import numpy as np
import pytest

from labplan import Model, design, fit, information, repeats_for

W = 1e-9                                  # pulse width, seconds
T = np.linspace(-3e-9, 3e-9, 25)          # sample times, seconds


def _pulse(th, x):
    return th[0] * np.exp(-0.5 * ((x[:, 0] - th[1]) / th[2]) ** 2)


def _pulse_exact_jac(th, x):
    u = (x[:, 0] - th[1]) / th[2]
    e = np.exp(-0.5 * u * u)
    return np.column_stack([e, th[0] * e * u / th[2],
                            th[0] * e * u * u / th[2]])


def _pulse_model(**kw):
    return Model("pulse", _pulse, ("A", "t0", "w"),
                 "Gaussian pulse, test suite", **kw)


def test_zero_parameter_slopes_are_refined():
    """t0 = 0 in seconds: the 0.1.0 fallback step (1e-9 s) equals the
    pulse width. The refined slopes must match the exact derivative,
    the planned error bars the exact (J^T W J)^-1, and the plan must
    not depend on writing time in seconds or nanoseconds."""
    m = _pulse_model()
    th = [1.0, 0.0, W]
    exact = _pulse_exact_jac(np.array(th), T[:, None])
    jac = m.jacobian(th, T)
    col_err = np.max(np.abs(jac - exact), axis=0) \
        / np.max(np.abs(exact), axis=0)
    assert np.all(col_err < 1e-5)
    sigma = 0.01
    want = np.sqrt(np.diag(np.linalg.inv(
        (exact / sigma).T @ (exact / sigma))))
    got = information(m, th, T, sigmas=sigma)["sigma"]
    for k, name in enumerate(("A", "t0", "w")):
        assert abs(got[name] - want[k]) < 1e-5 * want[k]
    ns = information(m, [1.0, 0.0, 1.0], T * 1e9, sigmas=sigma)["sigma"]
    assert abs(got["t0"] * 1e9 - ns["t0"]) < 1e-5 * ns["t0"]
    # a noiseless fit in seconds ends at t0 ~ 0 (the fallback is
    # used again for the final covariance) and reports the exact
    # error bars
    y = _pulse(np.array(th), T[:, None])
    res = fit(m, T, y, [0.9, 0.2e-9, 1.2e-9], sigmas=sigma)
    assert abs(res.values["t0"]) < 1e-6 * W
    assert abs(res.sigma["t0"] - want[1]) < 1e-5 * want[1]


def test_accurate_fallback_is_unchanged():
    """Where the 0.1.0 step 1e-6 * max(|theta|, 1e-3) = 1e-9 was
    already accurate, the result is bit-identical to that central
    difference, recomputed here by hand."""
    x = np.linspace(0.5, 5.0, 12)
    cases = [
        (lambda th, x: th[0] * x[:, 0] + th[1], [0.0, 1.0]),
        (lambda th, x: th[0] * x[:, 0] + th[1], [1e-300, 1.0]),
        (lambda th, x: th[0] * np.sin(x[:, 0] + th[1]), [2.0, 0.0]),
        (lambda th, x: np.exp(th[0] * x[:, 0]) + th[1], [0.0, 1.0]),
    ]
    for f, th in cases:
        m = Model("case", f, ("a", "b"), "fallback test model")
        jac = m.jacobian(th, x)
        for j in range(2):
            if th[j] != 0.0 and th[j] > 1e-3:
                continue
            h = 1e-9
            tp, tm = np.array(th, float), np.array(th, float)
            tp[j] += h
            tm[j] -= h
            fd = (f(tp, x[:, None]) - f(tm, x[:, None])) / (2 * h)
            assert np.array_equal(jac[:, j], fd)


def test_noisy_model_is_not_made_worse():
    """A model whose own evaluation loses digits (adding and
    subtracting 1e3): the refinement must not end on a noisier slope
    than the 0.1.0 step gave."""
    x = np.linspace(0.5, 5.0, 12)
    f = lambda th, x: ((th[0] * x[:, 0] + 1e3) - 1e3) + th[1]
    m = Model("cancel", f, ("a", "b"), "cancellation test model")
    h = 1e-9
    old = (f(np.array([h, 1.0]), x[:, None])
           - f(np.array([-h, 1.0]), x[:, None])) / (2 * h)
    new = m.jacobian([0.0, 1.0], x)[:, 0]
    assert np.max(np.abs(new - x)) <= np.max(np.abs(old - x))


def test_exact_jacobian_is_used_and_checked():
    m = _pulse_model(jac=_pulse_exact_jac)
    th = [1.0, 0.0, W]
    exact = _pulse_exact_jac(np.array(th), T[:, None])
    assert np.array_equal(m.jacobian(th, T), exact)
    num = m.numeric_jacobian(th, T)
    assert np.allclose(num, exact, rtol=0,
                       atol=1e-5 * np.max(np.abs(exact)))
    sigma = 0.01
    want = np.linalg.inv((exact / sigma).T @ (exact / sigma))
    plan = information(m, th, T, sigmas=sigma)
    d = np.sqrt(np.diag(want))
    # entries compared on the scale sqrt(C_ii C_jj): the pulse makes
    # some of them exactly zero, where only rounding is left
    assert np.all(np.abs(plan["cov"] - want) <= 1e-9 * np.outer(d, d))
    y = _pulse(np.array(th), T[:, None])
    res = fit(m, T, y, [0.9, 0.2e-9, 1.2e-9], sigmas=sigma)
    assert abs(res.values["A"] - 1.0) < 1e-9
    assert abs(res.values["w"] - W) < 1e-9 * W
    one = Model("slope only", lambda th, x: th[0] * x[:, 0],
                ("a",), "one-parameter test model",
                jac=lambda th, x: x[:, 0])          # (n,) is accepted
    assert one.jacobian([2.0], T).shape == (T.size, 1)
    with pytest.raises(ValueError, match="callable"):
        _pulse_model(jac="not a function")
    bad = _pulse_model(jac=lambda th, x: np.ones((x.shape[0], 2)))
    with pytest.raises(ValueError, match="array"):
        bad.jacobian(th, T)
    nan = _pulse_model(jac=lambda th, x: np.full((x.shape[0], 3),
                                                 np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        nan.jacobian(th, T)


def _logdet(rows, idx):
    s, d = np.linalg.slogdet(rows[list(idx)].T @ rows[list(idx)])
    return d if s > 0 else -np.inf


def test_exchange_reaches_exhaustive_optimum():
    m = Model("decay + offset",
              lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]) + th[2],
              ("A", "tau", "c"), "decay with offset, test suite")
    th = [1.0, 1.0, 0.2]
    rng = np.random.default_rng(0)
    greedy_missed = 0
    for _ in range(8):
        cand = np.sort(rng.uniform(0.05, 5.0, 12))
        rows = m.jacobian(th, cand) / 0.1
        for k in (3, 4, 5):
            g = design(m, th, cand, k, sigmas=0.1)["indices"]
            out = design(m, th, cand, k, sigmas=0.1, exchange=True)
            e = out["indices"]
            assert len(set(e)) == k
            best = max(_logdet(rows, s)
                       for s in itertools.combinations(range(12), k))
            lg, le = _logdet(rows, g), _logdet(rows, e)
            greedy_missed += lg < best - 1e-9
            assert le >= lg - 1e-12
            assert abs(le - best) < 1e-9
            # no single swap improves the exchange result
            for a in range(k):
                for j in set(range(12)) - set(e):
                    trial = list(e)
                    trial[a] = j
                    assert _logdet(rows, trial) <= le + 1e-9
            # the returned matrix belongs to the returned indices
            assert np.allclose(out["fisher"], rows[e].T @ rows[e],
                               rtol=1e-9)
    assert greedy_missed == 6


def test_readme_example_9_exchange_is_best_triple():
    """README example 9: 3 of 31 settings 0.0, 0.1, ..., 3.0 for a
    decay with background. The exchange result is the best of all
    4495 triples; the greedy one is not."""
    m = Model("decay with background",
              lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]) + th[2],
              ("amplitude", "tau", "background"),
              "illustrative exponential decay with background")
    guess = [1.0, 1.0, 0.2]
    cand = np.linspace(0.0, 3.0, 31)
    rows = m.jacobian(guess, cand) / 0.01
    best = max(_logdet(rows, s)
               for s in itertools.combinations(range(31), 3))
    g = design(m, guess, cand, 3, sigmas=0.01)["indices"]
    e = design(m, guess, cand, 3, sigmas=0.01, exchange=True)["indices"]
    assert abs(_logdet(rows, e) - best) < 1e-9
    assert _logdet(rows, g) < best - 1e-3
    assert np.allclose(np.sort(cand[e]), [0.0, 0.8, 3.0])


def test_design_output_feeds_repeats_for():
    """0.1.1 refused a `design` result in `repeats_for` ("not
    identifiable") because it had no `identifiable` key."""
    m = Model("line", lambda th, x: th[0] * x[:, 0] + th[1],
              ("gain", "offset"), "illustrative straight-line model")
    cand = np.linspace(0.5, 5.0, 12)
    out = design(m, [2.0, 0.0], cand, 4, sigmas=0.05)
    assert out["identifiable"]
    ref = information(m, [2.0, 0.0], cand[out["indices"]], sigmas=0.05)
    for name in ("gain", "offset"):
        assert abs(out["sigma"][name] - ref["sigma"][name]) \
            < 1e-9 * ref["sigma"][name]
    assert np.allclose(out["cov"], ref["cov"], rtol=1e-9, atol=0)
    r_out, pred_out = repeats_for(0.002, out)
    r_ref, pred_ref = repeats_for(0.002, ref)
    assert r_out == r_ref
    for name in ("gain", "offset"):
        assert abs(pred_out[name] - pred_ref[name]) \
            < 1e-9 * pred_ref[name]

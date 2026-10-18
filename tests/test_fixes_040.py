"""0.4.0 anchors for one wrong refusal, two clearer refusals and the
faster `design`.

- A parameter of 1e-9 or less whose effect on the readings is small
  next to their largest value (a 1 ns decay of 10 counts on a
  background of 1000 counts, in seconds) got the 0.1.0 fallback step
  1e-9, which reaches zero: the model divided by zero and
  `information`, `design` and `fit` refused the slopes as
  "non-finite". The step is now cut until the model is finite. The
  slopes are checked against the exact derivatives, the planned
  error bars against the exact (J^T W J)^-1 and the fit against the
  same fit with exact slopes. These tests fail on 0.3.0.
- `Model.numeric_jacobian(rel_step=0)` raised UnboundLocalError and
  `coverage_exact(29.9, 0.1)` answered for n = 29; both now refuse.
- `design` computes the candidate determinants in batches. Its picks
  and its information matrix are compared with the 0.3.0 loops kept
  below as the reference (with ==, until 0.5.0; with the stated
  tolerances of `_assert_same_design` since then).
"""
import math

import numpy as np
import pytest

from labplan import Model, coverage_exact, design, fit, information

REF = "exponential decay on a constant background, test suite"


def _decay(th, x):
    return th[0] * np.exp(-x[:, 0] / th[1]) + th[2]


def _decay_jac(th, x):
    t = x[:, 0]
    e = np.exp(-t / th[1])
    return np.column_stack([e, th[0] * t * e / th[1] ** 2,
                            np.ones_like(t)])


T = np.linspace(0.0, 5e-9, 41)          # seconds, starting at t = 0
TH = np.array([10.0, 1e-9, 1000.0])     # 10 counts, 1 ns, 1000 counts


def test_fallback_step_that_reaches_zero_is_cut():
    m = Model("decay on background", _decay, ("A", "tau", "B"), REF)
    exact = _decay_jac(TH, T[:, None])
    # 0.3.0: ValueError "the forward function returned non-finite
    # predictions" (tau - 1e-9 = 0 and 0/0 at t = 0)
    jac = m.jacobian(TH, T)
    col_err = np.max(np.abs(jac - exact), axis=0) \
        / np.max(np.abs(exact), axis=0)
    assert np.all(col_err < 1e-5)
    want = np.sqrt(np.diag(np.linalg.inv(exact.T @ exact)))
    got = information(m, TH, T, sigmas=1.0)["sigma"]
    for k, name in enumerate(("A", "tau", "B")):
        assert abs(got[name] - want[k]) < 1e-5 * want[k]
    # a fit started at the same guess equals the fit with exact slopes
    rng = np.random.default_rng(3)
    y = m.predict(TH, T) + rng.standard_normal(T.size)
    res = fit(m, T, y, TH, sigmas=1.0)
    ref = fit(Model("decay on background", _decay, ("A", "tau", "B"),
                    REF, jac=_decay_jac), T, y, TH, sigmas=1.0)
    err = np.sqrt(np.diag(ref.cov))
    assert np.all(np.abs(res.theta - ref.theta) < 1e-4 * err)
    assert np.allclose(np.sqrt(np.diag(res.cov)), err, rtol=1e-5)
    # the design tool works there too, and picks t = 0 among others
    pick = design(m, TH, T, 3, sigmas=1.0)
    assert pick["identifiable"] and 0 in pick["indices"]


def test_fallback_cut_does_not_touch_working_steps():
    """Where the 0.1.0 fallback step gives finite predictions it is
    used exactly as before: same slopes, bit for bit, as the central
    difference with step 1e-9 computed here by hand (tau = 2e-9 on the
    same background: the step does not reach zero)."""
    m = Model("decay on background", _decay, ("A", "tau", "B"), REF)
    th = np.array([10.0, 2e-9, 1000.0])
    t = np.linspace(0.0, 1e-8, 41)
    jac = m.jacobian(th, t)
    exact = _decay_jac(th, t[:, None])
    assert np.max(np.abs(jac[:, 1] - exact[:, 1])) \
        < 1e-5 * np.max(np.abs(exact[:, 1]))
    # A and B are >= 1e-3: relative steps, unchanged by construction
    for j in (0, 2):
        h = 1e-6 * abs(th[j])
        tp, tm = th.copy(), th.copy()
        tp[j] += h
        tm[j] -= h
        fd = (_decay(tp, t[:, None]) - _decay(tm, t[:, None])) / (2 * h)
        assert np.array_equal(jac[:, j], fd)


def test_a_model_that_is_never_finite_is_still_refused():
    m = Model("log of a parameter",
              lambda th, x: np.log(th[0]) * x[:, 0] + 1e6,
              ("a",), "refusal test model, test suite")
    with pytest.raises(ValueError, match="non-finite"):
        m.jacobian([0.0], [1.0, 2.0])


@pytest.mark.parametrize("step", [0.0, -1e-6, float("nan"), float("inf"),
                                  "abc", True])
def test_rel_step_must_be_positive(step):
    m = Model("line", lambda th, x: th[0] * x[:, 0] + th[1],
              ("gain", "offset"), "illustrative straight-line model")
    # 0.3.0: UnboundLocalError for 0 and negative steps
    with pytest.raises(ValueError, match="rel_step"):
        m.numeric_jacobian([2.0, 1.0], [1.0, 2.0, 3.0], rel_step=step)


def test_coverage_exact_needs_a_whole_n():
    # 0.3.0 returned 0.9 (the answer for n = 29)
    with pytest.raises(ValueError, match="whole number"):
        coverage_exact(29.9, 0.1)
    with pytest.raises(ValueError, match="whole number"):
        coverage_exact(True, 0.1)
    assert coverage_exact(29.0, 0.1) == coverage_exact(29, 0.1) == 0.9


# ------------------------------------------------------------------
# design: the 0.3.0 loops, kept verbatim as the reference


def _old_logdet(f):
    sign, val = np.linalg.slogdet(f)
    return float(val) if sign > 0 else -np.inf


def _old_exchange(rs, chosen, max_passes=1000):
    chosen = list(chosen)
    n = rs.shape[0]
    f = rs[chosen].T @ rs[chosen]
    cur = _old_logdet(f)
    for _ in range(max_passes):
        best, swap = cur, None
        for a, i in enumerate(chosen):
            fa = f - np.outer(rs[i], rs[i])
            for j in range(n):
                if j in chosen:
                    continue
                val = _old_logdet(fa + np.outer(rs[j], rs[j]))
                bar = best if not np.isfinite(best) \
                    else best + 1e-10 * max(1.0, abs(best))
                if val > bar:
                    best, swap = val, (a, j)
        if swap is None:
            return chosen
        chosen[swap[0]] = swap[1]
        f = rs[chosen].T @ rs[chosen]
        cur = _old_logdet(f)
    return chosen


def _old_design(model, theta, x, n_pick, sigmas, exchange):
    n, p = x.shape[0], model.n_params
    w = np.ones(n) if sigmas is None else 1.0 / np.broadcast_to(
        np.asarray(sigmas, dtype=float), (n,))
    rows = model.jacobian(np.asarray(theta, float), x) * w[:, None]
    scale = np.sqrt(np.mean(rows * rows, axis=0))
    scale[scale == 0.0] = 1.0
    rs = rows / scale
    eps = 1e-12 * float(np.max(np.sum(rs * rs, axis=1)))
    fs = eps * np.eye(p)
    chosen = []
    for _ in range(n_pick):
        best_j, best_det = -1, -np.inf
        for j in range(n):
            if j in chosen:
                continue
            det = float(np.linalg.slogdet(
                fs + np.outer(rs[j], rs[j]))[1])
            if det > best_det:
                best_j, best_det = j, det
        fs = fs + np.outer(rs[best_j], rs[best_j])
        chosen.append(best_j)
    if exchange:
        chosen = _old_exchange(rs, chosen)
        fs = eps * np.eye(p) + rs[chosen].T @ rs[chosen]
    return chosen, (fs - eps * np.eye(p)) * np.outer(scale, scale)


def _problems():
    rng = np.random.default_rng(21)
    decay = Model("decay + offset",
                  lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]) + th[2],
                  ("A", "tau", "c"), "decay with offset, test suite")
    peak = Model("peak + slope",
                 lambda th, x: th[0] * np.exp(
                     -0.5 * ((x[:, 0] - th[1]) / th[2]) ** 2)
                 + th[3] + th[4] * x[:, 0],
                 ("A", "c", "w", "B", "s"), "Gaussian peak, test suite")
    line = Model("line", lambda th, x: th[0] * x[:, 0] + th[1],
                 ("gain", "offset"), "illustrative straight-line model")
    plane = Model("plane", lambda th, x: th[0] * x[:, 0] + th[1] * x[:, 1]
                  + th[2], ("a", "b", "c"), "plane model, test suite")
    one = Model("scale", lambda th, x: th[0] * x[:, 0], ("k",),
                "one-parameter model, test suite")
    out = [(decay, [1.0, 1.0, 0.2], np.linspace(0.0, 3.0, 31), 3, 0.01)]
    for _ in range(6):
        out.append((decay, [1.0, 1.0, 0.2],
                    np.sort(rng.uniform(0.05, 5.0, 40)), 5, 0.1))
        out.append((peak, [2.0, 0.3, 0.8, 0.1, 0.05],
                    np.sort(rng.uniform(-4.0, 4.0, 60)), 8,
                    rng.uniform(0.01, 0.05, 60)))
        out.append((plane, [1.0, -2.0, 0.5],
                    rng.uniform(-1.0, 1.0, (50, 2)), 6, None))
    # exact ties: every candidate repeated, and a symmetric range
    out.append((line, [2.0, 0.0], np.repeat(np.linspace(-1, 1, 9), 3),
                4, 0.05))
    out.append((one, [3.0], np.linspace(-2.0, 2.0, 11), 3, 0.1))
    return out


def _assert_same_design(new_idx, new_fisher, idx, fisher):
    """The batched design against the 0.3.0 loops (since 0.5.0 with
    stated tolerances instead of ==, because batched and one-at-a-time
    LAPACK calls may differ in the last digits on other CPUs). The
    information matrix is a sum of the same rows, so its entries agree
    to rounding: 1e-12 of the largest entry (the sum of at most 8
    rows has a rounding error near 1e-15 of it). The picks must be the
    same, unless rounding flipped a tie: then both sets must have the
    same log-determinant to 1e-9 (relative)."""
    if new_idx != idx:
        def logdet(f):
            sign, val = np.linalg.slogdet(f)
            assert sign > 0
            return val
        a, b = logdet(new_fisher), logdet(fisher)
        assert abs(a - b) <= 1e-9 * max(1.0, abs(b))
        return
    scale = np.max(np.abs(fisher))
    assert np.max(np.abs(new_fisher - fisher)) <= 1e-12 * scale


@pytest.mark.parametrize("exchange", [False, True])
def test_batched_design_equals_the_030_loops(exchange):
    for model, th, cand, k, sig in _problems():
        x = cand[:, None] if np.ndim(cand) == 1 else cand
        new = design(model, th, x, k, sigmas=sig, exchange=exchange)
        idx, fisher = _old_design(model, th, x, k, sig, exchange)
        _assert_same_design(new["indices"], new["fisher"], idx, fisher)


def test_batched_design_in_small_batches(monkeypatch):
    """Batches of a few candidates (the memory limit for large
    problems) give the same picks as one batch."""
    import labplan.plan as plan
    model, th, cand, k, sig = _problems()[2]
    whole = design(model, th, cand, k, sigmas=sig, exchange=True)
    monkeypatch.setattr(plan, "_BATCH_BYTES", 8 * 25 * 7)
    parts = design(model, th, cand, k, sigmas=sig, exchange=True)
    _assert_same_design(parts["indices"], parts["fisher"],
                        whole["indices"], whole["fisher"])
    assert math.isfinite(parts["condition_number"])

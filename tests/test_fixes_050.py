"""0.5.0 anchors for one wrong answer and the faster `inverse_predict`.

- The conformal rank k = ceil((n + 1)(1 - alpha)) took any product
  within 1e-9 (relative) above an integer to be that integer. For
  large n that swallows true fractions: alpha = 0.0001 and
  n = 109998 gave k = 109988 instead of 109989, so
  `coverage_exact` returned 0.99989999909, below 1 - alpha, and
  `conformal_quantile` the score one rank too low. A decimal alpha
  of up to 8 significant digits is now read exactly; the test
  compares with exact rational arithmetic and fails on 0.4.0. Every
  other alpha, and every alpha at the n of the earlier tests, gives
  the same rank as the 0.4.0 rule, kept here as the reference.
- `inverse_predict` solves all readings at once. It is compared with
  the 0.4.0 loop, kept below as the reference, with tolerances
  derived from the rounding of the model (see `_close_to_040`), and
  its refusals with the 0.4.0 messages.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from labplan import Model, conformal_quantile, coverage_exact, \
    inverse_predict
from labplan.conformal import _rank
from labplan.model import Model as _Model

EPS = float(np.finfo(float).eps)


def _rank_040(n, a):
    t = (n + 1) * (1.0 - a)
    return int(np.ceil(t - 1e-9 * max(1.0, t)))


def _first_wrong_n(alpha):
    """The smallest n at which the fractional part of (n + 1)(1 - alpha)
    is 1/q (q the denominator of the decimal alpha) and the 0.4.0 rule
    swallows it: 1e-9 (n + 1)(1 - alpha) >= 1/q."""
    fa = Fraction(repr(alpha))
    q = fa.denominator
    start = math.ceil(1.0 / (q * 1e-9 * float(1 - fa)))
    for big_n in range(start, start + q + 1):
        if (big_n * (1 - fa)) % 1 == Fraction(1, q):
            return big_n - 1
    raise AssertionError("no such n")


def test_conformal_rank_exact_at_large_n():
    # the case in the notes: 0.4.0 gave k = 109988 and coverage
    # 0.99989999909 < 0.9999
    n = 109998
    assert _rank_040(n, 0.0001) == 109988
    assert _rank(n, 0.0001) == 109989
    assert coverage_exact(n, 0.0001) == 109989 / 109999.0
    assert coverage_exact(n, 0.0001) >= 1 - 0.0001
    scores = np.arange(1.0, n + 1.0)
    assert conformal_quantile(scores, 0.0001) == 109989.0
    for alpha in (0.001, 0.0001, 0.1234, 0.05, 0.1):
        n = _first_wrong_n(alpha)
        k = math.ceil((n + 1) * (1 - Fraction(repr(alpha))))
        assert _rank_040(n, alpha) == k - 1         # the 0.4.0 error
        assert _rank(n, alpha) == k
        assert coverage_exact(n, alpha) >= 1 - alpha


def test_conformal_rank_unchanged_where_it_was_right():
    """Typed decimals with 1 to 4 digits, and alphas that are not short
    decimals (1/3, 1 - 0.9, 0.1 * 3, 0.099999999999, which 0.4.0 reads
    as 0.1, ...), at every n up to 2000: the same rank as 0.4.0."""
    alphas = [i / 1000.0 for i in range(1, 1000, 7)]
    alphas += [round(v, 4) for v in
               np.random.default_rng(1).uniform(1e-4, 0.9999, 40)]
    alphas += [1 / 3, 2 / 3, 1 / 7, 1 - 0.9, 0.1 * 3, 1 - 0.95,
               0.7 - 0.6, 0.099999999999, 0.0999999999, 0.1 + 1e-12]
    for alpha in alphas:
        for n in range(1, 2001):
            assert _rank(n, alpha) == _rank_040(n, alpha), (alpha, n)
    # alphas computed by arithmetic are read as the decimal they stand
    # for, also at large n
    n = _first_wrong_n(0.1)
    assert _rank(n, 1 - 0.9) == _rank(n, 0.1) \
        == math.ceil((n + 1) * Fraction(9, 10))


# ------------------------------------------------------------------
# inverse_predict: the 0.4.0 loop, kept as the reference


def _root_sites_040(g):
    sites = []
    k, n = 0, g.size
    while k < n:
        if g[k] == 0.0:
            first = k
            while k + 1 < n and g[k + 1] == 0.0:
                k += 1
            sites.append(("zero", first, k))
        elif k + 1 < n and g[k + 1] != 0.0 and \
                (g[k] > 0.0) != (g[k + 1] > 0.0):
            sites.append(("cross", k))
        k += 1
    return sites


def _bisect_040(func, lo, hi, g_lo):
    best_x, best_g = lo, abs(g_lo)
    for _ in range(2200):
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            break
        gm = func(mid)
        if abs(gm) < best_g:
            best_x, best_g = mid, abs(gm)
        if gm == 0.0:
            return mid
        if (gm > 0.0) == (g_lo > 0.0):
            lo, g_lo = mid, gm
        else:
            hi = mid
    g_hi = abs(func(hi))
    return hi if g_hi < best_g else best_x


def _dfdx_040(model, theta, x0):
    wrapped = _Model("setting slope",
                     lambda xv, _: model.predict(theta, xv.reshape(1, 1)),
                     ("x",), "labplan.inverse_predict internal wrapper")
    return float(wrapped.numeric_jacobian([x0], np.zeros((1, 1)))[0, 0])


def _inverse_040(model, theta, cov, y, u, lo, hi, n_scan=1001,
                 max_slope_change=0.1):
    """The computation of `inverse_predict` in 0.4.0, one reading at a
    time (input checks left out; messages as in 0.4.0)."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    u = np.broadcast_to(np.asarray(u, dtype=float), y.shape)
    grid = np.linspace(lo, hi, n_scan)
    f_grid = model.predict(theta, grid[:, None])

    def g_at(xv, yv):
        return float(model.predict(theta, np.array([[xv]]))[0]) - yv

    x0 = np.empty(y.size)
    for i, yv in enumerate(y):
        g = f_grid - yv
        sites = _root_sites_040(g)
        if not sites:
            raise ValueError(
                f"the model does not reach the reading {yv:.6g} for "
                f"settings between {lo:.6g} and {hi:.6g} (there it runs "
                f"from {f_grid.min():.6g} to {f_grid.max():.6g}); "
                "widen the bracket, or check the reading")
        if len(sites) > 1:
            where = ", ".join(f"near {grid[s[1]]:.6g}" for s in sites[:4])
            raise ValueError(
                f"the model reaches the reading {yv:.6g} at "
                f"{len(sites)} settings in the bracket ({where}"
                f"{', ...' if len(sites) > 4 else ''}): the reading "
                "does not tell them apart. Narrow the bracket to the "
                "branch you calibrated")
        site = sites[0]
        if site[0] == "zero":
            if site[2] > site[1]:
                raise ValueError(
                    f"the model equals the reading {yv:.6g} on a whole "
                    f"range of settings ({grid[site[1]]:.6g} to "
                    f"{grid[site[2]]:.6g}): it is flat there and the "
                    "reading cannot tell the settings apart")
            x0[i] = grid[site[1]]
        else:
            k = site[1]
            x0[i] = _bisect_040(lambda xv: g_at(xv, yv), grid[k],
                                grid[k + 1], g[k])
    slope = np.array([_dfdx_040(model, theta, xv) for xv in x0])
    flat = np.abs(slope) * (hi - lo) \
        <= 1e-8 * float(np.max(np.abs(f_grid)))
    if np.any(flat):
        bad = float(x0[int(np.argmax(flat))])
        raise ValueError(f"the model is flat at the solution "
                         f"x0 = {bad:.6g} (zero slope): the reading "
                         "does not determine the setting there")
    a = model.jacobian(theta, x0[:, None])
    par_cov = a @ cov @ a.T
    par_cov = 0.5 * (par_cov + par_cov.T)
    out_cov = (par_cov + np.diag(u * u)) / np.outer(slope, slope)
    sigma = np.sqrt(np.maximum(np.diag(out_cov), 0.0))
    change = np.zeros(y.size)
    for i, (xv, sv) in enumerate(zip(x0, sigma)):
        if sv == 0.0:
            continue
        for xe in (min(xv + sv, hi), max(xv - sv, lo)):
            if xe != xv:
                change[i] = max(change[i],
                                abs(_dfdx_040(model, theta, xe)
                                    / slope[i] - 1.0))
        if change[i] > max_slope_change:
            raise ValueError(
                f"the slope of the model changes by "
                f"{100 * change[i]:.3g} % within one error bar of "
                f"x0 = {xv:.6g} (+/- {sv:.3g}): the first-order error "
                "bar is not reliable there (the curve bends, or "
                "flattens, within the uncertainty). Reduce the "
                "uncertainty (repeat the reading, calibrate with more "
                "points) or work where the curve is steeper; "
                "max_slope_change sets this limit")
    return {"value": x0, "sigma": sigma, "cov": out_cov, "slope": slope,
            "slope_change": change, "f_size": float(np.max(np.abs(f_grid)))}


def _close_to_040(new, old):
    """Tolerances from the rounding of the model. Evaluating the model
    for many settings in one call may round its value differently, by
    a few units in the last place of the largest prediction F (64 eps F
    allowed). The bisection then judges a midpoint within that distance
    of the reading on the other side, so x0 moves by at most about
    2 * 64 eps F / |slope| (plus 4 units in the last place of x0). The
    central-difference slope with step h = 1e-6 |x0| changes by at
    most 64 eps F / h from the rounding (plus |f''| times the shift of
    x0, far smaller); twice that bound, relative to the slope, plus
    1e-9, is the tolerance `rel` for the slope, and the error bars,
    which divide by the slope, get 2 rel; the slope change, a ratio of
    two slopes, gets 4 max(rel) (1 + change)."""
    f = old["f_size"]
    dx = 128 * EPS * f / np.abs(old["slope"]) \
        + 4 * np.spacing(np.abs(old["value"]))
    assert np.all(np.abs(new["value"] - old["value"]) <= dx)
    rel = 2 * 64 * EPS * f / (1e-6 * np.abs(old["value"])) \
        / np.abs(old["slope"]) + 1e-9
    assert np.all(np.abs(new["slope"] - old["slope"])
                  <= rel * np.abs(old["slope"]))
    assert np.all(np.abs(new["sigma"] - old["sigma"])
                  <= 2 * rel * old["sigma"])
    assert np.all(np.abs(new["slope_change"] - old["slope_change"])
                  <= 4 * rel.max() * (1 + old["slope_change"]))
    d = np.sqrt(np.outer(np.diag(old["cov"]), np.diag(old["cov"])))
    assert np.all(np.abs(new["cov"] - old["cov"])
                  <= 2 * np.add.outer(rel, rel) * d)


def _models():
    quad = Model("quadratic", lambda th, x: th[0] + th[1] * x[:, 0]
                 + th[2] * x[:, 0] ** 2, ("c0", "c1", "c2"),
                 "quadratic calibration, test suite")
    decay = Model("decay", lambda th, x: th[0] * np.exp(-x[:, 0] / th[1])
                  + th[2], ("A", "tau", "B"), "exponential decay, tests")
    sat = Model("saturating", lambda th, x: th[0] * (1 - np.exp(
        -x[:, 0] / th[1])), ("A", "c"), "saturating response, tests")
    logm = Model("log sensor", lambda th, x: th[0] * np.log(x[:, 0])
                 + th[1], ("g", "o"), "logarithmic sensor, tests")
    return [
        (quad, np.array([0.1, 0.04, 1e-5]),
         np.diag([1e-7, 1e-10, 1e-14]), (0.0, 100.0), (2.0, 98.0)),
        (decay, np.array([3.0, 1.7, 0.4]),
         np.array([[4e-4, 1e-5, -2e-5], [1e-5, 9e-4, -3e-5],
                   [-2e-5, -3e-5, 1e-4]]), (0.0, 8.0), (0.2, 4.0)),
        (sat, np.array([10.0, 1.0]), np.diag([1e-6, 1e-8]), (0.0, 8.0),
         (0.2, 2.5)),
        (logm, np.array([0.5, 2.0]), np.diag([1e-6, 1e-6]), (0.1, 50.0),
         (0.5, 40.0)),
    ]


def test_inverse_many_readings_equal_the_040_loop():
    rng = np.random.default_rng(31)
    for model, th, cov, (lo, hi), (a, b) in _models():
        settings = np.sort(rng.uniform(a, b, 40))
        y = model.predict(th, settings)
        grid = np.linspace(lo, hi, 1001)
        # two readings exactly on scan points (a zero of the scan)
        y[:2] = model.predict(th, grid[[200, 300]])
        u = rng.uniform(0.0, 1e-3, y.size)
        old = _inverse_040(model, th, cov, y, u, lo, hi)
        new = inverse_predict(model, reading=y, bracket=(lo, hi),
                              reading_sigma=u, theta=th, cov=cov)
        _close_to_040(new, old)
        assert np.array_equal(new["value"][:2], grid[[200, 300]])
        # one reading at a time
        for i in (0, 5, 17):
            one = inverse_predict(model, reading=y[i], bracket=(lo, hi),
                                  reading_sigma=u[i], theta=th, cov=cov)
            ref = _inverse_040(model, th, cov, y[i], u[i], lo, hi)
            _close_to_040({k: np.atleast_1d(v) for k, v in one.items()},
                          ref)


def test_inverse_refusals_keep_the_040_messages():
    quad, th, cov, (lo, hi), _ = _models()[0]
    peak = Model("peak", lambda th, x: th[0] * np.exp(-(x[:, 0] - th[1])
                                                      ** 2), ("A", "c"),
                 "Gaussian peak, test suite")
    ramp = Model("ramp", lambda th, x: th[0] * np.maximum(x[:, 0] - th[1],
                                                          0.0),
                 ("g", "x0"), "ramp from a threshold, test suite")
    sat = _models()[2]
    cases = [
        (quad, th, cov, [1.0, 50.0], 0.0, (lo, hi)),        # no root
        (quad, th, cov, [0.5, 1.0, 9.0], 0.0, (lo, hi)),
        (peak, np.array([1.0, 0.0]), np.diag([1e-6, 1e-6]), [0.5, 0.3],
         0.0, (-3.0, 3.0)),                                 # two roots
        (ramp, np.array([1.0, 5.0]), np.diag([1e-6, 1e-6]), [1.0, 0.0],
         0.0, (0.0, 10.0)),                                 # flat range
        (sat[0], sat[1], sat[2], [5.0, 9.99], 0.05, (0.0, 8.0)),
        (sat[0], sat[1], sat[2], [5.0, 9.5], 0.05, (0.0, 8.0)),
    ]
    for model, t, c, y, u, (a, b) in cases:
        with pytest.raises(ValueError) as old:
            _inverse_040(model, t, c, np.array(y), u, a, b,
                         max_slope_change=0.1)
        with pytest.raises(ValueError) as new:
            inverse_predict(model, reading=np.array(y), bracket=(a, b),
                            reading_sigma=u, theta=t, cov=c)
        assert str(new.value) == str(old.value)


def test_inverse_without_full_covariance():
    """full_cov=False gives the same values and, from the diagonal
    alone, the same error bars. The diagonal a_i C a_i^T is summed in
    a different order than in a @ C @ a.T; the rounding of either is at
    most 2 p eps sum_jk |a_ij C_jk a_ik|, which bounds the difference
    (with a factor 4 margin)."""
    rng = np.random.default_rng(5)
    for model, th, cov, (lo, hi), (a, b) in _models():
        y = model.predict(th, rng.uniform(a, b, 30))
        full = inverse_predict(model, reading=y, bracket=(lo, hi),
                               reading_sigma=1e-4, theta=th, cov=cov)
        diag = inverse_predict(model, reading=y, bracket=(lo, hi),
                               reading_sigma=1e-4, theta=th, cov=cov,
                               full_cov=False)
        assert diag["cov"] is None
        assert np.array_equal(diag["value"], full["value"])
        assert np.array_equal(diag["slope"], full["slope"])
        aj = model.jacobian(th, full["value"][:, None])
        p = th.size
        bound = 8 * p * EPS * np.einsum("ij,jk,ik->i", np.abs(aj),
                                        np.abs(cov), np.abs(aj))
        var_full = full["sigma"] ** 2 * full["slope"] ** 2
        var_diag = diag["sigma"] ** 2 * diag["slope"] ** 2
        assert np.all(np.abs(var_full - var_diag) <= bound
                      + 8 * EPS * var_full)
    with pytest.raises(ValueError, match="full_cov"):
        inverse_predict(model, reading=y, bracket=(lo, hi), theta=th,
                        cov=cov, full_cov="no")


def test_inverse_scan_in_blocks(monkeypatch):
    """The scan of many readings is cut into blocks of at most
    _SCAN_BLOCK grid values; blocks of 3 readings give the same sites,
    and since the bisection is then run on all readings together, the
    same numbers (compared with ==: the same operations on the same
    arrays), and a refusal in a later block names the same reading."""
    import labplan.inverse as inv
    model, th, cov, (lo, hi), (a, b) = _models()[1]
    y = model.predict(th, np.linspace(a, b, 20))
    whole = inverse_predict(model, reading=y, bracket=(lo, hi),
                            theta=th, cov=cov, full_cov=False)
    monkeypatch.setattr(inv, "_SCAN_BLOCK", 3 * 1001)
    parts = inverse_predict(model, reading=y, bracket=(lo, hi),
                            theta=th, cov=cov, full_cov=False)
    for key in ("value", "sigma", "slope"):
        assert np.array_equal(parts[key], whole[key])
    bad = y.copy()
    bad[13] = 99.0
    with pytest.raises(ValueError, match="does not reach the reading 99"):
        inverse_predict(model, reading=bad, bracket=(lo, hi), theta=th,
                        cov=cov)

"""0.2.0 anchors for error propagation, units in records, the
conformal rank and the sigmas check.

- `propagate` of a straight-line prediction equals the closed form
  [x 1] C [x 1]^T; a product of parameters equals the textbook
  first-order formula; seeded simulated fits scatter as propagated;
  planned and fitted propagation agree.
- The model's `units` note reaches the fit result, the audit record
  (surviving a JSON round trip) and the text report.
- The conformal rank k = ceil((n + 1)(1 - alpha)) equals the same
  expression in exact rational arithmetic (Python `fractions`) for
  every alpha with three decimals and n up to 120; before 0.2.0 it
  was one too large for 255 of those alphas at some n up to 3000
  (e.g. alpha = 0.7, n = 9: k = 4 instead of 3).
"""
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from labplan import (Model, audit_record, conformal_quantile,
                     coverage_exact, fit, information, propagate,
                     report_text)

X = np.linspace(0.5, 5.0, 12)


def _line(units=""):
    return Model("line", lambda th, x: th[0] * x[:, 0] + th[1],
                 ("gain", "offset"), "illustrative straight-line model",
                 units=units)


def test_propagate_line_prediction_closed_form():
    m = _line()
    rng = np.random.default_rng(1)
    y = 2.0 * X + 0.1 + 0.05 * rng.standard_normal(X.size)
    res = fit(m, X, y, [1.0, 0.0], sigmas=0.05)
    x_new = np.array([0.0, 2.5, 7.0])
    out = propagate(lambda th: m.predict(th, x_new), res.theta, res.cov)
    a = np.column_stack([x_new, np.ones(3)])
    want_cov = a @ res.cov @ a.T
    assert np.allclose(out["value"], a @ res.theta, rtol=1e-12)
    assert np.allclose(out["cov"], want_cov, rtol=1e-6, atol=0)
    assert np.allclose(out["sigma"], np.sqrt(np.diag(want_cov)),
                       rtol=1e-6, atol=0)
    # at x = 0 the prediction is the offset itself
    assert abs(out["sigma"][0] - res.sigma["offset"]) \
        < 1e-6 * res.sigma["offset"]
    one = propagate(lambda th: th[0] * 3.0 + th[1], res.theta, res.cov)
    assert isinstance(one["value"], float)
    assert isinstance(one["sigma"], float)


def test_propagate_product_first_order_formula():
    theta = np.array([3.0, -2.0])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    out = propagate(lambda th: th[0] * th[1], theta, cov)
    a, b = theta
    want = b * b * cov[0, 0] + a * a * cov[1, 1] + 2 * a * b * cov[0, 1]
    assert abs(out["sigma"] ** 2 - want) < 1e-6 * want
    assert out["value"] == -6.0


def test_propagate_matches_simulated_scatter():
    """The quantity gain * 4 + offset is linear in the parameters, so
    first-order propagation is exact: 300 seeded fits must scatter by
    the propagated error bar within 15 %."""
    m = _line()
    y0 = m.predict([2.0, 0.1], X)
    rng = np.random.default_rng(8)
    vals = []
    for _ in range(300):
        res = fit(m, X, y0 + 0.2 * rng.standard_normal(X.size),
                  [1.0, 0.0], sigmas=0.2)
        vals.append(4.0 * res.theta[0] + res.theta[1])
    prop = propagate(lambda th: 4.0 * th[0] + th[1], res.theta,
                     res.cov)["sigma"]
    assert abs(np.std(vals, ddof=1) - prop) < 0.15 * prop


def test_propagate_planned_equals_fitted():
    m = _line()
    plan = information(m, [2.0, 0.1], X, sigmas=0.05)
    res = fit(m, X, m.predict([2.0, 0.1], X), [1.0, 0.0], sigmas=0.05)
    g = lambda th: th[1] / th[0]
    a = propagate(g, [2.0, 0.1], plan["cov"])
    b = propagate(g, res.theta, res.cov)
    assert abs(a["sigma"] - b["sigma"]) < 1e-6 * b["sigma"]


def test_propagate_refusals():
    with pytest.raises(ValueError, match="not identifiable"):
        propagate(lambda th: th[0], [1.0], None)
    with pytest.raises(ValueError, match=r"\(2, 2\)"):
        propagate(lambda th: th[0], [1.0, 2.0], np.eye(3))
    with pytest.raises(ValueError, match="symmetric"):
        propagate(lambda th: th[0], [1.0, 2.0], [[1.0, 0.5], [0.0, 1.0]])
    with pytest.raises(ValueError, match="non-negative"):
        propagate(lambda th: th[0], [1.0], [[-1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        propagate(lambda th: np.inf * th[0], [1.0], [[1.0]])
    with pytest.raises(ValueError, match="1-D array"):
        propagate(lambda th: np.outer(th, th), [1.0, 2.0], np.eye(2))
    with pytest.raises(ValueError, match="callable"):
        propagate(3.0, [1.0], [[1.0]])


def test_units_reach_record_and_report():
    rng = np.random.default_rng(1)
    y = 2.0 * X + 0.1 + 0.05 * rng.standard_normal(X.size)
    units = {"gain": "V/K", "offset": "V"}
    res = fit(_line(units), X, y, [1.0, 0.0], sigmas=0.05)
    assert res.units == units
    rec = audit_record(res)
    assert rec["model_units"] == units
    assert json.loads(json.dumps(rec)) == rec
    assert "units:      gain: V/K, offset: V" in report_text(rec)
    rec2 = audit_record(fit(_line("x in K, y in V"), X, y, [1.0, 0.0],
                            sigmas=0.05))
    assert rec2["model_units"] == "x in K, y in V"
    assert "units:      x in K, y in V" in report_text(rec2)
    # records written by 0.1.x have no units key and still render
    del rec2["model_units"]
    assert "units:" not in report_text(rec2)
    assert audit_record(fit(_line(), X, y, [1.0, 0.0], sigmas=0.05)
                        )["model_units"] == ""


def test_conformal_rank_matches_exact_rational_arithmetic():
    for i in range(1, 1000):
        alpha = i / 1000.0
        fa = Fraction(repr(alpha))          # the decimal as typed
        for n in range(1, 121):
            k = math.ceil((n + 1) * (1 - fa))
            if k > n:
                with pytest.raises(ValueError):
                    coverage_exact(n, alpha)
            else:
                assert coverage_exact(n, alpha) == k / (n + 1.0)
    # the minimum n named in the refusal is the exact one
    for alpha in (0.01, 0.05, 0.1, 0.2, 0.3, 0.45):
        fa = Fraction(repr(alpha))
        need = math.ceil((1 - fa) / fa)
        assert need > 1
        with pytest.raises(ValueError, match=f"at least {need} "):
            conformal_quantile(np.ones(need - 1), alpha)
        conformal_quantile(np.ones(need), alpha)
    # the case fixed in 0.2.0: third smallest of 9 scores, not fourth
    s = np.arange(1.0, 10.0)
    assert conformal_quantile(s, 0.7) == 3.0
    assert coverage_exact(9, 0.7) == 0.3


def test_sigmas_length_is_explained():
    with pytest.raises(ValueError, match="one per reading"):
        information(_line(), [2.0, 0.0], X, sigmas=[0.1, 0.2])
    with pytest.raises(ValueError, match="one per reading"):
        fit(_line(), X, 2 * X, [1.0, 0.0], sigmas=np.ones((12, 1)))
    a = information(_line(), [2.0, 0.0], X, sigmas=[0.1])["sigma"]
    b = information(_line(), [2.0, 0.0], X, sigmas=0.1)["sigma"]
    assert a == b

"""Checks for statements the README already made in 0.1.1 but no test
asserted: the covariance of a fit without reading errors, per-unit
planning, the design's first picks, the closed form of
`repeats_for`, finite-difference slopes of a nonlinear model against
exact derivatives, the coverage bounds over many n and alpha, and
the documented refusals of every public function."""
import numpy as np
import pytest

from labplan import (Model, audit_record, conformal_interval,
                     conformal_quantile, coverage_exact, design, fit,
                     information, load_measurements_csv, repeats_for,
                     report_text, save_measurements_csv)

X = np.linspace(0.5, 5.0, 12)
REF = "illustrative straight-line sensor model"


def _line():
    return Model("line", lambda th, x: th[0] * x[:, 0] + th[1],
                 ("gain", "offset"), REF)


def test_fit_without_sigmas_uses_residual_scatter():
    """Closed form for ordinary least squares: cov = s^2 (A^T A)^-1
    with s^2 = RSS / (n - p), the estimate from np.linalg.lstsq."""
    rng = np.random.default_rng(0)
    y = 2.0 * X + 0.1 + 0.05 * rng.standard_normal(X.size)
    res = fit(_line(), X, y, [1.0, 0.0])
    a = np.column_stack([X, np.ones(X.size)])
    beta = np.linalg.lstsq(a, y, rcond=None)[0]
    s2 = np.sum((y - a @ beta) ** 2) / (X.size - 2)
    assert np.allclose(res.theta, beta, rtol=1e-8)
    assert np.allclose(res.cov, s2 * np.linalg.inv(a.T @ a), rtol=1e-6)
    assert res.chi2 is None and res.chi2_dof is None
    assert "not available" in report_text(audit_record(res))
    with pytest.raises(ValueError, match="error scale"):
        fit(_line(), X[:2], y[:2], [1.0, 0.0])


def test_information_per_unit_error_and_too_few_settings():
    m = _line()
    unit = information(m, [2.0, 0.0], X)["sigma"]
    s05 = information(m, [2.0, 0.0], X, sigmas=0.05)["sigma"]
    for k in unit:
        assert abs(s05[k] - 0.05 * unit[k]) < 1e-12 * s05[k]
    few = information(m, [2.0, 0.0], X[:1], sigmas=0.05)
    assert few["identifiable"] is False
    assert few["sigma"] is None and few["cov"] is None


def test_design_takes_the_ends_of_a_line_first():
    """README example 2: for a straight line the two extreme settings
    are picked first."""
    cand = np.linspace(0.5, 5.0, 12)
    idx = design(_line(), [2.0, 0.0], cand, 4, sigmas=0.05)["indices"]
    assert sorted(idx[:2]) == [0, 11]


def test_repeats_for_closed_form_and_refusals():
    plan = information(_line(), [2.0, 0.0], X, sigmas=0.05)
    t = 0.004
    r, pred = repeats_for(t, plan)
    worst = max(plan["sigma"].values()) / t
    assert r == int(np.ceil(worst ** 2))
    for k in pred:
        assert abs(pred[k] - plan["sigma"][k] / np.sqrt(r)) \
            < 1e-12 * pred[k]
    assert repeats_for(1e3, plan)[0] == 1
    bad = information(_line(), [2.0, 0.0], X[:1], sigmas=0.05)
    with pytest.raises(ValueError, match="not identifiable"):
        repeats_for(t, bad)
    with pytest.raises(ValueError, match="unknown parameter"):
        repeats_for({"slope": t}, plan)
    with pytest.raises(ValueError, match="positive"):
        repeats_for(0.0, plan)
    with pytest.raises(ValueError, match="positive"):
        repeats_for({"gain": -1.0}, plan)


def test_numeric_slopes_of_a_nonlinear_model():
    """Exponential decay: exact slopes e and A t e / tau^2."""
    m = Model("decay", lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]),
              ("A", "tau"), "exponential decay, test suite")
    th = np.array([3.0, 1.7])
    e = np.exp(-X / th[1])
    exact = np.column_stack([e, th[0] * X * e / th[1] ** 2])
    assert np.allclose(m.jacobian(th, X), exact, rtol=1e-6, atol=0)


def test_coverage_bounds_many_levels():
    for n in range(1, 200):
        for alpha in (0.01, 0.05, 0.1, 0.2, 0.25, 0.5):
            try:
                c = coverage_exact(n, alpha)
            except ValueError:
                continue
            assert 1 - alpha - 1e-12 <= c <= 1 - alpha + 1 / (n + 1)


def test_conformal_refusals():
    with pytest.raises(ValueError, match="certifiable"):
        coverage_exact(5, 0.1)
    with pytest.raises(ValueError):
        coverage_exact(0, 0.1)
    with pytest.raises(ValueError):
        coverage_exact(30, 0.0)
    with pytest.raises(ValueError, match=">= 0"):
        conformal_interval([1.0], -0.1)
    with pytest.raises(ValueError, match=">= 0"):
        conformal_interval([1.0], np.inf)
    with pytest.raises(ValueError, match=">= 0"):
        conformal_quantile([0.1] * 29 + [np.nan])


def test_model_and_fit_refusals():
    with pytest.raises(ValueError, match="name"):
        Model(" ", lambda th, x: x[:, 0], ("a",), REF)
    m = Model("wrong length", lambda th, x: x[:2, 0], ("a",), REF)
    with pytest.raises(ValueError, match="one prediction per"):
        m.predict([1.0], X)
    m = Model("nan", lambda th, x: x[:, 0] * np.nan, ("a",), REF)
    with pytest.raises(ValueError, match="non-finite"):
        m.predict([1.0], X)
    with pytest.raises(ValueError, match="non-empty"):
        Model("m", lambda th, x: x[:, 0], (), REF)
    with pytest.raises(ValueError, match="2 entries"):
        _line().predict([1.0], X)
    with pytest.raises(ValueError, match="settings must be finite"):
        _line().predict([1.0, 0.0], [1.0, np.nan])
    with pytest.raises(ValueError, match="finite and positive"):
        information(_line(), [2.0, 0.0], X, sigmas=np.inf)
    with pytest.raises(ValueError, match="finite"):
        fit(_line(), X, np.full(X.size, np.inf), [1.0, 0.0], sigmas=0.1)
    with pytest.raises(ValueError, match="one measured reading"):
        fit(_line(), X, X[:3], [1.0, 0.0], sigmas=0.1)
    with pytest.raises(ValueError, match="n_pick"):
        design(_line(), [2.0, 0.0], X, 13, sigmas=0.1)
    decay = Model("decay",
                  lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]),
                  ("A", "tau"), "exponential decay, test suite")
    y = decay.predict([3.0, 1.7], X)
    with pytest.raises(RuntimeError, match="converge"):
        fit(decay, X, y, [1.0, 3.0], sigmas=0.01, max_iter=2)


def test_parameter_no_reading_responds_to():
    """A parameter that no reading responds to (an all-zero slope
    column): `information` reports it not identifiable, `fit` and
    `design` refuse."""
    m = Model("unused b", lambda th, x: th[0] * x[:, 0], ("a", "b"), REF)
    assert np.all(m.jacobian([1.0, 2.0], X)[:, 1] == 0.0)
    plan = information(m, [1.0, 2.0], X, sigmas=0.1)
    assert plan["identifiable"] is False and plan["sigma"] is None
    with pytest.raises(ValueError, match="cannot tell"):
        fit(m, X, X, [1.0, 2.0], sigmas=0.1)
    with pytest.raises(ValueError, match="identifiable"):
        design(m, [1.0, 2.0], X, 3, sigmas=0.1)


def test_model_with_dict_units_is_hashable():
    """Model is a frozen dataclass; a dictionary `units` note must not
    make it unhashable (it did in 0.1.x)."""
    f = lambda th, x: th[0] * x[:, 0] + th[1]
    a = Model("line", f, ("gain", "offset"), REF,
              units={"gain": "V/K", "offset": "V"})
    b = Model("line", f, ("gain", "offset"), REF,
              units={"gain": "V/K", "offset": "V"})
    c = Model("line", f, ("gain", "offset"), REF, units="V")
    assert hash(a) == hash(b) and a == b
    assert a != c
    assert len({a, b, c}) == 2


def test_record_refusals():
    res = fit(_line(), X, 2.0 * X, [1.0, 0.0], sigmas=0.1)
    with pytest.raises(ValueError, match="strings"):
        audit_record(res, operator=3)
    with pytest.raises(ValueError, match="strings"):
        audit_record(res, note=None)


def test_csv_refusals_and_exact_awkward_values(tmp_path):
    p = tmp_path / "m.csv"
    x = np.array([1.0 / 3.0, 1e-300, -2.5e17, 0.1])
    y = np.array([np.pi, -1e-12, 7.0, 2.0 / 3.0])
    save_measurements_csv(p, x, y, sigmas=[0.1, 1.0 / 7.0, 3.0, 1e-9])
    x2, y2, s2 = load_measurements_csv(p)
    assert np.array_equal(x2[:, 0], x) and np.array_equal(y2, y)
    assert np.array_equal(s2, [0.1, 1.0 / 7.0, 3.0, 1e-9])
    cases = {
        "": "empty",
        "x1,y\n": "no data rows",
        "x1,y\n1.0,2.0,3.0\n": "does not match",
        "x1,y\n1.0,abc\n": "non-numeric",
        "x1,y,sigma\n1.0,2.0,-1.0\n": "positive",
        "x1,y\n1.0,nan\n": "finite",
        "x2,y\n1.0,2.0\n": "header",
    }
    for text, msg in cases.items():
        p.write_text(text)
        with pytest.raises(ValueError, match=msg):
            load_measurements_csv(p)

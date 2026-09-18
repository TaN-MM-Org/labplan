"""Conformal, record and file anchors: the split-conformal quantile
is the exact rank formula recomputed independently; seeded simulation
with continuous scores matches the EXACT closed-form coverage
ceil((n+1)(1-alpha))/(n+1) within binomial tolerance, which sits
inside the two-sided [1-alpha, 1-alpha + 1/(n+1)] guarantee of the
references; too-small calibration sets are refused with the minimum
named; the audit record survives JSON round trip exactly and pins
the data by sha256 (any change to the data changes the digest); and
the measurement CSV round trip is bit-exact."""
import json

import numpy as np
import pytest

from labplan import (Model, audit_record, conformal_interval,
                     conformal_quantile, coverage_exact, fit,
                     load_measurements_csv, report_text,
                     save_measurements_csv)

REF = "line model, unit test suite (y = a x + b)"


def test_quantile_is_exact_rank_formula():
    rng = np.random.default_rng(2)
    s = rng.exponential(1.0, 40)
    for alpha in (0.05, 0.1, 0.25):
        q = conformal_quantile(s, alpha)
        k = int(np.ceil((s.size + 1) * (1.0 - alpha)))
        assert q == float(np.sort(s)[k - 1])


def test_exact_coverage_by_seeded_simulation():
    """Continuous iid scores: P(new score <= q) is EXACTLY
    ceil((n+1)(1-alpha))/(n+1). 4000 seeded trials must match the
    closed form within 4 binomial standard errors, and the closed
    form must sit inside the two-sided guarantee."""
    n, alpha = 29, 0.1
    p_exact = coverage_exact(n, alpha)
    assert 1 - alpha <= p_exact <= 1 - alpha + 1.0 / (n + 1)
    rng = np.random.default_rng(7)
    hits = 0
    trials = 4000
    for _ in range(trials):
        s = rng.standard_normal(n) ** 2
        q = conformal_quantile(s, alpha)
        hits += float(rng.standard_normal() ** 2) <= q
    p_hat = hits / trials
    se = np.sqrt(p_exact * (1 - p_exact) / trials)
    assert abs(p_hat - p_exact) < 4.0 * se


def test_interval_and_refusals():
    lo, hi = conformal_interval(np.array([1.0, 2.0]), 0.5)
    assert np.allclose(lo, [0.5, 1.5]) and np.allclose(hi, [1.5, 2.5])
    with pytest.raises(ValueError, match="at least"):
        conformal_quantile([0.1, 0.2, 0.3], alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile([0.1] * 30, alpha=1.5)
    with pytest.raises(ValueError, match=">= 0"):
        conformal_quantile([-1.0] * 30)


def _fit():
    m = Model("line", lambda th, x: th[0] * x[:, 0] + th[1],
              ("slope", "intercept"), REF)
    x = np.linspace(0.5, 5.0, 8)[:, None]
    y = m.predict([2.0, 1.0], x)
    return m, x, y, fit(m, x, y, [1.0, 0.0], sigmas=0.05)


def test_audit_record_round_trip_and_digest_pins_data():
    m, x, y, res = _fit()
    rec = audit_record(res, operator="test bench",
                       note="unit test record")
    again = json.loads(json.dumps(rec))
    assert again == rec
    assert rec["parameters"]["slope"]["value"] == res.values["slope"]
    assert rec["data_sha256"] == res.data_digest
    # the digest answers for the exact data: change one reading and
    # the fit of the changed data carries a different digest
    y2 = y.copy()
    y2[0] += 1e-9
    res2 = fit(m, x, y2, [1.0, 0.0], sigmas=0.05)
    assert res2.data_digest != res.data_digest
    txt = report_text(rec)
    assert "labplan calibration record" in txt
    assert "slope" in txt and "test bench" in txt
    with pytest.raises(ValueError, match="audit_record"):
        report_text({"nope": 1})


def test_measurements_csv_round_trip_exact(tmp_path):
    rng = np.random.default_rng(9)
    x = rng.standard_normal((7, 3))
    y = rng.standard_normal(7)
    sig = np.abs(rng.standard_normal(7)) + 0.1
    path = tmp_path / "meas.csv"
    save_measurements_csv(path, x, y, sig)
    x2, y2, s2 = load_measurements_csv(path)
    assert np.array_equal(x, x2)
    assert np.array_equal(y, y2)
    assert np.array_equal(sig, s2)
    save_measurements_csv(path, x[:, :1], y)         # 1-D, no sigma
    x3, y3, s3 = load_measurements_csv(path)
    assert x3.shape == (7, 1) and s3 is None
    bad = tmp_path / "bad.csv"
    bad.write_text("wrong,header\n1,2\n")
    with pytest.raises(ValueError, match="header"):
        load_measurements_csv(bad)


def test_version_metadata():
    import pathlib
    import re

    import labplan
    root = pathlib.Path(__file__).resolve().parents[1]
    v = labplan.__version__
    py = (root / "pyproject.toml").read_text()
    assert re.search(rf'^version = "{re.escape(v)}"$', py, re.M)
    cff = (root / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(v)}$", cff, re.M)
    for name in labplan.__all__:
        assert hasattr(labplan, name), name

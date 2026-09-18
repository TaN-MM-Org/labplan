"""Distribution-free prediction intervals with an exact guarantee.

The model-based error bars elsewhere in this package assume the model
is right and the noise is Gaussian. Split conformal prediction makes
a different, weaker promise that holds with NO such assumptions: hold
out calibration measurements, score how wrong the model was on each
(for instance |measured - predicted|), and take

    q = the ceil((n + 1)(1 - alpha))-th smallest of the n scores.

Then a new measurement, exchangeable with the calibration set, falls
within q of its prediction with probability at least 1 - alpha --
exactly, at finite n, whatever the model and whatever the noise (V.
Vovk, A. Gammerman and G. Shafer, Algorithmic Learning in a Random
World, Springer (2005); J. Lei et al., J. Am. Stat. Assoc. 113, 1094
(2018); A. N. Angelopoulos and S. Bates, arXiv:2107.07511). For
continuous scores the coverage is also at most 1 - alpha + 1/(n + 1),
and its exact value is ceil((n + 1)(1 - alpha)) / (n + 1) -- a rank
statement the tests verify by seeded simulation against the closed
form.

Honest limits, stated plainly: the guarantee is marginal (on average
over calibration sets and test points, not conditional on a
particular x), and it needs exchangeability -- calibration data from
last month's instrument state do not certify next month's drift.
"""
from __future__ import annotations

import numpy as np

__all__ = ["conformal_quantile", "conformal_interval",
           "coverage_exact"]


def conformal_quantile(scores, alpha=0.1):
    """The split-conformal quantile of held-out error scores.

    scores : (n,) nonnegative error scores of held-out calibration
        measurements (e.g. |measured - predicted|).
    alpha : miscoverage level (0.1 = 90% intervals).

    Returns q such that |new error| <= q with probability >= 1-alpha.
    Refuses when n is too small for the level: the rank
    ceil((n+1)(1-alpha)) must exist among n scores, which needs
    n >= (1-alpha)/alpha.
    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size < 1 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("scores must be finite and >= 0 (use "
                         "absolute errors)")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    n = s.size
    k = int(np.ceil((n + 1) * (1.0 - a)))
    if k > n:
        need = int(np.ceil((1.0 - a) / a))
        raise ValueError(
            f"{n} calibration scores cannot certify level "
            f"{1 - a:.3g}: the required rank {k} exceeds n. Collect "
            f"at least {need} calibration measurements, or lower the "
            "confidence")
    return float(np.sort(s)[k - 1])


def conformal_interval(prediction, q):
    """The interval the guarantee applies to: prediction +/- q.

    Returns (lo, hi) arrays matching the prediction's shape.
    """
    pred = np.asarray(prediction, dtype=float)
    qv = float(q)
    if not (np.isfinite(qv) and qv >= 0.0):
        raise ValueError("q must be finite and >= 0")
    return pred - qv, pred + qv


def coverage_exact(n, alpha):
    """The exact marginal coverage for continuous scores:
    ceil((n+1)(1-alpha)) / (n+1). It always lies in
    [1-alpha, 1-alpha + 1/(n+1)] -- the two-sided guarantee of the
    references, recovered as arithmetic."""
    n = int(n)
    a = float(alpha)
    if n < 1 or not (0.0 < a < 1.0):
        raise ValueError("need n >= 1 and alpha in (0, 1)")
    k = int(np.ceil((n + 1) * (1.0 - a)))
    if k > n:
        raise ValueError("level not certifiable at this n (see "
                         "conformal_quantile)")
    return k / (n + 1.0)

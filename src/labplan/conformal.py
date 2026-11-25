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

import functools
import math
from fractions import Fraction

import numpy as np

__all__ = ["conformal_quantile", "conformal_interval",
           "coverage_exact"]


# floating-point slack when matching alpha to a short decimal: a value
# computed as 1 - c is off by at most about one unit in the last place
# of 1 (1 - 0.9 = 0.09999999999999998)
_ULP_ONE = float(np.finfo(float).eps)


@functools.lru_cache(maxsize=256)
def _decimal_alpha(a):
    """The decimal number alpha stands for, as an exact Fraction, when
    alpha is a decimal of at most 8 significant digits (0.1, 0.05,
    0.0001, 0.1234) up to floating-point rounding (1 - 0.9 counts as
    0.1); otherwise None (since 0.5.0)."""
    tol = min(4.0 * _ULP_ONE, 1e-9 * a)
    for digits in range(1, 9):
        text = f"{a:.{digits}g}"
        if abs(float(text) - a) <= tol:
            return Fraction(text)
    return None


@functools.lru_cache(maxsize=256)
def _fraction_alpha(a):
    """alpha as a fraction with a denominator of at most 10**6 (1/3,
    2/7, 1/3000) when the float alpha is within floating-point
    rounding of one (the tolerance of `_decimal_alpha`); otherwise
    None (since 0.7.0)."""
    frac = Fraction(a).limit_denominator(10 ** 6)
    if frac.numerator > 0 and \
            abs(float(frac) - a) <= min(4.0 * _ULP_ONE, 1e-9 * a):
        return frac
    return None


@functools.lru_cache(maxsize=256)
def _read_alpha(a):
    """The exact number that alpha stands for, as a Fraction (since
    0.7.0). In this order:

    1. a decimal of at most 8 significant digits, up to floating-point
       rounding (0.1, 0.0001, and also 1 - 0.9); since 0.5.0;
    2. a fraction with a denominator of at most 10**6, up to the same
       rounding (1/3, 2/7);
    3. otherwise the shortest decimal that gives back the same float,
       which is the number as typed for any decimal of up to 15
       significant digits (0.099999999999 stays 0.099999999999)."""
    for read in (_decimal_alpha, _fraction_alpha):
        value = read(a)
        if value is not None:
            return value
    return Fraction(repr(a))


def _rank(n, a):
    """k = ceil((n + 1)(1 - a)), with alpha read as the number the
    user typed (see `_read_alpha`), in exact integer arithmetic at
    every n.

    History. In binary floating point 1 - 0.7 is 0.30000000000000004,
    so a plain ceil turns an exact integer such as 10 * 0.3 = 3 into
    4. From 0.2.0 to 0.4.0 a product within 1e-9 (relative) above an
    integer was taken to be that integer; for large n that also
    swallowed true fractions (alpha = 0.0001 and n = 109998 gave
    k = 109988 instead of 109989, a coverage below 1 - alpha). 0.5.0
    read decimals of up to 8 digits exactly and kept the old rule for
    every other alpha. That rule read alpha = 0.099999999999 as 0.1:
    for n = 9 it gave k = 9 and a coverage of 0.9, below the promised
    1 - alpha = 0.900000000001. Since 0.7.0 every alpha is read
    exactly, so k / (n + 1) >= 1 - alpha always holds for the number
    alpha is read as."""
    value = _read_alpha(a)
    num = (n + 1) * (value.denominator - value.numerator)
    return -(-num // value.denominator)


def _min_n(a):
    """The smallest n whose rank _rank(n, a) exists among n scores.
    k <= n means (n + 1)(1 - alpha) <= n, that is n >= (1 - alpha) /
    alpha; computed exactly since 0.7.0 (before, by a search)."""
    value = _read_alpha(a)
    return max(1, math.ceil((1 - value) / value))


def conformal_quantile(scores, alpha: float = 0.1) -> float:
    """The split-conformal quantile of held-out error scores.

    scores : (n,) nonnegative error scores of held-out calibration
        measurements (e.g. |measured - predicted|).
    alpha : miscoverage level (0.1 = 90% intervals).

    Returns q such that |new error| <= q with probability >= 1-alpha.
    Refuses when n is too small for the level: the rank
    ceil((n+1)(1-alpha)) must exist among n scores, which needs
    n >= (1-alpha)/alpha. alpha is read as an exact number (see
    `_read_alpha`; a short decimal since 0.5.0, any alpha since
    0.7.0), so alpha=0.7 means 7/10 and n=9 gives k=3.
    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size < 1 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("scores must be finite and >= 0 (use "
                         "absolute errors)")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    n = s.size
    k = _rank(n, a)
    if k > n:
        need = _min_n(a)
        # the level to 10 digits: with 3 (before 0.6.0) any alpha below
        # 0.0005 printed as "level 1"
        raise ValueError(
            f"{n} calibration scores cannot certify level "
            f"{1 - a:.10g}: the required rank {k} exceeds n. Collect "
            f"at least {need} calibration measurements, or lower the "
            "confidence")
    return float(np.sort(s)[k - 1])


def conformal_interval(prediction, q: float) -> tuple:
    """The interval the guarantee applies to: prediction +/- q.

    Returns (lo, hi) arrays matching the prediction's shape.
    """
    pred = np.asarray(prediction, dtype=float)
    qv = float(q)
    if not (np.isfinite(qv) and qv >= 0.0):
        raise ValueError("q must be finite and >= 0")
    return pred - qv, pred + qv


def coverage_exact(n: int, alpha: float) -> float:
    """The exact marginal coverage for continuous scores:
    ceil((n+1)(1-alpha)) / (n+1). It always lies in
    [1-alpha, 1-alpha + 1/(n+1)] -- the two-sided guarantee of the
    references, recovered as arithmetic. n must be a whole number
    (since 0.4.0; 0.3.0 silently cut 29.9 to 29)."""
    if isinstance(n, bool) or not np.isfinite(float(n)) \
            or float(n) != int(n):
        raise ValueError(f"n must be a whole number; got {n!r}")
    n = int(n)
    a = float(alpha)
    if n < 1 or not (0.0 < a < 1.0):
        raise ValueError("need n >= 1 and alpha in (0, 1)")
    k = _rank(n, a)
    if k > n:
        raise ValueError("level not certifiable at this n (see "
                         "conformal_quantile)")
    return k / (n + 1.0)

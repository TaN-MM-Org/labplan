"""The shared exact statistics: weighted least squares, Fisher
information, scale-invariant identifiability.

Everything here is the standard machinery of independent Gaussian
measurement errors, stated plainly (see e.g. D. R. Cox and D. V.
Hinkley, Theoretical Statistics (1974), or any statistics text under
"Cramer-Rao bound"): the information matrix of a design is J^T W J
with J the model sensitivities and W = diag(1/sigma^2), the
covariance of the weighted-least-squares estimate is its inverse, and
both are the SAME matrix -- which is why labplan can promise, before
any data exist, exactly the error bars the fit will report.

Identifiability is judged on the correlation-scaled matrix
D^-1 F D^-1 with D = sqrt(diag F): parameters carry arbitrary units,
so the raw condition number partly measures the units, while exact
functional degeneracies of the model survive the scaling and unit
mismatches do not.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["invert_information", "gamma_regularized"]

COND_MAX = 1e10

_EPS = float(np.finfo(float).eps)

SINGULAR_MSG = ("these measurements cannot tell the parameters apart "
                "(singular or near-singular information matrix): some "
                "combination of parameters changes nothing the design "
                "can see. Measure settings that respond differently "
                "to each parameter -- `information` shows which "
                "designs work before you spend the instrument time")


def invert_information(fisher, names):
    """Scale-invariant inversion of an information matrix.

    Returns (identifiable, condition_number, cov, sigma_dict); cov and
    sigma are None when the design is not identifiable.
    """
    fisher = np.asarray(fisher, dtype=float)
    d = np.sqrt(np.diag(fisher))
    if np.any(d <= 0.0) or not np.all(np.isfinite(d)):
        return False, np.inf, None, None
    fs = fisher / np.outer(d, d)
    sv = np.linalg.svd(fs, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    if not (np.isfinite(cond) and cond <= COND_MAX):
        return False, cond, None, None
    cov = np.linalg.inv(fs) / np.outer(d, d)
    err = np.sqrt(np.diag(cov))
    return True, cond, cov, {n: float(s) for n, s in zip(names, err)}


def gamma_regularized(a, x):
    """The regularized incomplete gamma functions (P(a, x), Q(a, x)),
    with P + Q = 1, for a > 0 and x >= 0.

    P(a, x) = gamma(a, x) / Gamma(a) is the lower and Q(a, x) the upper
    function (NIST Digital Library of Mathematical Functions, chapter
    8, section 8.2). The chi-squared distribution with k degrees of
    freedom has P(X <= t) = P(k/2, t/2) and P(X >= t) = Q(k/2, t/2).

    Computed as in W. H. Press, S. A. Teukolsky, W. T. Vetterling and
    B. P. Flannery, Numerical Recipes, 3rd ed., Cambridge University
    Press (2007), section 6.2: the power series of P for x < a + 1 and
    the continued fraction of Q (evaluated by the modified Lentz
    method of their section 5.2) otherwise. The function that is
    computed directly is the smaller one near the tail, so a small P
    or a small Q keeps its full relative precision; the other one is
    1 minus it."""
    a = float(a)
    x = float(x)
    if not (math.isfinite(a) and a > 0.0):
        raise ValueError("a must be finite and positive")
    if not (math.isfinite(x) and x >= 0.0):
        raise ValueError("x must be finite and >= 0")
    if x == 0.0:
        return 0.0, 1.0
    log_front = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:
        # P = x^a e^-x / Gamma(a + 1) * sum_n x^n / ((a+1)...(a+n))
        term = total = 1.0 / a
        ap = a
        for _ in range(100000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * _EPS:
                break
        else:                                   # pragma: no cover
            raise ArithmeticError("incomplete gamma series did not "
                                  "converge")
        p = total * math.exp(log_front)
        p = min(p, 1.0)
        return p, 1.0 - p
    # Q = x^a e^-x / Gamma(a) * 1/(x+1-a- 1(1-a)/(x+3-a- 2(2-a)/...))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= 4.0 * _EPS:
            break
    else:                                       # pragma: no cover
        raise ArithmeticError("incomplete gamma continued fraction did "
                              "not converge")
    q = min(h * math.exp(log_front), 1.0)
    return 1.0 - q, q


def check_sigmas(sigmas, n):
    if sigmas is None:
        return None
    arr = np.asarray(sigmas, dtype=float)
    if arr.size != 1 and arr.shape != (n,):
        raise ValueError(f"sigmas must be one number for all readings "
                         f"or {n} numbers, one per reading; got shape "
                         f"{arr.shape}")
    sig = np.broadcast_to(arr.reshape(()) if arr.size == 1 else arr,
                          (n,)).copy()
    if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
        raise ValueError("sigmas must be finite and positive")
    return sig

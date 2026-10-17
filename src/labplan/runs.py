"""Do the residuals wander? The runs test for systematic misfit.

Chi-squared (`goodness_of_fit`) asks whether the residuals are too
large for the stated reading errors. It cannot see a pattern: a
straight line through gently curved readings, with reading errors
that are a little overstated, can give a perfectly ordinary chi2
while the residuals are positive at both ends and negative in the
middle. Ordered by the setting, such residuals come in long runs of
one sign.

`runs_test` counts the runs (maximal blocks of residuals with the
same sign) and compares the count with its distribution when every
arrangement of the n1 positive and n2 negative signs is equally
likely, which is the case for independent readings whose errors are
exchangeable (A. Wald and J. Wolfowitz, "On a test whether two
samples are from the same population", Ann. Math. Statist. 11,
147-162 (1940), doi:10.1214/aoms/1177731909). The number of runs R
then has the exact distribution, with C(m, k) the binomial
coefficient and N = n1 + n2,

    P(R = 2k)     = 2 C(n1-1, k-1) C(n2-1, k-1) / C(N, n1),
    P(R = 2k + 1) = [C(n1-1, k) C(n2-1, k-1)
                     + C(n1-1, k-1) C(n2-1, k)] / C(N, n1),

with mean 1 + 2 n1 n2 / N and variance
2 n1 n2 (2 n1 n2 - N) / (N^2 (N - 1)). The tests check all three by
enumerating every arrangement of small n1 and n2, and the
probabilities against exact integer arithmetic for larger ones.

Residuals of a least-squares fit are not exactly independent (they
sum to zero against every slope of the model), so for fit residuals
the test is a close approximation, not exact; the tests measure its
false-alarm rate on correct straight-line fits.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["runs_test"]


def _log_comb(m, k):
    return math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)


def _log_pmf(r, n1, n2):
    """log P(R = r) for n1, n2 >= 1, or -inf when r is impossible."""
    total = _log_comb(n1 + n2, n1)
    if r < 2:
        return -math.inf
    if r % 2 == 0:
        k = r // 2
        if k - 1 > n1 - 1 or k - 1 > n2 - 1:
            return -math.inf
        return (math.log(2.0) + _log_comb(n1 - 1, k - 1)
                + _log_comb(n2 - 1, k - 1) - total)
    k = (r - 1) // 2
    parts = []
    if k <= n1 - 1 and k - 1 <= n2 - 1:
        parts.append(_log_comb(n1 - 1, k) + _log_comb(n2 - 1, k - 1))
    if k - 1 <= n1 - 1 and k <= n2 - 1:
        parts.append(_log_comb(n1 - 1, k - 1) + _log_comb(n2 - 1, k))
    if not parts:
        return -math.inf
    return _log_sum(parts) - total


def _log_sum(logs):
    top = max(logs)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(v - top) for v in logs))


def _tail(rs, n1, n2):
    return min(1.0, math.exp(_log_sum([_log_pmf(r, n1, n2) for r in rs])))


def runs_test(residuals, order=None) -> dict:
    """The Wald-Wolfowitz runs test on the signs of residuals (new in
    0.4.0).

    residuals : (n,) residuals, e.g. y - model.predict(res.theta, x)
        (or those divided by the reading errors: only the signs
        count).
    order : optional (n,) values to sort the residuals by before the
        runs are counted, normally the setting x of each reading. Without
        it the residuals are taken in the order given. Ties keep their
        given order.

    Residuals that are exactly zero have no sign and are left out
    (counted in `n_zero`).

    Returns a dictionary:

    runs : the number of runs of one sign.
    n_positive, n_negative, n_zero : the counts of each sign.
    expected_runs, sigma_runs : mean and standard deviation of the
        number of runs for a random arrangement of the signs.
    p_value : probability of this few runs or fewer, P(R <= runs), for
        a random arrangement. A small value means the residuals come
        in long blocks of one sign: the model misses a systematic
        trend in the data (a bend, a drift, a missing term).
    p_value_high : P(R >= runs), the opposite tail: a small value
        means the signs alternate more than chance allows (readings
        that are anti-correlated, or an over-flexible model).

    Both tails are sums of the exact distribution above, computed in
    logarithms (they agree with exact integer arithmetic to 1e-9
    relative in the tests, for n up to 1600); since R takes whole
    values, p_value <= alpha happens with probability at most alpha
    for a random arrangement. Refuses fewer than 2 nonzero residuals,
    and residuals that all have the same sign (no runs to count: with
    a fitted offset that cannot happen, without one it says the model
    misses the level of the data).
    """
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 1:
        raise ValueError("residuals must be a 1-D array")
    if not np.all(np.isfinite(r)):
        raise ValueError("residuals must be finite")
    if order is not None:
        key = np.asarray(order, dtype=float)
        if key.shape != r.shape:
            raise ValueError(f"order must have one value per residual "
                             f"({r.size}); got shape {key.shape}")
        if not np.all(np.isfinite(key)):
            raise ValueError("order must be finite")
        r = r[np.argsort(key, kind="stable")]
    n_zero = int(np.sum(r == 0.0))
    signs = r[r != 0.0] > 0.0
    n1 = int(np.sum(signs))
    n2 = int(signs.size - n1)
    if signs.size < 2:
        raise ValueError(f"need at least 2 nonzero residuals; got "
                         f"{signs.size}")
    if n1 == 0 or n2 == 0:
        raise ValueError(
            f"all {signs.size} nonzero residuals are "
            f"{'positive' if n2 == 0 else 'negative'}: there are no "
            "runs to count. The model misses the overall level of the "
            "data (with a fitted offset this cannot happen)")
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    n = n1 + n2
    mean = 1.0 + 2.0 * n1 * n2 / n
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1.0))
    r_max = 2 * min(n1, n2) + (1 if n1 != n2 else 0)
    low = _tail(range(2, runs + 1), n1, n2)
    high = _tail(range(runs, r_max + 1), n1, n2)
    return {"runs": runs, "n_positive": n1, "n_negative": n2,
            "n_zero": n_zero, "expected_runs": mean,
            "sigma_runs": math.sqrt(var), "p_value": low,
            "p_value_high": high}

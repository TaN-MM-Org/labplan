"""How to share the measurements among the candidate settings, and how
close any design comes to the best one (new in 0.6.0).

`design` picks distinct settings one at a time, and its result is a
good set, not a proven best one. This module answers two questions
that `design` cannot:

- How should a number of measurements be shared among the candidate
  settings, repeats allowed, to make the determinant of the
  information matrix as large as possible? `design_weights` returns
  the share (the weight) of each candidate in that best design, and,
  for a given total number of measurements, whole numbers of
  measurements per candidate.
- How good is a given design compared with the best one over the same
  candidates? `design_efficiency` returns proven lower and upper
  bounds on its D-efficiency.

Terms, stated once. For candidate i with slopes J_i (one per parameter)
and reading error sigma_i, g_i = J_i / sigma_i. A design that puts the
fraction w_i of its N measurements at candidate i has the information
matrix N M(w), with the information per measurement

    M(w) = sum_i w_i g_i g_i^T,

and its "variance function" d_i(w) = g_i^T M(w)^-1 g_i, which is N
times the variance of the predicted reading at candidate i, in units
of that candidate's reading error squared. The D-efficiency of a design
with information per measurement M is

    eff = (det M / det M*)^(1/p),

with M* the information per measurement of the best weights and p the
number of parameters. A design of efficiency eff needs 1/eff times as
many measurements as the best one to make the determinant as large.

Facts used, each a short calculation:

- sum_i w_i d_i(w) = trace(M^-1 M) = p for every w, so max_i d_i >= p.
- At the best weights max_i d_i = p (the equivalence theorem of J.
  Kiefer and J. Wolfowitz, "The equivalence of two extremum problems",
  Canadian Journal of Mathematics 12, 363-366 (1960)).
- For any weights w, eff(w) >= p / max_i d_i(w). Proof: the
  eigenvalues l_k of M^-1 M* have det(M^-1 M*) = prod l_k and
  trace(M^-1 M*) = sum l_k = sum_i w*_i d_i(w) <= max_i d_i(w), so by
  the inequality of the arithmetic and geometric means
  (det M* / det M)^(1/p) <= max_i d_i(w) / p.

The bound needs no knowledge of the best design, so it certifies any
design, including the picks of `design`. Since every design with N
measurements on these candidates is one choice of weights (w_i = n_i
/ N), the best weights are at least as good as the best whole-number
design, so a lower bound on the efficiency also holds against the best
whole-number design (an upper bound does not).

The weights are found by alternating two steps: the multiplicative
update w_i <- w_i d_i(w) / p, and a "vertex exchange" that moves the
best amount of weight from the support point with the smallest d_i to
the candidate with the largest d_i (the determinant after the move is
a quadratic in the amount, by the matrix determinant lemma, and its
maximum over the allowed amounts, which include none, is taken). The
search stops when max_i d_i <= p (1 + tol), so the returned weights
have efficiency at least 1 / (1 + tol); the stopping test does not
rely on how the search got there.
"""
from __future__ import annotations

import numpy as np

from .model import Model, _settings
from .plan import _slogdets_with
from .stats import SINGULAR_MSG, check_sigmas, invert_information

__all__ = ["design_weights", "design_efficiency"]


def _scaled_rows(model, theta, candidates, sigmas):
    """The rows g_i = J_i / sigma_i (unscaled), the same rows divided by
    the root-mean-square of each column (unit-free; this multiplies every
    determinant by one constant and leaves every d_i unchanged), and the
    column scale. Refuses candidates that cannot identify the
    parameters, as `design` does."""
    if not isinstance(model, Model):
        raise ValueError("model must be a labplan Model")
    x = _settings(candidates)
    n, p = x.shape[0], model.n_params
    sig = check_sigmas(sigmas, n)
    w = np.ones(n) if sig is None else 1.0 / sig
    theta = np.asarray(theta, dtype=float).ravel()
    rows = model.jacobian(theta, x) * w[:, None]
    ok = n >= p and invert_information(rows.T @ rows,
                                       model.param_names)[0]
    if not ok:
        raise ValueError("even the full candidate list is not "
                         "identifiable: " + SINGULAR_MSG)
    scale = np.sqrt(np.mean(rows * rows, axis=0))
    return rows, rows / scale, scale


def _inverse_and_d(rs, w):
    m = rs.T @ (w[:, None] * rs)
    minv = np.linalg.inv(m)
    minv = 0.5 * (minv + minv.T)
    d = np.einsum("ij,jk,ik->i", rs, minv, rs)
    return minv, d


def _check_tol(tol, max_iter):
    try:
        t = float(tol)
    except (TypeError, ValueError):
        raise ValueError("tol must be a finite number > 0") from None
    if isinstance(tol, bool) or not (np.isfinite(t) and t > 0.0):
        raise ValueError("tol must be a finite number > 0")
    return t, _whole(max_iter, "max_iter", 1)


def _whole(value, name, least):
    """A whole number >= least, or a ValueError naming `name`."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = np.nan
    if isinstance(value, bool) or not np.isfinite(v) or v != int(v) \
            or v < least:
        raise ValueError(f"{name} must be a whole number of at least "
                         f"{least}; got {value!r}")
    return int(v)


def _best_weights(rs, tol, max_iter, n_exchange=50):
    """Weights with max_i d_i <= p (1 + tol); see the module docstring.
    Returns (weights, d, number of outer iterations)."""
    n, p = rs.shape
    w = np.full(n, 1.0 / n)
    for it in range(1, max_iter + 1):
        minv, d = _inverse_and_d(rs, w)
        if float(np.max(d)) <= p * (1.0 + tol):
            return w, d, it
        # multiplicative step; sum_i w_i d_i = p keeps the sum at 1
        w = w * d / p
        w = w / np.sum(w)
        minv, d = _inverse_and_d(rs, w)
        for _ in range(n_exchange):
            j = int(np.argmax(d))
            support = np.flatnonzero(w > 0.0)
            k = int(support[np.argmin(d[support])])
            if d[j] <= d[k]:
                break
            u_j = minv @ rs[j]
            d_jk = float(rs[k] @ u_j)
            # det after moving t from k to j, over det before:
            # (1 + t d_j)(1 - t d_k) + t^2 d_jk^2; its maximum
            den = 2.0 * (d[j] * d[k] - d_jk * d_jk)
            t = (d[j] - d[k]) / den if den > 0.0 else w[k]
            t = min(t, w[k])
            if not t > 0.0:
                break
            # the two rank-one updates of M^-1 and of every d_i
            c = rs @ u_j
            minv = minv - t * np.outer(u_j, u_j) / (1.0 + t * d[j])
            d = d - t * c * c / (1.0 + t * d[j])
            u_k = minv @ rs[k]
            keep = 1.0 - t * d[k]
            if not keep > 1e-12:
                break           # recomputed from scratch next round
            c = rs @ u_k
            minv = minv + t * np.outer(u_k, u_k) / keep
            d = d + t * c * c / keep
            w[j] += t
            w[k] = 0.0 if t == w[k] else w[k] - t
    minv, d = _inverse_and_d(rs, w)
    raise RuntimeError(
        f"design_weights did not reach tol = {tol:g} within max_iter = "
        f"{max_iter} rounds (the weights found have a D-efficiency of "
        f"at least {rs.shape[1] / float(np.max(d)):.6g}); raise max_iter "
        "or tol")


def _check_counts(counts, n):
    c = np.asarray(counts)
    if c.shape != (n,):
        raise ValueError(f"counts must be {n} whole numbers, one per "
                         f"candidate; got shape {c.shape}")
    if c.dtype == bool:
        raise ValueError("counts must be whole numbers, not True/False")
    cf = c.astype(float)
    if not np.all(np.isfinite(cf)) or np.any(cf != np.round(cf)) \
            or np.any(cf < 0.0):
        raise ValueError("counts must be whole numbers >= 0")
    if np.sum(cf) < 1:
        raise ValueError("counts must contain at least one measurement")
    return cf.astype(np.int64)


def _logdet(m):
    sign, val = np.linalg.slogdet(m)
    return float(val) if sign > 0 else -np.inf


def _round_to_counts(rs, w, n_total):
    """Whole numbers of measurements, summing to n_total, from the
    weights: floor(n_total w_i) first, then one measurement at a time
    where it increases the determinant most (the greedy rule of
    `design`, repeats allowed), then single moves of one measurement
    from one candidate to another while one increases the determinant
    (the exchange of `design`, repeats allowed)."""
    n, p = rs.shape
    counts = np.floor(n_total * w).astype(np.int64)
    eps = 1e-12 * float(np.max(np.sum(rs * rs, axis=1)))
    fs = eps * np.eye(p) + rs.T @ (counts[:, None] * rs)
    idx = np.arange(n)
    while int(np.sum(counts)) < n_total:
        _, logdets = _slogdets_with(fs, rs, idx)
        j = int(np.argmax(logdets))
        counts[j] += 1
        fs = fs + np.outer(rs[j], rs[j])
    f = rs.T @ (counts[:, None] * rs)
    cur = _logdet(f)
    for _ in range(100 * n_total):
        if not np.isfinite(cur):
            break
        finv = np.linalg.inv(f)
        dv = np.einsum("ij,jk,ik->i", rs, finv, rs)
        src = np.flatnonzero(counts > 0)
        cross = (rs[src] @ finv) @ rs.T           # (support, n)
        # det(F - g_i g_i^T + g_j g_j^T) / det F, by the matrix
        # determinant lemma
        ratio = (1.0 - dv[src])[:, None] * (1.0 + dv)[None, :] \
            + cross ** 2
        ratio[np.arange(src.size), src] = 1.0
        a, b = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        if not ratio[a, b] > 1.0 + 1e-10:
            break
        counts[src[a]] -= 1
        counts[b] += 1
        f = rs.T @ (counts[:, None] * rs)
        cur = _logdet(f)
    return counts


def _efficiency(rs, counts, m_best_logdet, d_best_max):
    """Bounds on the D-efficiency of the design with these counts, from
    the scaled rows; see the module docstring. A design that cannot
    identify the parameters (by the rule of `information`) gets 0."""
    p = rs.shape[1]
    total = float(np.sum(counts))
    m = rs.T @ ((counts / total)[:, None] * rs)
    ld = _logdet(m)
    if not np.isfinite(ld) or not invert_information(m, range(p))[0]:
        return 0.0, 0.0, np.inf
    ratio = float(np.exp((ld - m_best_logdet) / p))
    minv = np.linalg.inv(m)
    d_max = float(np.max(np.einsum("ij,jk,ik->i", rs, minv, rs)))
    low = max(p / d_max, ratio * p / d_best_max)
    high = min(1.0, ratio)
    return min(low, high), high, d_max


def design_weights(model: Model, theta, candidates, sigmas=None, *,
                   n_total=None, tol: float = 1e-6,
                   max_iter: int = 10000) -> dict:
    """The best share of the measurements for each candidate setting,
    repeats allowed (D-optimal weights; new in 0.6.0).

    model, theta, candidates, sigmas : as for `design`: the model, the
        working point, the (n, d) candidate settings and their expected
        reading errors (one number or one per candidate; without them,
        per unit reading error).
    n_total : optional whole number of measurements (at least the
        number of parameters). The result then also has `counts`.
    tol : the search stops when the largest d_i is at most
        p (1 + tol), which proves an efficiency of at least
        1 / (1 + tol) (default 1e-6).
    max_iter : largest number of rounds of the search; it raises a
        RuntimeError (with the efficiency reached) if that is not
        enough.

    Returns a dictionary:

    weights : (n,) shares that add up to 1, in candidate order. On a
        fine grid of candidates the share of one best setting can be
        split between neighboring candidates.
    max_variance : the largest d_i at these weights (p at the best
        weights; see the module docstring).
    efficiency_bound : p / max_variance, a proven lower bound on the
        D-efficiency of the weights (at least 1 / (1 + tol)).
    fisher, identifiable, condition_number, sigma, cov : as returned by
        `information`, for the information per measurement M(w). So
        `sigma` holds the error bars for ONE measurement spread by the
        weights, N measurements give sigma / sqrt(N), and the result
        can be passed to `repeats_for`, whose `r` is then the total
        number of measurements.
    n_iter : rounds of the search.

    With n_total, also:

    counts : (n,) whole numbers of measurements per candidate that add
        up to n_total: the weights times n_total rounded down, the rest
        added one at a time where it helps most, then improved by
        moving single measurements while that helps. The rounding is
        not proven best; its efficiency is bounded instead:
    counts_efficiency_low, counts_efficiency_high : proven bounds on the
        D-efficiency of `counts` against the best weights (see
        `design_efficiency`).

    Refuses, as `design` does, candidates that even all together
    cannot identify the parameters.
    """
    t, it_max = _check_tol(tol, max_iter)
    rows, rs, scale = _scaled_rows(model, theta, candidates, sigmas)
    n, p = rs.shape
    if n_total is not None:
        n_total = _whole(n_total, "n_total (at least the number of "
                         "parameters)", p)
    w, d, n_iter = _best_weights(rs, t, it_max)
    d_max = float(np.max(d))
    fisher = rows.T @ (w[:, None] * rows)
    ok, cond, cov, sigma = invert_information(fisher, model.param_names)
    # p / d_max cannot exceed 1 except by rounding (sum w d = p)
    out = {"weights": w, "max_variance": d_max,
           "efficiency_bound": min(1.0, p / d_max), "fisher": fisher,
           "identifiable": ok, "condition_number": cond, "sigma": sigma,
           "cov": cov, "n_iter": n_iter}
    if n_total is not None:
        counts = _round_to_counts(rs, w, n_total)
        best_ld = _logdet(rs.T @ (w[:, None] * rs))
        low, high, _ = _efficiency(rs, counts, best_ld, d_max)
        out["counts"] = counts
        out["counts_efficiency_low"] = low
        out["counts_efficiency_high"] = high
    return out


def design_efficiency(model: Model, theta, candidates, counts,
                      sigmas=None, *, tol: float = 1e-6,
                      max_iter: int = 10000) -> dict:
    """How close a design comes to the best one over the same
    candidates: proven bounds on its D-efficiency (new in 0.6.0).

    model, theta, candidates, sigmas : as for `design_weights`.
    counts : (n,) whole numbers, the number of measurements at each
        candidate. For the picks of `design`, use
        `np.bincount(result["indices"], minlength=len(candidates))`.
        A setting of your own that is not among the candidates can be
        appended to them (with its count); the comparison is then with
        the best design on the enlarged list.
    tol, max_iter : for the best weights, as in `design_weights`.

    Returns a dictionary:

    efficiency_low, efficiency_high : proven bounds on the D-efficiency
        (det M / det M*)^(1/p) of the design, with M its information per
        measurement and M* that of the best weights on these
        candidates: low is the larger of p / max_variance (which needs
        no knowledge of the best design) and the ratio to the computed
        best weights times their own bound; high is the ratio to the
        computed best weights (at most 1). Both are 0 for a design that
        cannot identify the parameters. A design with efficiency eff
        needs 1/eff times as many measurements as the best one to make
        the determinant as large.
    max_variance : the largest d_i of the design (p if it is a best
        design, larger otherwise).
    n_total : the number of measurements in the design.
    fisher, identifiable, condition_number, sigma, cov : as returned by
        `information` for the design with these counts (so it can be
        passed to `repeats_for`, whose `r` is then the number of
        repeats of the whole design).
    best : the result of `design_weights` for the same candidates.
    """
    rows, rs, _ = _scaled_rows(model, theta, candidates, sigmas)
    counts = _check_counts(counts, rs.shape[0])
    best = design_weights(model, theta, candidates, sigmas, tol=tol,
                          max_iter=max_iter)
    best_ld = _logdet(rs.T @ (best["weights"][:, None] * rs))
    low, high, d_max = _efficiency(rs, counts, best_ld,
                                   best["max_variance"])
    fisher = rows.T @ (counts[:, None] * rows)
    ok, cond, cov, sigma = invert_information(fisher, model.param_names)
    return {"efficiency_low": low, "efficiency_high": high,
            "max_variance": d_max, "n_total": int(np.sum(counts)),
            "fisher": fisher, "identifiable": ok,
            "condition_number": cond, "sigma": sigma, "cov": cov,
            "best": best}

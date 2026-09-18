"""Plan the measurement before taking it.

`information` answers, from the design alone, the question every
instrument session should start with: can these planned measurements
determine the parameters at all, and how small would the error bars
come out? It is not an approximation of the fit -- it is the same
(J^T W J) matrix the fit inverts, evaluated at your expected
parameters, so on data the model describes the promise is kept
exactly (the tests assert it).

`design` picks the most informative subset of candidate measurements
by the standard determinant criterion (D-optimal design; F.
Pukelsheim, Optimal Design of Experiments, SIAM (2006)): each greedy
pick most shrinks the joint parameter uncertainty. The greedy rule is
transparent and each step can only add information (the rank-one
determinant identity det(A + g g^T) = det(A)(1 + g^T A^-1 g)), but it
is a good-practice heuristic, not a proof of the globally best
subset.

`repeats_for` is a closed form, not a search: repeating a design r
times scales its covariance by exactly 1/r.
"""
from __future__ import annotations

import numpy as np

from .model import Model, _settings
from .stats import SINGULAR_MSG, check_sigmas, invert_information

__all__ = ["information", "design", "repeats_for"]


def information(model: Model, theta, x, sigmas=None):
    """Predicted error bars for a planned design.

    theta : the working point (your expected parameters -- a previous
        calibration, a datasheet, a cited set).
    x : (n, d) planned measurement settings.
    sigmas : expected 1-sigma reading error (scalar or (n,)). Without
        it the answer is per unit measurement error, and real error
        bars scale linearly with your sigma.

    Returns dict(fisher, identifiable, condition_number, sigma, cov):
    `sigma` maps each parameter to the error bar the weighted fit
    would report, or is None when the design cannot tell the
    parameters apart.
    """
    x = _settings(x)
    sig = check_sigmas(sigmas, x.shape[0])
    w = np.ones(x.shape[0]) if sig is None else 1.0 / sig
    theta = np.asarray(theta, dtype=float).ravel()
    jac = model.jacobian(theta, x) * w[:, None]
    fisher = jac.T @ jac
    identifiable, cond, cov, sigma = invert_information(
        fisher, model.param_names)
    if x.shape[0] < model.n_params:
        identifiable, cov, sigma = False, None, None
    return {"fisher": fisher, "identifiable": identifiable,
            "condition_number": cond, "sigma": sigma, "cov": cov}


def design(model: Model, theta, candidates, n_pick, sigmas=None):
    """Pick the most informative subset of candidate measurements.

    Greedy D-optimal selection over the candidate settings rows.
    Returns dict(indices, fisher, condition_number, sigma) with the
    chosen candidate indices in pick order. Refuses when even the
    full candidate list cannot identify the parameters.
    """
    x = _settings(candidates)
    n, p = x.shape[0], model.n_params
    n_pick = int(n_pick)
    if not p <= n_pick <= n:
        raise ValueError(f"n_pick must be between {p} (the number of "
                         f"parameters) and {n} (the number of "
                         "candidates)")
    sig = check_sigmas(sigmas, n)
    full = information(model, theta, x, sig)
    if not full["identifiable"]:
        raise ValueError("even the full candidate list is not "
                         "identifiable: " + SINGULAR_MSG)
    w = np.ones(n) if sig is None else 1.0 / sig
    theta = np.asarray(theta, dtype=float).ravel()
    rows = model.jacobian(theta, x) * w[:, None]
    # column-scaled (unit-free) greedy: identical scaling multiplies
    # every candidate determinant by the same constant, so the picks
    # are unchanged, while the tiny start-up regularizer stays
    # meaningful in every direction
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
    fisher = (fs - eps * np.eye(p)) * np.outer(scale, scale)
    _, cond, _, sigma = invert_information(fisher, model.param_names)
    return {"indices": list(chosen), "fisher": fisher,
            "condition_number": cond, "sigma": sigma}


def repeats_for(target_sigma, plan):
    """How many repeats of a planned design meet a target error bar?

    Exact closed form: r identical repeats of a design multiply its
    information by r, so every error bar shrinks by exactly
    1/sqrt(r). `target_sigma` is the largest acceptable error bar of
    any parameter, in that parameter's own units -- pass a dict
    {name: target} to set per-parameter targets instead.

    Returns (r, predicted) with `predicted` the per-parameter error
    bars at r repeats. Refuses a non-identifiable plan: no number of
    repeats can identify what one copy cannot.
    """
    if not plan.get("identifiable") or plan.get("sigma") is None:
        raise ValueError("the plan is not identifiable; no number of "
                         "repeats can identify what one copy of the "
                         "design cannot -- change the design")
    sigma = plan["sigma"]
    if isinstance(target_sigma, dict):
        ratios = []
        for name, t in target_sigma.items():
            if name not in sigma:
                raise ValueError(f"unknown parameter {name!r} in "
                                 "target_sigma")
            t = float(t)
            if not (np.isfinite(t) and t > 0.0):
                raise ValueError("targets must be positive")
            ratios.append(sigma[name] / t)
        worst = max(ratios)
    else:
        t = float(target_sigma)
        if not (np.isfinite(t) and t > 0.0):
            raise ValueError("target_sigma must be positive")
        worst = max(s / t for s in sigma.values())
    r = max(1, int(np.ceil(worst ** 2)))
    predicted = {k: v / np.sqrt(r) for k, v in sigma.items()}
    return r, predicted

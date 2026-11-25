"""Error bars of anything computed from the parameters.

A fit (or a plan) gives the parameters and their covariance. What a
user often needs next is the error bar of something computed from
them: the predicted reading at a new setting, a time constant
1 / (2 pi fc) from a fitted corner frequency, a ratio of two fitted
values. `propagate` gives it by the standard first-order rule
("propagation of uncertainty", the delta method):

    cov_out = J C J^T,

with C the parameter covariance and J the slopes of the computed
quantities with respect to the parameters, taken by the same
finite-difference routine the fit uses (`Model.numeric_jacobian`).
The rule is exact when the quantity is a straight-line (linear)
function of the parameters; otherwise it is the usual linear
approximation around the given parameter values, as good as the
quantity is close to linear within about one error bar. This is the
"law of propagation of uncertainty" of JCGM 100:2008, Evaluation of
measurement data -- Guide to the expression of uncertainty in
measurement, sections 5.1.2 and 5.2 (with covariances).

`propagate_mc` (since 0.7.0) answers the same question without the
straight-line approximation, by drawing the parameters at random
(the Monte Carlo method of JCGM 101:2008), so that the two can be
compared.
"""
from __future__ import annotations

import numpy as np

from .model import Model

__all__ = ["propagate", "propagate_mc"]

# smallest eigenvalue allowed for the correlation matrix of `cov`;
# rounding in a covariance computed by labplan stays far above it
_PSD_TOL = 1e-10


def _check_positive_semidefinite(cov, d):
    """Refuse a covariance that is not positive semidefinite.

    Such a matrix gives some combination of parameters a negative
    variance. Before 0.3.0 `propagate` then clipped the variance of
    the result to zero and returned an error bar of 0 (and a negative
    entry in `cov`). The test is made on the correlation matrix
    C_ij / sqrt(C_ii C_jj), whose eigenvalues do not depend on the
    units of the parameters."""
    ds = np.where(d > 0.0, d, 1.0)
    rho = cov / np.outer(ds, ds)
    low = float(np.min(np.linalg.eigvalsh(0.5 * (rho + rho.T))))
    if low < -_PSD_TOL:
        raise ValueError(
            "cov is not positive semidefinite (the smallest eigenvalue "
            f"of its correlation matrix is {low:.3g}): some combination "
            "of parameters would have a negative variance. Check that "
            "every correlation lies between -1 and 1, and copy "
            "covariances at full precision: rounding a strongly "
            "correlated matrix to a few digits can make it invalid")


def _check_inputs(func, theta, cov):
    """The checks shared by `propagate` and `propagate_mc`; returns
    theta and cov as arrays."""
    if not callable(func):
        raise ValueError("func must be a callable func(theta)")
    theta = np.asarray(theta, dtype=float).ravel()
    p = theta.size
    if p < 1 or not np.all(np.isfinite(theta)):
        raise ValueError("theta must be a non-empty array of finite "
                         "numbers")
    if cov is None:
        raise ValueError("cov is None: the fit or plan it came from "
                         "is not identifiable, so there are no error "
                         "bars to propagate")
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (p, p) or not np.all(np.isfinite(cov)):
        raise ValueError(f"cov must be a finite ({p}, {p}) matrix, "
                         "one row and column per parameter")
    d = np.sqrt(np.abs(np.diag(cov)))
    if np.any(np.diag(cov) < 0.0) or np.any(
            np.abs(cov - cov.T) > 1e-8 * np.outer(d, d)):
        raise ValueError("cov must be symmetric with a non-negative "
                         "diagonal")
    _check_positive_semidefinite(cov, d)

    return theta, cov


def propagate(func, theta, cov) -> dict:
    """Value and error bar of quantities computed from the parameters.

    func : callable func(theta) -> one number or a 1-D array of
        numbers computed from the parameter vector theta (arrays with
        more dimensions are refused).
    theta : parameter values (e.g. `FitResult.theta`, or the guess
        you planned at).
    cov : their covariance (e.g. `FitResult.cov`, or
        `information(...)["cov"]` for planned error bars).

    Returns dict(value, sigma, cov): the computed value(s), their
    1-sigma error bars and their covariance. `value` and `sigma` are
    plain floats when func returns one number, arrays otherwise.

    Refuses a `cov` that is not symmetric, has a negative diagonal or
    (since 0.3.0) is not positive semidefinite.
    """
    theta, cov = _check_inputs(func, theta, cov)
    p = theta.size
    raw = np.asarray(func(theta), dtype=float)
    if raw.ndim > 1:
        raise ValueError("func must return one number or a 1-D array "
                         f"of numbers; got an array of shape "
                         f"{raw.shape} (flatten it yourself if that "
                         "is what you mean)")
    scalar = raw.ndim == 0
    m = raw.size
    if m < 1:
        raise ValueError("func returned nothing")
    wrapped = Model("derived quantity",
                    lambda th, x: np.asarray(func(th), dtype=float),
                    tuple(f"p{i}" for i in range(p)),
                    "labplan.propagate internal wrapper")
    dummy = np.zeros((m, 1))
    value = wrapped.predict(theta, dummy)
    jac = wrapped.numeric_jacobian(theta, dummy)
    out = jac @ cov @ jac.T
    out = 0.5 * (out + out.T)
    sigma = np.sqrt(np.maximum(np.diag(out), 0.0))
    if scalar:
        return {"value": float(value[0]), "sigma": float(sigma[0]),
                "cov": out}
    return {"value": value, "sigma": sigma, "cov": out}


def _shape_of(raw):
    if raw.ndim > 1:
        raise ValueError("func must return one number or a 1-D array "
                         f"of numbers; got an array of shape "
                         f"{raw.shape} (flatten it yourself if that "
                         "is what you mean)")
    if raw.size < 1:
        raise ValueError("func returned nothing")
    return raw.ndim == 0, raw.size


def propagate_mc(func, theta, cov, *, n_draws: int = 100000, seed=0,
                 level: float = 0.95) -> dict:
    """Value, error bar and coverage interval of quantities computed
    from the parameters, by drawing the parameters at random (new in
    0.7.0).

    `propagate` is first order: exact when func is a straight-line
    function of the parameters, approximate otherwise. This function
    makes no such approximation. It draws n_draws parameter vectors
    from the Gaussian distribution with mean theta and covariance cov,
    evaluates func at each, and summarizes the results: the "Monte
    Carlo method" of JCGM 101:2008, Evaluation of measurement data --
    Supplement 1 to the GUM, Propagation of distributions using a
    Monte Carlo method (a multivariate Gaussian input as in its clause
    6.4.8; the probabilistically symmetric coverage interval of its
    clauses 3.15 and 5.3.3). Comparing the result with `propagate`
    shows whether the first-order error bar can be trusted.

    func, theta, cov : as for `propagate` (the same checks apply).
    n_draws : number of random parameter vectors (default 100000).
        The value has a random error of about sigma / sqrt(n_draws),
        and sigma one of about sigma / sqrt(2 n_draws) for a quantity
        close to Gaussian. It must be a whole number large enough to
        leave at least 10 draws beyond each end of the interval:
        n_draws (1 - level) / 2 >= 10.
    seed : seed of NumPy's default random generator (default 0, so
        that a result can be repeated exactly); None takes fresh
        randomness from the operating system.
    level : coverage probability of the interval (default 0.95).

    Returns a dictionary:

    value : the mean of func over the draws.
    sigma : the standard deviation of func over the draws (divisor
        n_draws - 1).
    cov : their covariance (for several numbers).
    low, high : the interval that holds each number with probability
        `level`, with equal probability (1 - level) / 2 below and
        above: the (1 - level)/2 and (1 + level)/2 quantiles of the
        draws (NumPy's default, linear interpolation between sorted
        values).
    n_draws, seed, level : the inputs, for the record.

    value, sigma, low and high are plain floats when func returns one
    number, arrays otherwise.

    The draws are theta + L z, with z independent standard Gaussian
    numbers and L L^T = cov (L from the eigenvectors and eigenvalues
    of cov, so that a parameter held fixed, with zero variance, is
    allowed). func is called once per draw with a 1-D parameter
    vector, so 100000 draws of a function that takes 5 microseconds
    take about half a second; every result must be finite.
    """
    theta, cov = _check_inputs(func, theta, cov)
    try:
        n_f = float(n_draws)
    except (TypeError, ValueError):
        n_f = np.nan
    if isinstance(n_draws, bool) or not np.isfinite(n_f) \
            or n_f != int(n_f) or n_f < 1:
        raise ValueError(f"n_draws must be a whole number; got "
                         f"{n_draws!r}")
    n_draws = int(n_f)
    try:
        lev = float(level)
    except (TypeError, ValueError):
        lev = np.nan
    if isinstance(level, bool) or not 0.0 < lev < 1.0:
        raise ValueError(f"level must lie between 0 and 1; got "
                         f"{level!r}")
    need = int(np.ceil(20.0 / (1.0 - lev)))
    if n_draws < need:
        raise ValueError(f"n_draws = {n_draws} leaves fewer than 10 "
                         f"draws beyond each end of a {lev:g} interval; "
                         f"use at least {need}")
    if seed is not None and (isinstance(seed, bool)
                             or not isinstance(seed, (int, np.integer))
                             or seed < 0):
        raise ValueError("seed must be a whole number >= 0, or None")

    first = np.asarray(func(theta), dtype=float)
    scalar, m = _shape_of(first)
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    root = vecs * np.sqrt(np.maximum(vals, 0.0))[None, :]
    rng = np.random.default_rng(seed)
    draws = theta[None, :] + rng.standard_normal((n_draws, theta.size)) \
        @ root.T
    out = np.empty((n_draws, m))
    for i in range(n_draws):
        row = np.asarray(func(draws[i]), dtype=float)
        if row.size != m or row.ndim > 1:
            raise ValueError(f"func returned {row.size} numbers at "
                             f"draw {i}, but {m} at theta")
        out[i] = row.ravel()
    if not np.all(np.isfinite(out)):
        i = int(np.argmax(~np.all(np.isfinite(out), axis=1)))
        raise ValueError(
            f"func is not finite at draw {i} (parameters "
            f"{np.array2string(draws[i], precision=6)}): the "
            "parameters within their error bars reach a region where "
            "the quantity is not defined, so it has no finite mean or "
            "error bar")
    value = np.mean(out, axis=0)
    out_cov = np.atleast_2d(np.cov(out, rowvar=False))
    sigma = np.sqrt(np.diag(out_cov))
    low = np.quantile(out, 0.5 * (1.0 - lev), axis=0)
    high = np.quantile(out, 0.5 * (1.0 + lev), axis=0)
    result = {"value": value, "sigma": sigma, "cov": out_cov,
              "low": low, "high": high, "n_draws": n_draws,
              "seed": seed, "level": lev}
    if scalar:
        for key in ("value", "sigma", "low", "high"):
            result[key] = float(result[key][0])
    return result

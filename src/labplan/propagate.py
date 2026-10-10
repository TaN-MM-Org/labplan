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
"""
from __future__ import annotations

import numpy as np

from .model import Model

__all__ = ["propagate"]

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

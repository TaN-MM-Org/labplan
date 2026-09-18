"""Fit the model to measured data, with error bars that mean it.

Levenberg-Marquardt weighted least squares in pure NumPy: no
dependency beyond the one this package already has. With measurement
errors supplied, the parameter covariance is the exact known-noise
result (J^T W J)^-1 and a chi-squared consistency check is reported;
without them, the error bars are scaled from the residual scatter,
which needs at least one spare measurement -- stated, not hidden.

A design that cannot tell the parameters apart is refused with an
explanation, never silently pseudo-inverted; a fit that does not
converge raises instead of returning the last iterate with a
well-formatted covariance.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from .model import Model, _settings
from .stats import SINGULAR_MSG, check_sigmas, invert_information

__all__ = ["FitResult", "fit"]


@dataclasses.dataclass
class FitResult:
    """Result of `fit`.

    values : fitted value of each parameter, by name.
    sigma : 1-sigma uncertainty of each parameter, by name.
    theta, cov : the same as arrays, in `param_names` order.
    chi2, chi2_dof : goodness of fit (chi2 is None when no
        measurement errors were given).
    condition_number : of the unit-free information matrix.
    data_digest : sha256 of the (x, y, sigma) arrays -- the exact
        data this result answers for, recorded for the audit trail.
    model_name, reference : carried from the model.
    """

    values: dict
    sigma: dict
    theta: np.ndarray
    cov: np.ndarray
    chi2: float
    chi2_dof: int
    n_points: int
    condition_number: float
    n_iter: int
    model_name: str
    reference: str
    data_digest: str


def data_digest(x, y=None, sigmas=None):
    """sha256 over the byte content of the data arrays, so a report
    can say exactly which data it answers for."""
    h = hashlib.sha256()
    for a in (x, y, sigmas):
        if a is None:
            h.update(b"\x00none\x00")
        else:
            arr = np.ascontiguousarray(np.asarray(a, dtype=float))
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def fit(model: Model, x, y, theta0, sigmas=None, max_iter=200,
        tol=1e-12):
    """Weighted least-squares fit of `model` to measured data.

    model : a `Model`.
    x : (n, d) measurement settings (or (n,) for one setting).
    y : (n,) measured readings.
    theta0 : starting parameter vector, in `param_names` order.
    sigmas : optional 1-sigma measurement errors (scalar or (n,)).

    Returns a `FitResult`. Refuses non-identifiable designs, too few
    points, and non-convergence, each with an explanation.
    """
    x = _settings(x)
    y = np.asarray(y, dtype=float).ravel()
    n, p = x.shape[0], model.n_params
    if y.size != n:
        raise ValueError("need one measured reading per settings row")
    if not np.all(np.isfinite(y)):
        raise ValueError("measured readings must be finite")
    sig = check_sigmas(sigmas, n)
    if sig is None and n < p + 1:
        raise ValueError(f"{n} measurements cannot determine {p} "
                         "parameters and an error scale; add points "
                         "or supply sigmas")
    if n < p:
        raise ValueError(f"{n} measurements cannot determine {p} "
                         "parameters")
    w = np.ones(n) if sig is None else 1.0 / sig
    th = np.asarray(theta0, dtype=float).ravel()
    if th.size != p or not np.all(np.isfinite(th)):
        raise ValueError(f"theta0 must be {p} finite starting values")

    def cost(t):
        r = (model.predict(t, x) - y) * w
        return r, float(r @ r)

    r, c = cost(th)
    mu = 1e-3
    converged = False
    it = 0
    c_old = c
    for it in range(1, max_iter + 1):
        jac = model.jacobian(th, x) * w[:, None]
        jtj = jac.T @ jac
        g = jac.T @ r
        stepped = False
        for _ in range(60):
            try:
                dth = np.linalg.solve(
                    jtj + mu * np.diag(np.maximum(np.diag(jtj),
                                                  1e-30)), -g)
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
            try:
                r_new, c_new = cost(th + dth)
            except ValueError:
                mu *= 10.0
                continue
            if c_new <= c:
                th, r, c_old, c = th + dth, r_new, c, c_new
                mu = max(mu / 10.0, 1e-15)
                stepped = True
                break
            mu *= 10.0
        if not stepped:
            converged = True
            break
        if abs(c_old - c) <= tol * (1.0 + c) and \
                float(np.max(np.abs(dth))) \
                <= 1e-10 * (1.0 + float(np.max(np.abs(th)))):
            converged = True
            break
    if not converged:
        raise RuntimeError("fit did not converge; check the starting "
                           "values and that the readings actually "
                           "vary with the parameters")

    jac = model.jacobian(th, x) * w[:, None]
    fisher = jac.T @ jac
    ok, cond, cov, _ = invert_information(fisher, model.param_names)
    if not ok:
        raise ValueError(SINGULAR_MSG)
    if sig is None:
        cov = cov * (c / (n - p))
        chi2, chi2_dof = None, None
    else:
        chi2, chi2_dof = c, n - p
    err = np.sqrt(np.diag(cov))
    return FitResult(
        values={k: float(v) for k, v in zip(model.param_names, th)},
        sigma={k: float(s) for k, s in zip(model.param_names, err)},
        theta=th.copy(), cov=cov, chi2=chi2, chi2_dof=chi2_dof,
        n_points=n, condition_number=cond, n_iter=it,
        model_name=model.name, reference=model.reference,
        data_digest=data_digest(x, y, sig))

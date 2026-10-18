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

`goodness_of_fit` (since 0.3.0) turns the chi-squared of a fit into
the probability of a value at least as large (and one at most as
large) when the model and the stated reading errors are right.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from .model import Model, _settings
from .stats import (SINGULAR_MSG, check_sigmas, gamma_regularized,
                    invert_information)

__all__ = ["FitResult", "fit", "goodness_of_fit"]


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
    model_name, reference, units : carried from the model (units
        since 0.2.0).
    x_adjusted : for a fit by `fit_xy` (since 0.5.0), the fitted true
        settings, an (n, d) array; None for `fit`.
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
    units: object = ""
    x_adjusted: object = None


def data_digest(x, y=None, sigmas=None, x_sigmas=None) -> str:
    """sha256 over the byte content of the data arrays, so a report
    can say exactly which data it answers for. `x_sigmas` (the errors
    of the settings, for `fit_xy`; since 0.5.0) is included only when
    given, so the digest of every other fit is unchanged."""
    h = hashlib.sha256()
    arrays = (x, y, sigmas) if x_sigmas is None \
        else (x, y, sigmas, x_sigmas)
    for a in arrays:
        if a is None:
            h.update(b"\x00none\x00")
        else:
            arr = np.ascontiguousarray(np.asarray(a, dtype=float))
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def _cost_unit(c, sig, n, p):
    """The size of one unit of the cost in the stopping test
    |c_old - c| <= tol * (unit + c).

    With reading errors the cost is chi-squared, which has no units,
    and the unit is 1 (as in every version). Without them the cost is
    the sum of squared residuals in the readings' own units, so a
    fixed 1 is a statement about those units: for readings of order
    1e-12 (a current in amperes, say) every change of the cost is
    below tol * 1 and 0.2.0 stopped after the first, still damped,
    step. Since 0.3.0 the unit is then the current residual variance
    c / (n - p), the scale the error bars are computed on, capped at
    1 so that no fit stops earlier than before."""
    if sig is not None:
        return 1.0
    return min(1.0, c / (n - p))


def fit(model: Model, x, y, theta0, sigmas=None, max_iter: int = 200,
        tol: float = 1e-12) -> FitResult:
    """Weighted least-squares fit of `model` to measured data.

    model : a `Model`.
    x : (n, d) measurement settings (or (n,) for one setting).
    y : (n,) measured readings.
    theta0 : starting parameter vector, in `param_names` order.
    sigmas : optional 1-sigma measurement errors (scalar or (n,)).
    max_iter : largest number of Levenberg-Marquardt steps.
    tol : the search stops when a step changes the cost by at most
        tol * (unit + cost) and moves no parameter by more than
        1e-10 * (1 + largest |parameter|). The cost is chi-squared
        with `sigmas` (unit 1); without them it is the sum of squared
        residuals and the unit is the residual variance, at most 1,
        so the test does not depend on the units of the readings
        (since 0.3.0).

    Returns a `FitResult`. Refuses non-identifiable designs, too few
    points, and non-convergence, each with an explanation.
    """
    if isinstance(max_iter, bool) or int(max_iter) != max_iter \
            or max_iter < 1:
        raise ValueError("max_iter must be a whole number >= 1")
    if not (np.isfinite(tol) and tol >= 0.0):
        raise ValueError("tol must be finite and >= 0")
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
        if abs(c_old - c) <= tol * (_cost_unit(c, sig, n, p) + c) and \
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
        data_digest=data_digest(x, y, sig), units=model.units)


def goodness_of_fit(result: FitResult | None = None, *,
                    chi2: float | None = None,
                    dof: int | None = None) -> dict:
    """Are the readings consistent with the model and their stated
    errors? The chi-squared test of a fit.

    Pass a `FitResult` from a fit with `sigmas`, or the numbers
    `chi2=` and `dof=` (degrees of freedom: readings minus fitted
    parameters) from anywhere else.

    Returns a dictionary:

    chi2, dof : the inputs.
    reduced_chi2 : chi2 / dof; near 1 when model and errors agree.
    p_value : probability that chi2 comes out at least this large
        when the model is right and the readings scatter with the
        stated Gaussian errors, Q(dof/2, chi2/2). A very small value
        (say below 0.001) means the model, or the stated errors, do
        not describe the data.
    p_value_low : probability that chi2 comes out at most this large,
        P(dof/2, chi2/2) = 1 - p_value. A very small value means the
        readings agree with the model better than their stated errors
        allow: the errors are probably overstated.
    birge_ratio : sqrt(chi2 / dof) (R. T. Birge, Phys. Rev. 40, 207
        (1932)): the factor by which the stated reading errors would
        have to be scaled to make chi2 equal dof.

    The chi-squared distribution with dof degrees of freedom is exact
    for a model that is a straight-line function of its parameters
    with independent Gaussian reading errors, and the usual
    approximation for other models. The probabilities are computed
    with the regularized incomplete gamma functions P and Q (see
    `labplan.stats.gamma_regularized`), with small values kept to
    full relative precision.
    """
    if result is not None:
        if chi2 is not None or dof is not None:
            raise ValueError("pass either a FitResult or chi2= and "
                             "dof=, not both")
        if not isinstance(result, FitResult):
            raise ValueError("result must be a FitResult from `fit`")
        if result.chi2 is None:
            raise ValueError(
                "this fit has no chi-squared: it was made without "
                "reading errors (`sigmas`), so its error bars were "
                "scaled from the residual scatter, and the scatter "
                "cannot also test itself. Fit again with the reading "
                "errors you know, or pass chi2= and dof=")
        chi2, dof = result.chi2, result.chi2_dof
    elif chi2 is None or dof is None:
        raise ValueError("pass a FitResult, or both chi2= and dof=")
    chi2 = float(chi2)
    if not (np.isfinite(chi2) and chi2 >= 0.0):
        raise ValueError("chi2 must be finite and >= 0")
    if isinstance(dof, bool) or not float(dof).is_integer():
        raise ValueError("dof must be a whole number")
    dof = int(dof)
    if dof < 1:
        raise ValueError(
            f"dof = {dof}: with no more readings than parameters the "
            "fit passes through the data whatever the errors are, so "
            "chi-squared cannot test anything. Add readings")
    p_low, p_high = gamma_regularized(0.5 * dof, 0.5 * chi2)
    return {"chi2": chi2, "dof": dof, "reduced_chi2": chi2 / dof,
            "p_value": p_high, "p_value_low": p_low,
            "birge_ratio": float(np.sqrt(chi2 / dof))}

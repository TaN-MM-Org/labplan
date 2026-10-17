"""Read a calibration curve backwards: from a new reading to the
setting that produced it, with its error bar.

After a calibration fit, the everyday question is the inverse one: an
unknown sample gives reading y0; which setting x0 (a temperature, a
concentration, a pressure) does that correspond to, and how sure is
it? `inverse_predict` solves f(theta, x0) = y0 for x0 inside a range
you name (the bracket), and gives the error bar of x0 from both
sources of uncertainty: the error bar of the new reading itself and
the covariance of the fitted parameters.

The error bar is the first-order law of propagation of uncertainty
(JCGM 100:2008, Evaluation of measurement data -- Guide to the
expression of uncertainty in measurement, section 5.2) applied to the
implicit function x0(y0, theta): with s = df/dx and a = df/dtheta at
x0,

    u(x0)^2 = (u(y0)^2 + a^T C a) / s^2.

For a straight line y = a + b x this is, term by term, the formula of
the NIST/SEMATECH e-Handbook of Statistical Methods, section
2.3.6.7.3, "Comparison of check standard analysis and propagation of
error" (https://itl.nist.gov/div898/handbook/mpc/section3/mpc3673.htm):
u^2 = s_Y'^2 / b^2 + s_a^2 / b^2 + (Y' - a)^2 s_b^2 / b^4
+ 2 (Y' - a) s_ab / b^3. Several readings share the parameter
uncertainty, so their x0 values are correlated; the full covariance
is returned.

What it refuses, because the answer would not be reliable: a reading
that the model does not reach inside the bracket, or reaches more
than once there (a peak, an oscillation: narrow the bracket); a
model that is flat at the solution; and a solution where the slope
df/dx changes by more than `max_slope_change` (default 10 %) within
one error bar of x0, where the first-order error bar is no longer a
good description.
"""
from __future__ import annotations

import math

import numpy as np

from .fit import FitResult
from .model import Model
from .propagate import _check_positive_semidefinite

__all__ = ["inverse_predict"]


def _parameters(model, result, theta, cov):
    if result is not None:
        if theta is not None or cov is not None:
            raise ValueError("pass either a FitResult or theta= and "
                             "cov=, not both")
        if not isinstance(result, FitResult):
            raise ValueError("result must be a FitResult from `fit` "
                             "(or pass theta= and cov=)")
        if result.model_name != model.name:
            raise ValueError(f"the result is a fit of model "
                             f"{result.model_name!r}, not of "
                             f"{model.name!r}")
        theta, cov = result.theta, result.cov
    elif theta is None:
        raise ValueError("pass a FitResult, or both theta= and cov= "
                         "(for example from `information` to plan)")
    theta = np.asarray(theta, dtype=float).ravel()
    p = model.n_params
    if theta.size != p or not np.all(np.isfinite(theta)):
        raise ValueError(f"theta must be {p} finite numbers, one per "
                         "parameter")
    if cov is None:
        raise ValueError("cov is None: the fit or plan it came from is "
                         "not identifiable, so there are no error bars "
                         "to propagate")
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (p, p) or not np.all(np.isfinite(cov)):
        raise ValueError(f"cov must be a finite ({p}, {p}) matrix")
    d = np.sqrt(np.abs(np.diag(cov)))
    if np.any(np.diag(cov) < 0.0) or np.any(
            np.abs(cov - cov.T) > 1e-8 * np.outer(d, d)):
        raise ValueError("cov must be symmetric with a non-negative "
                         "diagonal")
    _check_positive_semidefinite(cov, d)
    return theta, cov


def _root_sites(g):
    """Where g (the model minus the reading on the scan grid) is zero
    or changes sign: a list of ("zero", first, last) runs of exact
    zeros and ("cross", k) sign changes between points k and k + 1."""
    sites = []
    k, n = 0, g.size
    while k < n:
        if g[k] == 0.0:
            first = k
            while k + 1 < n and g[k + 1] == 0.0:
                k += 1
            sites.append(("zero", first, k))
        elif k + 1 < n and g[k + 1] != 0.0 and \
                (g[k] > 0.0) != (g[k + 1] > 0.0):
            sites.append(("cross", k))
        k += 1
    return sites


def _bisect(func, lo, hi, g_lo):
    """Bisection to the last representable digit; func(lo) has the
    sign of g_lo and func(hi) the opposite sign."""
    best_x, best_g = lo, abs(g_lo)
    for _ in range(2200):
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            break
        gm = func(mid)
        if abs(gm) < best_g:
            best_x, best_g = mid, abs(gm)
        if gm == 0.0:
            return mid
        if (gm > 0.0) == (g_lo > 0.0):
            lo, g_lo = mid, gm
        else:
            hi = mid
    g_hi = abs(func(hi))
    return hi if g_hi < best_g else best_x


def _dfdx(model, theta, x0):
    """df/dx at the setting x0, by the same checked central
    differences `Model.numeric_jacobian` uses for parameters."""
    wrapped = Model("setting slope",
                    lambda xv, _: model.predict(theta, xv.reshape(1, 1)),
                    ("x",), "labplan.inverse_predict internal wrapper")
    return float(wrapped.numeric_jacobian([x0], np.zeros((1, 1)))[0, 0])


def inverse_predict(model: Model, result: FitResult | None = None,
                    reading=None, bracket=None, reading_sigma=0.0, *,
                    theta=None, cov=None, n_scan: int = 1001,
                    max_slope_change: float = 0.1) -> dict:
    """The setting x0 at which the model predicts `reading`, with its
    error bar (calibration read backwards; new in 0.4.0).

    model : the calibrated `Model`; its settings must be one number
        per measurement (an (n, 1) array).
    result : the `FitResult` of the calibration fit. Instead, pass
        theta= and cov= (for example the guess and
        `information(...)["cov"]`, to plan how well an unknown will be
        determined before calibrating).
    reading : the new reading y0 (one number), or a 1-D array of
        readings of several unknowns.
    bracket : (low, high), the range of settings to search: normally
        the calibrated range. The model must reach the reading exactly
        once inside it.
    reading_sigma : the 1-sigma error of the new reading (one number,
        or one per reading); 0 (the default) counts only the
        calibration's uncertainty. For the mean of m repeated readings
        with scatter sigma use sigma / sqrt(m).
    n_scan : number of evenly spaced settings at which the bracket is
        scanned for sign changes before the root is refined by
        bisection (default 1001). Two crossings closer together than
        the scan spacing are not seen; narrow the bracket if the model
        can turn back that quickly.
    max_slope_change : refuse when the slope df/dx at x0 - u and
        x0 + u (u the error bar of x0, both points kept inside the
        bracket) differs from the slope at x0 by more than this
        fraction (default 0.1). In the tests, on a saturating
        response, the scatter of seeded simulations was 1.001, 1.014
        and 1.32 times the first-order error bar where the slope
        changed by 1 %, 10.5 % and 65 %.

    Also refuses a model that is flat at x0 (its slope there would
    change the prediction across the whole bracket by at most 1e-8 of
    the model's largest value in the bracket), a FitResult of a
    different model (by name), and a covariance that is not
    symmetric and positive semidefinite (as `propagate` does).

    Returns a dictionary: value (x0), sigma (its 1-sigma error bar),
    cov (covariance of the x0 values), slope (df/dx at x0),
    sigma_from_reading (the part from the reading's error,
    u(y0) / |slope|), sigma_from_parameters (the part from the
    calibration, sqrt(a^T C a) / |slope|; sigma is the root sum of
    squares of the two) and slope_change (the largest relative change
    of the slope within one error bar). value, sigma, slope and the
    two parts are plain floats for one reading and arrays otherwise.
    """
    if not isinstance(model, Model):
        raise ValueError("model must be a labplan Model")
    theta, cov = _parameters(model, result, theta, cov)
    if reading is None:
        raise ValueError("give the new reading, reading=...")
    y = np.asarray(reading, dtype=float)
    if y.ndim > 1:
        raise ValueError("reading must be one number or a 1-D array")
    scalar = y.ndim == 0
    y = y.ravel()
    if y.size < 1 or not np.all(np.isfinite(y)):
        raise ValueError("readings must be finite numbers")
    u = np.asarray(reading_sigma, dtype=float)
    if u.size == 1:
        u = np.full(y.size, float(u.reshape(())))
    elif u.shape != y.shape:
        raise ValueError(f"reading_sigma must be one number or {y.size} "
                         "numbers, one per reading")
    if not np.all(np.isfinite(u)) or np.any(u < 0.0):
        raise ValueError("reading_sigma must be finite and >= 0")
    if bracket is None:
        raise ValueError("give the range of settings to search, "
                         "bracket=(low, high)")
    try:
        lo, hi = (float(v) for v in bracket)
    except (TypeError, ValueError):
        raise ValueError("bracket must be two numbers (low, high)") \
            from None
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError("bracket must be two finite numbers with "
                         "low < high")
    if isinstance(n_scan, bool) or int(n_scan) != n_scan or n_scan < 3:
        raise ValueError("n_scan must be a whole number >= 3")
    if not (math.isfinite(max_slope_change) and max_slope_change > 0.0):
        raise ValueError("max_slope_change must be finite and > 0")

    grid = np.linspace(lo, hi, int(n_scan))
    f_grid = model.predict(theta, grid[:, None])

    def g_at(xv, yv):
        return float(model.predict(theta, np.array([[xv]]))[0]) - yv

    x0 = np.empty(y.size)
    for i, yv in enumerate(y):
        g = f_grid - yv
        sites = _root_sites(g)
        if not sites:
            raise ValueError(
                f"the model does not reach the reading {yv:.6g} for "
                f"settings between {lo:.6g} and {hi:.6g} (there it runs "
                f"from {f_grid.min():.6g} to {f_grid.max():.6g}); "
                "widen the bracket, or check the reading")
        if len(sites) > 1:
            where = ", ".join(
                f"near {grid[s[1]]:.6g}" for s in sites[:4])
            raise ValueError(
                f"the model reaches the reading {yv:.6g} at "
                f"{len(sites)} settings in the bracket ({where}"
                f"{', ...' if len(sites) > 4 else ''}): the reading "
                "does not tell them apart. Narrow the bracket to the "
                "branch you calibrated")
        site = sites[0]
        if site[0] == "zero":
            if site[2] > site[1]:
                raise ValueError(
                    f"the model equals the reading {yv:.6g} on a whole "
                    f"range of settings ({grid[site[1]]:.6g} to "
                    f"{grid[site[2]]:.6g}): it is flat there and the "
                    "reading cannot tell the settings apart")
            x0[i] = grid[site[1]]
        else:
            k = site[1]
            x0[i] = _bisect(lambda xv: g_at(xv, yv), grid[k],
                            grid[k + 1], g[k])

    slope = np.array([_dfdx(model, theta, xv) for xv in x0])
    # flat: across the whole bracket this slope would change the
    # prediction by at most 1e-8 of the model's size there (rounding)
    flat = np.abs(slope) * (hi - lo) \
        <= 1e-8 * float(np.max(np.abs(f_grid)))
    if np.any(flat):
        bad = float(x0[int(np.argmax(flat))])
        raise ValueError(f"the model is flat at the solution "
                         f"x0 = {bad:.6g} (zero slope): the reading "
                         "does not determine the setting there")
    a = model.jacobian(theta, x0[:, None])
    par_cov = a @ cov @ a.T
    par_cov = 0.5 * (par_cov + par_cov.T)
    out_cov = (par_cov + np.diag(u * u)) / np.outer(slope, slope)
    sigma = np.sqrt(np.maximum(np.diag(out_cov), 0.0))
    from_reading = u / np.abs(slope)
    from_params = np.sqrt(np.maximum(np.diag(par_cov), 0.0)) \
        / np.abs(slope)

    change = np.zeros(y.size)
    for i, (xv, sv) in enumerate(zip(x0, sigma)):
        if sv == 0.0:
            continue
        for xe in (min(xv + sv, hi), max(xv - sv, lo)):
            if xe != xv:
                change[i] = max(change[i],
                                abs(_dfdx(model, theta, xe) / slope[i]
                                    - 1.0))
        if change[i] > max_slope_change:
            raise ValueError(
                f"the slope of the model changes by "
                f"{100 * change[i]:.3g} % within one error bar of "
                f"x0 = {xv:.6g} (+/- {sv:.3g}): the first-order error "
                "bar is not reliable there (the curve bends, or "
                "flattens, within the uncertainty). Reduce the "
                "uncertainty (repeat the reading, calibrate with more "
                "points) or work where the curve is steeper; "
                "max_slope_change sets this limit")

    if scalar:
        return {"value": float(x0[0]), "sigma": float(sigma[0]),
                "cov": out_cov, "slope": float(slope[0]),
                "sigma_from_reading": float(from_reading[0]),
                "sigma_from_parameters": float(from_params[0]),
                "slope_change": float(change[0])}
    return {"value": x0, "sigma": sigma, "cov": out_cov,
            "slope": slope, "sigma_from_reading": from_reading,
            "sigma_from_parameters": from_params,
            "slope_change": change}

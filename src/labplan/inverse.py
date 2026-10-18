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
                    max_slope_change: float = 0.1,
                    full_cov: bool = True) -> dict:
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

    full_cov : True (the default) returns the full covariance `cov` of
        the x0 values of several readings; False returns cov=None and
        computes only the error bars, so that memory and time grow
        with the number of readings, not with its square (new in
        0.5.0; use it to convert long series of readings).

    Several readings are solved together (since 0.5.0): the model is
    evaluated at the settings of all of them in one call, at each
    step of the scan, the bisection and the slopes. For one reading
    the steps are those of 0.4.0; for many, results agree with
    solving them one at a time to within the rounding of the model
    (the tests compare with the 0.4.0 loop). For 100 to 1000 readings
    it was 54 to 63 times faster than 0.4.0 in the measurements of the
    0.5.0 release notes.

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

    if not isinstance(full_cov, bool):
        raise ValueError("full_cov must be True or False")

    grid = np.linspace(lo, hi, int(n_scan))
    f_grid = model.predict(theta, grid[:, None])
    x0 = _roots(model, theta, grid, f_grid, y, lo, hi)

    slope = _dfdx_many(model, theta, x0)
    # flat: across the whole bracket this slope would change the
    # prediction by at most 1e-8 of the model's size there (the
    # resolution of the numerical slopes)
    flat = np.abs(slope) * (hi - lo) \
        <= 1e-8 * float(np.max(np.abs(f_grid)))
    if np.any(flat):
        bad = float(x0[int(np.argmax(flat))])
        raise ValueError(f"the model is flat at the solution "
                         f"x0 = {bad:.6g} (zero slope): the reading "
                         "does not determine the setting there")
    a = model.jacobian(theta, x0[:, None])
    if full_cov:
        par_cov = a @ cov @ a.T
        par_cov = 0.5 * (par_cov + par_cov.T)
        out_cov = (par_cov + np.diag(u * u)) / np.outer(slope, slope)
        sigma = np.sqrt(np.maximum(np.diag(out_cov), 0.0))
        par_var = np.diag(par_cov)
    else:
        # only the diagonal: memory and time grow like the number of
        # readings, not its square (since 0.5.0)
        par_var = np.einsum("ij,jk,ik->i", a, cov, a)
        out_cov = None
        sigma = np.sqrt(np.maximum((par_var + u * u) / (slope * slope),
                                   0.0))
    from_reading = u / np.abs(slope)
    from_params = np.sqrt(np.maximum(par_var, 0.0)) / np.abs(slope)

    change = _slope_changes(model, theta, x0, sigma, slope, lo, hi)
    over = change > max_slope_change
    if np.any(over):
        i = int(np.argmax(over))
        raise ValueError(
            f"the slope of the model changes by "
            f"{100 * change[i]:.3g} % within one error bar of "
            f"x0 = {x0[i]:.6g} (+/- {sigma[i]:.3g}): the first-order "
            "error bar is not reliable there (the curve bends, or "
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


# grid values per block when all readings are scanned at once
_SCAN_BLOCK = 2 ** 21


def _site_error(g, grid, f_grid, yv, lo, hi):
    """The refusal of 0.4.0 for a reading whose scan row g does not
    have exactly one isolated root site; None when it has one."""
    sites = _root_sites(g)
    if not sites:
        return (f"the model does not reach the reading {yv:.6g} for "
                f"settings between {lo:.6g} and {hi:.6g} (there it runs "
                f"from {f_grid.min():.6g} to {f_grid.max():.6g}); "
                "widen the bracket, or check the reading")
    if len(sites) > 1:
        where = ", ".join(f"near {grid[s[1]]:.6g}" for s in sites[:4])
        return (f"the model reaches the reading {yv:.6g} at "
                f"{len(sites)} settings in the bracket ({where}"
                f"{', ...' if len(sites) > 4 else ''}): the reading "
                "does not tell them apart. Narrow the bracket to the "
                "branch you calibrated")
    site = sites[0]
    if site[0] == "zero" and site[2] > site[1]:
        return (f"the model equals the reading {yv:.6g} on a whole "
                f"range of settings ({grid[site[1]]:.6g} to "
                f"{grid[site[2]]:.6g}): it is flat there and the "
                "reading cannot tell the settings apart")
    return None


def _roots(model, theta, grid, f_grid, y, lo, hi):
    """x0 for every reading: the scan for sign changes and the
    bisection of 0.4.0, done for all readings at once (since 0.5.0).

    The scan compares the same numbers f_grid - y as before, so the
    sites found are the same; readings without exactly one isolated
    site get the 0.4.0 refusal, in reading order. The bisection makes
    the same steps for every reading, but evaluates the model at the
    midpoints of all readings in one call, so a midpoint at which the
    model is within rounding of the reading can be judged on the other
    side; the root then moves by about the rounding of the model
    divided by its slope (the tests compare with the 0.4.0 loop)."""
    m, n = y.size, grid.size
    x0 = np.empty(m)
    cross_k = np.full(m, -1)
    step = max(1, _SCAN_BLOCK // n)
    for start in range(0, m, step):
        block = slice(start, min(m, start + step))
        g = f_grid[None, :] - y[block, None]
        zero = g == 0.0
        nonzero = ~zero
        pos = g > 0.0
        cross = nonzero[:, :-1] & nonzero[:, 1:] & (pos[:, :-1] != pos[:, 1:])
        run_start = zero.copy()
        run_start[:, 1:] &= nonzero[:, :-1]
        n_sites = cross.sum(axis=1) + run_start.sum(axis=1)
        n_zero = zero.sum(axis=1)
        good_cross = (n_sites == 1) & (n_zero == 0)
        good_zero = (n_sites == 1) & (n_zero == 1)
        for r in np.flatnonzero(~(good_cross | good_zero)):
            msg = _site_error(g[r], grid, f_grid, float(y[start + r]),
                              lo, hi)
            if msg is not None:
                raise ValueError(msg)
        rows = np.arange(block.start, block.stop)
        k = np.argmax(cross, axis=1)
        cross_k[rows[good_cross]] = k[good_cross]
        x0[rows[good_zero]] = grid[np.argmax(zero, axis=1)[good_zero]]
    sel = np.flatnonzero(cross_k >= 0)
    if sel.size:
        k = cross_k[sel]
        x0[sel] = _bisect_many(model, theta, grid[k], grid[k + 1],
                               f_grid[k] - y[sel], y[sel])
    return x0


def _bisect_many(model, theta, lo, hi, g_lo, y):
    """The bisection of 0.4.0 (to the last representable digit, keeping
    the point where the model came closest to the reading) for many
    brackets at once: the same midpoints and the same rules, with one
    model call per step for all brackets."""
    lo, hi, g_lo = lo.copy(), hi.copy(), g_lo.copy()
    best_x, best_g = lo.copy(), np.abs(g_lo)
    out = np.empty(lo.size)
    exact = np.zeros(lo.size, dtype=bool)
    active = np.arange(lo.size)
    for _ in range(2200):
        mid = 0.5 * (lo[active] + hi[active])
        inside = (lo[active] < mid) & (mid < hi[active])
        active, mid = active[inside], mid[inside]
        if active.size == 0:
            break
        gm = model.predict(theta, mid[:, None]) - y[active]
        closer = np.abs(gm) < best_g[active]
        best_x[active[closer]] = mid[closer]
        best_g[active[closer]] = np.abs(gm[closer])
        hit = gm == 0.0
        out[active[hit]] = mid[hit]
        exact[active[hit]] = True
        same = (gm > 0.0) == (g_lo[active] > 0.0)
        up = ~hit & same
        lo[active[up]] = mid[up]
        g_lo[active[up]] = gm[up]
        down = ~hit & ~same
        hi[active[down]] = mid[down]
        active = active[~hit]
    rest = np.flatnonzero(~exact)
    if rest.size:
        g_hi = np.abs(model.predict(theta, hi[rest, None]) - y[rest])
        out[rest] = np.where(g_hi < best_g[rest], hi[rest], best_x[rest])
    return out


def _dfdx_many(model, theta, xs):
    """`_dfdx` at many settings: the relative step of
    `Model.numeric_jacobian` for all of them in two model calls, and
    the one-at-a-time routine (with its checked fallback step) only
    where that step does not resolve the change (since 0.5.0)."""
    xs = np.asarray(xs, dtype=float)
    out = np.empty(xs.size)
    h = 1e-6 * np.abs(xs)
    fast = np.flatnonzero(h > 0.0)
    done = np.zeros(xs.size, dtype=bool)
    if fast.size:
        xf, hf = xs[fast], h[fast]
        fp = model.predict(theta, (xf + hf)[:, None])
        fm = model.predict(theta, (xf - hf)[:, None])
        size = np.maximum(np.abs(fp), np.abs(fm))
        ok = (np.abs(fp - fm) > 1e-8 * size) \
            | (1e-6 * np.maximum(np.abs(xf), 1e-3) == hf)
        out[fast[ok]] = ((fp - fm) / (2.0 * hf))[ok]
        done[fast[ok]] = True
    for i in np.flatnonzero(~done):
        out[i] = _dfdx(model, theta, float(xs[i]))
    return out


def _slope_changes(model, theta, x0, sigma, slope, lo, hi):
    """The largest relative change of df/dx between x0 and
    x0 +/- sigma (kept inside the bracket), for every reading."""
    change = np.zeros(x0.size)
    for xe in (np.minimum(x0 + sigma, hi), np.maximum(x0 - sigma, lo)):
        use = np.flatnonzero((sigma != 0.0) & (xe != x0))
        if use.size:
            rel = np.abs(_dfdx_many(model, theta, xe[use]) / slope[use]
                         - 1.0)
            change[use] = np.maximum(change[use], rel)
    return change

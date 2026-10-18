"""Fit a model when the settings have errors too (errors in variables).

`fit` treats the settings x as exact. Often they are not: a reference
thermometer, a reference gauge or a second instrument gives the
setting with its own error bar, and a straight line fitted as if the
settings were exact comes out too flat, with error bars and a
chi-squared that are wrong. `fit_xy` fits the model and, at the same
time, the true settings xi_i, by minimizing

    S = sum_i ((y_i - f(theta, xi_i)) / sigma_y,i)^2
        + sum_i,k ((x_ik - xi_ik) / sigma_x,ik)^2

over theta and every xi_ik whose error sigma_x,ik is not zero (the
maximum-likelihood fit for independent Gaussian errors in both
coordinates; "orthogonal distance regression" when the errors are
equal). For a straight line the minimum is the line of D. York et
al., "Unified equations for the slope, intercept, and standard errors
of the best straight line", Am. J. Phys. 72, 367-375 (2004),
doi:10.1119/1.1632486 (without correlation between the x and y
errors); the tests compare with their iteration and with the
Pearson-York benchmark data.

The parameter covariance is the first-order (Gauss-Newton) one at the
solution. After the true settings are eliminated it is the inverse of

    sum_i a_i a_i^T / (sigma_y,i^2 + sum_k s_ik^2 sigma_x,ik^2),

with a_i = df/dtheta and s_ik = df/dx_k at the fitted settings (the
"effective variance" weights), and for a straight line it is the
covariance of York et al. evaluated at the adjusted points. The
minimum S is reported as chi2 with n - p degrees of freedom.
"""
from __future__ import annotations

import numpy as np

from .fit import FitResult, data_digest
from .model import Model, _settings
from .stats import SINGULAR_MSG, check_sigmas, invert_information

__all__ = ["fit_xy"]


def _x_sigmas(x_sigmas, n, d):
    arr = np.asarray(x_sigmas, dtype=float)
    if arr.size == 1:
        arr = np.full((n, d), float(arr.reshape(())))
    elif d == 1 and arr.shape == (n,):
        arr = arr[:, None].copy()
    elif arr.shape != (n, d):
        raise ValueError(f"x_sigmas must be one number, or one per "
                         f"setting: shape ({n}, {d})"
                         + (f" or ({n},)" if d == 1 else "")
                         + f"; got shape {arr.shape}")
    else:
        arr = arr.copy()
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("x_sigmas must be finite and >= 0 (0 marks a "
                         "setting that is exact)")
    return arr


def _x_steps(xi, x, xs):
    """Central-difference steps for df/dx: 1e-6 of the larger of the
    setting's size and the largest size in its column (the column's
    natural scale), so that a setting at or near zero still gets a
    step that resolves the change."""
    col = np.max(np.abs(x), axis=0)
    col = np.where(col > 0.0, col, np.max(xs, axis=0))
    col = np.where(col > 0.0, col, 1.0)
    return 1e-6 * np.maximum(np.abs(xi), col[None, :])


def _x_slopes(model, theta, xi, h, cols):
    """df_i/dx_ik for the columns in cols; a row's prediction must
    depend only on that row's settings (checked once by `fit_xy`)."""
    s = np.zeros(xi.shape)
    for k in cols:
        xp, xm = xi.copy(), xi.copy()
        xp[:, k] += h[:, k]
        xm[:, k] -= h[:, k]
        s[:, k] = (model.predict(theta, xp) - model.predict(theta, xm)) \
            / (2.0 * h[:, k])
    return s


def _check_rowwise(model, theta, x, h, free):
    """Refuse a model whose prediction for one row changes when another
    row's settings change: the true settings could then not be
    adjusted row by row."""
    n = x.shape[0]
    if n < 2:
        return
    r = int(np.argmax(np.any(free, axis=1)))
    xp = x.copy()
    xp[r, free[r]] += h[r, free[r]]
    base = model.predict(theta, x)
    moved = model.predict(theta, xp)
    others = np.arange(n) != r
    if not np.array_equal(base[others], moved[others]):
        raise ValueError(
            "fit_xy needs a model whose prediction for each measurement "
            "depends only on that measurement's own settings; changing "
            f"the settings of row {r} changed the predictions of other "
            "rows")


def _normal_blocks(model, theta, xi, y, sig, x, xs, free, h, cols):
    """Residuals, cost and the blocks of the normal equations of the
    joint problem: A (p, p) for the parameters, B (n, p, d) coupling
    them to each row's settings, D (n, d, d) for those settings, and
    the gradients."""
    xs_safe = np.where(free, xs, 1.0)
    ry = (model.predict(theta, xi) - y) / sig
    rx = np.where(free, (xi - x) / xs_safe, 0.0)
    jac = model.jacobian(theta, xi) / sig[:, None]
    sw = _x_slopes(model, theta, xi, h, cols) / sig[:, None]
    sw = np.where(free, sw, 0.0)
    a = jac.T @ jac
    b = jac[:, :, None] * sw[:, None, :]
    dmat = sw[:, :, None] * sw[:, None, :]
    diag = np.where(free, 1.0 / (xs_safe * xs_safe), 1.0)
    idx = np.arange(x.shape[1])
    dmat[:, idx, idx] += diag
    g_t = jac.T @ ry
    g_x = np.where(free, sw * ry[:, None] + rx / xs_safe, 0.0)
    cost = float(ry @ ry + np.sum(rx * rx))
    return a, b, dmat, g_t, g_x, cost


def _cost(model, theta, xi, y, sig, x, xs, free):
    xs_safe = np.where(free, xs, 1.0)
    ry = (model.predict(theta, xi) - y) / sig
    rx = np.where(free, (xi - x) / xs_safe, 0.0)
    return float(ry @ ry + np.sum(rx * rx))


def _reduced(a, b, dmat, mu):
    """The damped blocks and the parameter matrix after the settings
    are eliminated (Schur complement)."""
    p, d = a.shape[0], dmat.shape[1]
    a_d = a + mu * np.diag(np.maximum(np.diag(a), 1e-30))
    idx = np.arange(d)
    d_d = dmat.copy()
    d_d[:, idx, idx] += mu * np.maximum(dmat[:, idx, idx], 1e-30)
    e = np.linalg.solve(d_d, np.transpose(b, (0, 2, 1)))   # (n, d, p)
    s = a_d - np.einsum("npd,ndq->pq", b, e)
    return d_d, e, 0.5 * (s + s.T)


def fit_xy(model: Model, x, y, theta0, sigmas, x_sigmas, *,
           max_iter: int = 200, tol: float = 1e-12) -> FitResult:
    """Weighted least-squares fit when the settings have errors too
    (new in 0.5.0).

    model : a `Model`. Its prediction for each measurement must depend
        only on that measurement's own row of settings (checked).
    x : (n, d) measured settings (or (n,) for one setting).
    y : (n,) measured readings.
    theta0 : starting parameter vector, in `param_names` order.
    sigmas : 1-sigma errors of the readings (one number or n numbers;
        required: without them the balance between the two kinds of
        error is unknown).
    x_sigmas : 1-sigma errors of the settings: one number, n numbers
        (one setting per measurement), or an (n, d) array. 0 marks a
        setting that is exact; with all of them 0 this is `fit`.
    max_iter, tol : as for `fit`; the search also stops only when no
        fitted setting moves by more than 1e-8 of its error bar.

    Returns a `FitResult` like `fit`, with chi2 the minimum of S (see
    the module docstring) for n - p degrees of freedom, `cov` the
    first-order parameter covariance, `x_adjusted` the fitted true
    settings, and a `data_digest` that also covers `x_sigmas`. It can
    be passed to `goodness_of_fit`, `propagate`, `inverse_predict` and
    `audit_record`.

    The slopes df/dx are central differences with a step of 1e-6 of
    the larger of the setting's size and the largest size in its
    column. Refuses what `fit` refuses, missing `sigmas`, `x_sigmas`
    of the wrong shape or negative, and a model whose predictions of
    one row change with the settings of another.
    """
    if isinstance(max_iter, bool) or int(max_iter) != max_iter \
            or max_iter < 1:
        raise ValueError("max_iter must be a whole number >= 1")
    if not (np.isfinite(tol) and tol >= 0.0):
        raise ValueError("tol must be finite and >= 0")
    if not isinstance(model, Model):
        raise ValueError("model must be a labplan Model")
    x = _settings(x)
    y = np.asarray(y, dtype=float).ravel()
    n, d = x.shape
    p = model.n_params
    if y.size != n:
        raise ValueError("need one measured reading per settings row")
    if not np.all(np.isfinite(y)):
        raise ValueError("measured readings must be finite")
    if sigmas is None:
        raise ValueError("fit_xy needs the reading errors `sigmas` as "
                         "well as `x_sigmas`: the fit balances the two "
                         "kinds of error, and their ratio cannot be "
                         "estimated from the scatter")
    sig = check_sigmas(sigmas, n)
    xs = _x_sigmas(x_sigmas, n, d)
    if n < p:
        raise ValueError(f"{n} measurements cannot determine {p} "
                         "parameters")
    th = np.asarray(theta0, dtype=float).ravel()
    if th.size != p or not np.all(np.isfinite(th)):
        raise ValueError(f"theta0 must be {p} finite starting values")
    free = xs > 0.0
    cols = [k for k in range(d) if np.any(free[:, k])]
    xs_safe = np.where(free, xs, 1.0)
    xi = x.copy()
    if cols:
        _check_rowwise(model, th, x, _x_steps(x, x, xs), free)

    c = _cost(model, th, xi, y, sig, x, xs, free)
    mu = 1e-3
    converged = False
    it = 0
    c_old = c
    for it in range(1, max_iter + 1):
        h = _x_steps(xi, x, xs)
        a, b, dmat, g_t, g_x, c = _normal_blocks(
            model, th, xi, y, sig, x, xs, free, h, cols)
        stepped = False
        for _ in range(60):
            try:
                d_d, e, s = _reduced(a, b, dmat, mu)
                dg = np.linalg.solve(d_d, g_x[:, :, None])[:, :, 0]
                dth = np.linalg.solve(
                    s, -(g_t - np.einsum("npd,nd->p", b, dg)))
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
            dxi = -(dg + np.einsum("ndp,p->nd", e, dth))
            dxi = np.where(free, dxi, 0.0)
            try:
                c_new = _cost(model, th + dth, xi + dxi, y, sig, x, xs,
                              free)
            except ValueError:
                mu *= 10.0
                continue
            if c_new <= c:
                th, xi, c_old, c = th + dth, xi + dxi, c, c_new
                mu = max(mu / 10.0, 1e-15)
                stepped = True
                break
            mu *= 10.0
        if not stepped:
            converged = True
            break
        if abs(c_old - c) <= tol * (1.0 + c) and \
                float(np.max(np.abs(dth))) \
                <= 1e-10 * (1.0 + float(np.max(np.abs(th)))) and \
                float(np.max(np.abs(dxi) / xs_safe)) <= 1e-8:
            converged = True
            break
    if not converged:
        raise RuntimeError("fit_xy did not converge; check the starting "
                           "values and that the readings actually "
                           "vary with the parameters")

    h = _x_steps(xi, x, xs)
    a, b, dmat, _, _, c = _normal_blocks(model, th, xi, y, sig, x, xs,
                                         free, h, cols)
    _, _, reduced = _reduced(a, b, dmat, 0.0)
    ok, cond, cov, _ = invert_information(reduced, model.param_names)
    if not ok:
        raise ValueError(SINGULAR_MSG)
    err = np.sqrt(np.diag(cov))
    return FitResult(
        values={k: float(v) for k, v in zip(model.param_names, th)},
        sigma={k: float(s) for k, s in zip(model.param_names, err)},
        theta=th.copy(), cov=cov, chi2=c, chi2_dof=n - p,
        n_points=n, condition_number=cond, n_iter=it,
        model_name=model.name, reference=model.reference,
        data_digest=data_digest(x, y, sig, xs), units=model.units,
        x_adjusted=xi.copy())

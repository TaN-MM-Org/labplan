"""The shared exact statistics: weighted least squares, Fisher
information, scale-invariant identifiability.

Everything here is the standard machinery of independent Gaussian
measurement errors, stated plainly (see e.g. D. R. Cox and D. V.
Hinkley, Theoretical Statistics (1974), or any statistics text under
"Cramer-Rao bound"): the information matrix of a design is J^T W J
with J the model sensitivities and W = diag(1/sigma^2), the
covariance of the weighted-least-squares estimate is its inverse, and
both are the SAME matrix -- which is why labplan can promise, before
any data exist, exactly the error bars the fit will report.

Identifiability is judged on the correlation-scaled matrix
D^-1 F D^-1 with D = sqrt(diag F): parameters carry arbitrary units,
so the raw condition number partly measures the units, while exact
functional degeneracies of the model survive the scaling and unit
mismatches do not.
"""
from __future__ import annotations

import numpy as np

__all__ = ["invert_information"]

COND_MAX = 1e10

SINGULAR_MSG = ("these measurements cannot tell the parameters apart "
                "(singular or near-singular information matrix): some "
                "combination of parameters changes nothing the design "
                "can see. Measure settings that respond differently "
                "to each parameter -- `information` shows which "
                "designs work before you spend the instrument time")


def invert_information(fisher, names):
    """Scale-invariant inversion of an information matrix.

    Returns (identifiable, condition_number, cov, sigma_dict); cov and
    sigma are None when the design is not identifiable.
    """
    fisher = np.asarray(fisher, dtype=float)
    d = np.sqrt(np.diag(fisher))
    if np.any(d <= 0.0) or not np.all(np.isfinite(d)):
        return False, np.inf, None, None
    fs = fisher / np.outer(d, d)
    sv = np.linalg.svd(fs, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    if not (np.isfinite(cond) and cond <= COND_MAX):
        return False, cond, None, None
    cov = np.linalg.inv(fs) / np.outer(d, d)
    err = np.sqrt(np.diag(cov))
    return True, cond, cov, {n: float(s) for n, s in zip(names, err)}


def check_sigmas(sigmas, n):
    if sigmas is None:
        return None
    sig = np.broadcast_to(np.asarray(sigmas, dtype=float), (n,)).copy()
    if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
        raise ValueError("sigmas must be finite and positive")
    return sig

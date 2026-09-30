"""The one thing every calibration needs: a forward model with a name,
named parameters, and a source.

A `Model` wraps YOUR prediction function -- any Python callable that
maps a parameter vector and measurement settings to predicted values.
That is the whole requirement: if you can predict what your instrument
would read, labplan can plan the measurement, fit the parameters, and
keep the record. The model carries its provenance (`reference`) the
same way every physical constant in this organization does: it is
required, on purpose, because a fit whose model nobody can trace is
not a calibration.

Conventions, stated once: `theta` is a 1-D parameter vector in the
order of `param_names`; `x` is an (n, d) array of measurement settings
(one row per planned or performed measurement -- a field value, a
frequency, a temperature, anything, in your units); the forward
function returns the (n,) predicted readings in your measurement
units. Nothing here assumes any physics: the statistics downstream are
exact for any model that is differentiable in its parameters.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["Model"]


def _check_ref(reference):
    if not isinstance(reference, str) or len(reference.strip()) < 8:
        raise ValueError(
            "a real `reference` string is required: a model without a "
            "traceable source (a paper, a manual, your own derivation "
            "note) cannot anchor a calibration record")


_EPS = np.finfo(float).eps


@dataclasses.dataclass(frozen=True)
class Model:
    """A named forward model.

    name : short name of the model (appears in every report).
    f : callable f(theta, x) -> (n,) predictions, with theta a 1-D
        parameter vector and x an (n, d) settings array.
    param_names : one name per parameter, in theta order.
    reference : where the model comes from. Required, on purpose.
    units : optional note on the units of settings, readings and
        parameters: free text, or a dictionary such as
        {"fc": "Hz"}. It is copied into `fit` results, audit records
        and text reports (since 0.2.0).
    jac : optional callable jac(theta, x) -> (n, p) array of the exact
        slopes d(prediction_i)/d(theta_j), one column per parameter in
        `param_names` order. When given, `jacobian` (and so
        `information`, `design` and `fit`) uses it instead of finite
        differences. labplan cannot check that it is correct; compare
        it once with `numeric_jacobian` at a typical point.
    """

    name: str
    f: object
    param_names: tuple
    reference: str
    # a dictionary is not hashable: leave units out of the hash (it
    # still takes part in ==), so a model with dict units can be hashed
    units: object = dataclasses.field(default="", hash=False)
    jac: object = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("the model needs a non-empty name")
        if not callable(self.f):
            raise ValueError("f must be a callable f(theta, x) -> "
                             "predictions")
        if self.jac is not None and not callable(self.jac):
            raise ValueError("jac must be None or a callable "
                             "jac(theta, x) -> (n, p) slopes")
        names = tuple(str(n) for n in self.param_names)
        if len(names) < 1 or len(set(names)) != len(names):
            raise ValueError("param_names must be non-empty and "
                             "unique")
        object.__setattr__(self, "param_names", names)
        _check_ref(self.reference)

    @property
    def n_params(self):
        return len(self.param_names)

    def _theta(self, theta):
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != self.n_params:
            raise ValueError(f"theta must have {self.n_params} "
                             "entries, one per parameter name")
        return theta

    def predict(self, theta, x):
        """Evaluate the forward model with shape checking."""
        theta = self._theta(theta)
        x = _settings(x)
        y = np.asarray(self.f(theta, x), dtype=float).ravel()
        if y.size != x.shape[0]:
            raise ValueError("the forward function must return one "
                             "prediction per settings row")
        if not np.all(np.isfinite(y)):
            raise ValueError("the forward function returned non-"
                             "finite predictions; check theta and x")
        return y

    def jacobian(self, theta, x, rel_step=1e-6):
        """d(prediction)/d(parameter), one column per parameter.

        Uses the exact slopes from `jac` when the model has one
        (rel_step is then ignored), and `numeric_jacobian` otherwise.
        """
        if self.jac is None:
            return self.numeric_jacobian(theta, x, rel_step)
        theta = self._theta(theta)
        x = _settings(x)
        n, p = x.shape[0], self.n_params
        jac = np.asarray(self.jac(theta, x), dtype=float)
        if p == 1 and jac.shape == (n,):
            jac = jac[:, None]
        if jac.shape != (n, p):
            raise ValueError(f"jac must return an ({n}, {p}) array: "
                             "one row per settings row, one column "
                             "per parameter")
        if not np.all(np.isfinite(jac)):
            raise ValueError("jac returned non-finite slopes; check "
                             "theta and x")
        return jac

    def numeric_jacobian(self, theta, x, rel_step=1e-6):
        """d(prediction)/d(parameter) by central differences, one
        column per parameter, whether or not the model has `jac`.

        The step is first taken relative to each parameter's own size,
        rel_step * |theta_j|, so the result does not depend on the
        units a parameter is written in (a 1e-9 s time constant gets
        the same relative step as 1 ns). A parameter that is merely
        close to zero on its natural scale (a slope of 1e-12 next to
        an offset of 1) would then move the predictions by less than
        floating-point rounding can resolve; when the change is below
        1e-8 of the predictions' size, or the parameter is exactly
        zero, the 0.1.0 step rel_step * max(|theta_j|, 1e-3) is used
        instead.

        That fallback step knows nothing about the parameter's natural
        scale, so since 0.2.0 it is checked: the slopes are recomputed
        with a step 10 times smaller, and if the two disagree by more
        than 1e-6 of the column's largest slope (plus a bound on
        floating-point rounding), the smaller step is taken and the
        check repeated, at most 8 times. A fallback step that was
        already accurate is kept unchanged."""
        theta = self._theta(theta)
        x = _settings(x)
        jac = np.empty((x.shape[0], self.n_params))

        def shifted(j, h):
            tp, tm = theta.copy(), theta.copy()
            tp[j] += h
            tm[j] -= h
            return self.predict(tp, x), self.predict(tm, x)

        for j in range(self.n_params):
            h = rel_step * abs(theta[j])
            resolved = False
            if h > 0.0:
                fp, fm = shifted(j, h)
                resolved = _resolved(fp, fm)
            h_old = rel_step * max(abs(theta[j]), 1e-3)
            if not resolved and h_old != h:
                fp, fm = shifted(j, h_old)
                jac[:, j] = _refine(shifted, j, h_old, fp, fm)
            else:
                jac[:, j] = (fp - fm) / (2.0 * h)
        return jac


def _resolved(fp, fm):
    """True when a +/- step changes the predictions by more than
    1e-8 of their size (well above floating-point rounding)."""
    size = max(np.max(np.abs(fp)), np.max(np.abs(fm)))
    return bool(np.max(np.abs(fp - fm)) > 1e-8 * size)


def _refine(shifted, j, h, fp, fm, max_steps=8):
    """Check a fallback central difference against one with a 10x
    smaller step; move to the smaller step while they disagree.

    The central-difference error shrinks like h^2 while rounding
    grows like eps / h. The current slopes are kept as soon as the
    finer step agrees with them within 1e-6 of the column's largest
    slope plus a rounding bound. Otherwise the disagreement between
    successive steps estimates the error of the coarser one, and the
    loop returns the slopes with the smallest such estimate when the
    finer step no longer resolves the change in the predictions, when
    the disagreement stops shrinking (rounding, or noise in the model
    itself, has taken over), or after max_steps."""
    d = (fp - fm) / (2.0 * h)
    best_d, best_diff = d, np.inf
    for _ in range(max_steps):
        h2 = h / 10.0
        fp2, fm2 = shifted(j, h2)
        if not _resolved(fp2, fm2):
            break
        d2 = (fp2 - fm2) / (2.0 * h2)
        size = max(np.max(np.abs(fp2)), np.max(np.abs(fm2)))
        diff = float(np.max(np.abs(d2 - d)))
        noise = 16.0 * _EPS * size / h2
        if diff <= 1e-6 * float(np.max(np.abs(d2))) + noise:
            return d
        if diff >= best_diff:
            break
        best_d, best_diff = d, diff
        d, h = d2, h2
    return best_d


def _settings(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("settings must be an (n, d) array: one row "
                         "per measurement")
    if not np.all(np.isfinite(x)):
        raise ValueError("settings must be finite")
    return x

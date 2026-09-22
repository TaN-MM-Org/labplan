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


@dataclasses.dataclass(frozen=True)
class Model:
    """A named forward model.

    name : short name of the model (appears in every report).
    f : callable f(theta, x) -> (n,) predictions, with theta a 1-D
        parameter vector and x an (n, d) settings array.
    param_names : one name per parameter, in theta order.
    reference : where the model comes from. Required, on purpose.
    units : optional free-text note on the units of settings and
        readings, carried into reports verbatim.
    """

    name: str
    f: object
    param_names: tuple
    reference: str
    units: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("the model needs a non-empty name")
        if not callable(self.f):
            raise ValueError("f must be a callable f(theta, x) -> "
                             "predictions")
        names = tuple(str(n) for n in self.param_names)
        if len(names) < 1 or len(set(names)) != len(names):
            raise ValueError("param_names must be non-empty and "
                             "unique")
        object.__setattr__(self, "param_names", names)
        _check_ref(self.reference)

    @property
    def n_params(self):
        return len(self.param_names)

    def predict(self, theta, x):
        """Evaluate the forward model with shape checking."""
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.size != self.n_params:
            raise ValueError(f"theta must have {self.n_params} "
                             "entries, one per parameter name")
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
        """d(prediction)/d(parameter) by central differences, one
        column per parameter.

        The step is first taken relative to each parameter's own size,
        rel_step * |theta_j|, so the result does not depend on the
        units a parameter is written in (a 1e-9 s time constant gets
        the same relative step as 1 ns). A parameter that is merely
        close to zero on its natural scale (a slope of 1e-12 next to
        an offset of 1) would then move the predictions by less than
        floating-point rounding can resolve; when the change is below
        1e-8 of the predictions' size, or the parameter is exactly
        zero, the 0.1.0 step rel_step * max(|theta_j|, 1e-3) is used
        instead."""
        theta = np.asarray(theta, dtype=float).ravel()
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
                size = max(np.max(np.abs(fp)), np.max(np.abs(fm)))
                resolved = np.max(np.abs(fp - fm)) > 1e-8 * size
            h_old = rel_step * max(abs(theta[j]), 1e-3)
            if not resolved and h_old != h:
                h = h_old
                fp, fm = shifted(j, h)
            jac[:, j] = (fp - fm) / (2.0 * h)
        return jac


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

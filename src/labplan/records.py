"""A plain, checked file contract for measurement records.

One CSV per record: setting columns x1..xd, the measured reading y,
and optionally its 1-sigma error. Values are written with `repr`, so
the round trip is bit-exact; the header and every value are checked
on load, and a malformed file is refused, not guessed at -- the same
contract style every package in this organization uses.
"""
from __future__ import annotations

import csv

import numpy as np

from .model import _settings
from .stats import check_sigmas

__all__ = ["save_measurements_csv", "load_measurements_csv"]


def _header(d, has_sigma):
    cols = tuple(f"x{i + 1}" for i in range(d)) + ("y",)
    return cols + ("sigma",) if has_sigma else cols


def save_measurements_csv(path, x, y, sigmas=None):
    """Write a measurement record; the exact inverse of
    `load_measurements_csv`."""
    x = _settings(x)
    y = np.asarray(y, dtype=float).ravel()
    if y.size != x.shape[0]:
        raise ValueError("need one reading per settings row")
    if not np.all(np.isfinite(y)):
        raise ValueError("readings must be finite")
    sig = check_sigmas(sigmas, x.shape[0])
    header = _header(x.shape[1], sig is not None)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for i in range(x.shape[0]):
            row = [repr(float(v)) for v in x[i]] + [repr(float(y[i]))]
            if sig is not None:
                row.append(repr(float(sig[i])))
            wr.writerow(row)


def load_measurements_csv(path):
    """Read a measurement record written by `save_measurements_csv`.

    Returns (x, y, sigmas) with sigmas None when the file has no
    sigma column.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError("empty measurement file")
    header = tuple(rows[0])
    if len(header) < 2 or header[-1] not in ("y", "sigma"):
        raise ValueError(f"unrecognized header {header}")
    has_sigma = header[-1] == "sigma"
    d = len(header) - (2 if has_sigma else 1)
    if d < 1 or header != _header(d, has_sigma):
        raise ValueError(f"measurement header must be "
                         f"{_header(max(d, 1), has_sigma)}; got "
                         f"{header}")
    body = rows[1:]
    if not body:
        raise ValueError("measurement file has no data rows")
    xs, ys, ss = [], [], []
    for row in body:
        if len(row) != len(header):
            raise ValueError(f"row {row!r} does not match the header")
        try:
            vals = [float(v) for v in row]
        except ValueError as exc:
            raise ValueError(f"non-numeric value in row {row!r}") \
                from exc
        xs.append(vals[:d])
        ys.append(vals[d])
        if has_sigma:
            ss.append(vals[d + 1])
    x = np.array(xs)
    y = np.array(ys)
    sig = np.array(ss) if has_sigma else None
    _settings(x)
    if not np.all(np.isfinite(y)):
        raise ValueError("readings must be finite")
    if sig is not None:
        check_sigmas(sig, y.size)
    return x, y, sig

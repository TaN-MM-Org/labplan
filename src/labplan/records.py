"""A plain, checked file contract for measurement records.

One CSV per record: setting columns x1..xd, the measured reading y,
and optionally its 1-sigma error. Values are written with `repr`, so
the round trip is bit-exact; the header and every value are checked
on load, and a malformed file is refused, not guessed at -- the same
contract style every package in this organization uses.

`load_columns` (since 0.3.0) reads the columns you name from any
delimited text table with a header row, as instruments, data loggers
and spreadsheet programs export them, under the same rule: every
value it returns was read as a finite number from the cell you named,
and anything it would have to guess is refused with the line number.
"""
from __future__ import annotations

import csv

import numpy as np

from .model import _settings
from .stats import check_sigmas

__all__ = ["save_measurements_csv", "load_measurements_csv",
           "load_columns"]


def _header(d, has_sigma):
    cols = tuple(f"x{i + 1}" for i in range(d)) + ("y",)
    return cols + ("sigma",) if has_sigma else cols


def save_measurements_csv(path, x, y, sigmas=None) -> None:
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


def load_measurements_csv(path) -> tuple:
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


_DELIMITERS = (",", "\t", ";")


def _split(line, delimiter):
    if delimiter is None:
        return line.split()
    return next(csv.reader([line], delimiter=delimiter))


def _cell_value(text, decimal, where):
    cell = text.strip()
    if decimal == ",":
        if "." in cell:
            raise ValueError(f"{where}: {cell!r} has a '.' although "
                             "decimal=',' was given (a thousands "
                             "separator?); refusing to guess")
        cell = cell.replace(",", ".")
    if cell == "" or "_" in cell:
        raise ValueError(f"{where}: {text!r} is not a number")
    try:
        value = float(cell)
    except ValueError as exc:
        hint = ""
        if decimal == "." and "," in cell:
            hint = " (if the file uses decimal commas, pass decimal=',')"
        raise ValueError(f"{where}: {text!r} is not a number{hint}") \
            from exc
    if not np.isfinite(value):
        raise ValueError(f"{where}: {text!r} is not a finite number; "
                         "remove or repair that row")
    return value


def load_columns(path, x, y, sigma=None, *, delimiter=None,
                 decimal=".", comment="#", skip_rows=0, header=True,
                 encoding="utf-8-sig") -> tuple:
    """Read settings, readings and (optionally) reading errors from
    the named columns of a delimited text table.

    path : the file.
    x : the settings column, or a list of columns (one per setting
        dimension, in order). A column is its name in the header row,
        exactly as written there (surrounding spaces removed), or its
        position counted from 0.
    y : the readings column.
    sigma : optional column of 1-sigma reading errors.
    delimiter : ",", "\\t", ";" or any other single character, or
        None (the default) to detect it from the header row: the one
        of comma, tab and semicolon that occurs there, or runs of
        spaces and tabs when none does. If more than one occurs the
        file is refused and you pass `delimiter=`.
    decimal : "." (default) or "," for files written with decimal
        commas (then the delimiter cannot be a comma).
    comment : lines whose first non-blank character is this are
        skipped (None to switch off). Blank lines are skipped.
    skip_rows : number of lines to skip at the top before anything
        else (an instrument's preamble that is not marked as comment).
    header : True when the first remaining line names the columns;
        False when there is no header (then columns are positions).
    encoding : text encoding; the default also removes the byte-order
        mark that some spreadsheet programs write.

    Returns (x, y, sigmas) like `load_measurements_csv`: x is (n, d),
    sigmas is None without a sigma column. Other columns may hold
    anything (text, timestamps) and are not read.

    Refuses, naming the line and the column: a column name that is
    not in the header (the message lists the names that are) or that
    appears twice, the same column asked for twice, a row with a
    different number of fields than the header, an empty cell, text,
    NaN or infinity in a requested column, a reading error that is
    not positive, and a file with no data rows.
    """
    xs_req = list(x) if isinstance(x, (list, tuple)) else [x]
    if not xs_req:
        raise ValueError("x must name at least one column")
    requested = xs_req + [y] + ([] if sigma is None else [sigma])
    for col in requested:
        if isinstance(col, bool) or not isinstance(col, (str, int)):
            raise ValueError("columns are given by header name (str) "
                             f"or by position (int); got {col!r}")
    if decimal not in (".", ","):
        raise ValueError("decimal must be '.' or ','")
    if delimiter is not None:
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ValueError("delimiter must be one character, or None "
                             "to detect it")
        if delimiter == decimal:
            raise ValueError("the delimiter and the decimal mark must "
                             "differ")
    skip_rows = int(skip_rows)
    if skip_rows < 0:
        raise ValueError("skip_rows must be >= 0")
    if comment is not None and (not isinstance(comment, str)
                                or len(comment) != 1):
        raise ValueError("comment must be one character or None")

    with open(path, newline="", encoding=encoding) as fh:
        lines = fh.read().splitlines()
    numbered = []
    for i, line in enumerate(lines[skip_rows:], start=skip_rows + 1):
        stripped = line.strip()
        if not stripped:
            continue
        if comment is not None and stripped.startswith(comment):
            continue
        numbered.append((i, line))
    if not numbered:
        raise ValueError(f"{path}: no header and no data rows")

    first_no, first = numbered[0]
    if delimiter is None:
        found = [d for d in _DELIMITERS if d in first and d != decimal]
        if len(found) > 1:
            shown = ", ".join(repr(d) for d in found)
            raise ValueError(f"line {first_no}: the delimiter is "
                             f"ambiguous ({shown} all occur); pass "
                             "delimiter=")
        delimiter = found[0] if found else None
    width = len(_split(first, delimiter))

    if header:
        names = [n.strip() for n in _split(first, delimiter)]
        body = numbered[1:]
    else:
        names = None
        body = numbered

    def index_of(col):
        if isinstance(col, int):
            if not 0 <= col < width:
                raise ValueError(f"column position {col} is outside "
                                 f"the {width} columns of the file")
            return col
        if names is None:
            raise ValueError(f"column {col!r} is a name, but "
                             "header=False; give positions instead")
        hits = [k for k, n in enumerate(names) if n == col]
        if not hits:
            raise ValueError(f"no column named {col!r}; the header "
                             f"(line {first_no}) has {names}")
        if len(hits) > 1:
            raise ValueError(f"the column name {col!r} appears "
                             f"{len(hits)} times in the header; give "
                             "its position instead")
        return hits[0]

    idx = [index_of(col) for col in requested]
    if len(set(idx)) != len(idx):
        raise ValueError("the same column is asked for twice")
    label = [repr(c) if isinstance(c, str) else f"#{c}"
             for c in requested]
    if not body:
        raise ValueError(f"{path}: the file has no data rows")
    rows = []
    for line_no, line in body:
        fields = _split(line, delimiter)
        if len(fields) != width:
            raise ValueError(f"line {line_no}: {len(fields)} fields, "
                             f"but the {'header' if header else 'first row'}"
                             f" has {width}")
        rows.append([_cell_value(fields[k], decimal,
                                 f"line {line_no}, column {lab}")
                     for k, lab in zip(idx, label)])
    arr = np.array(rows, dtype=float)
    d = len(xs_req)
    xv = _settings(arr[:, :d])
    yv = arr[:, d].copy()
    sig = None
    if sigma is not None:
        sig = arr[:, d + 1].copy()
        if np.any(sig <= 0.0):
            bad = body[int(np.argmax(sig <= 0.0))][0]
            raise ValueError(f"line {bad}: reading errors must be "
                             "positive")
    return xv, yv, sig

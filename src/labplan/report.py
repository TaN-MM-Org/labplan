"""The audit trail: what was fitted, to which data, by what, when.

A calibration that cannot be traced is an opinion. `audit_record`
turns a `FitResult` into a plain dictionary that says exactly what
happened: the model's name and source, every fitted value with its
error bar, the goodness of fit, the identifiability diagnostics, a
sha256 digest of the exact data arrays the result answers for, the
package versions that produced it, and a UTC timestamp. It is
JSON-serializable as-is, so it can live next to the data forever;
`report_text` renders the same record for a human or a logbook.

Nothing in the record is re-derived at reading time: it is a
statement of record, checked round-trip in the tests
(json.dumps -> json.loads reproduces it exactly).
"""
from __future__ import annotations

import datetime
import json

import numpy as np

from .fit import FitResult

__all__ = ["audit_record", "report_text"]


def audit_record(result: FitResult, operator="", note=""):
    """A JSON-serializable statement of record for one fit.

    operator : who ran the calibration (free text, e.g. a name or an
        instrument id); recorded verbatim.
    note : anything else worth remembering, recorded verbatim.
    """
    from . import __version__
    if not isinstance(operator, str) or not isinstance(note, str):
        raise ValueError("operator and note must be strings")
    rec = {
        "record": "labplan.calibration",
        "model": result.model_name,
        "model_reference": result.reference,
        "parameters": {
            name: {"value": result.values[name],
                   "sigma": result.sigma[name]}
            for name in result.values
        },
        "covariance": [[float(v) for v in row] for row in result.cov],
        "chi2": result.chi2,
        "chi2_dof": result.chi2_dof,
        "n_points": result.n_points,
        "condition_number": result.condition_number,
        "data_sha256": result.data_digest,
        "software": {"labplan": __version__,
                     "numpy": np.__version__},
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "operator": operator,
        "note": note,
    }
    # the record must survive serialization exactly
    json.dumps(rec)
    return rec


def report_text(record):
    """Render an audit record for a human or a logbook."""
    if not isinstance(record, dict) \
            or record.get("record") != "labplan.calibration":
        raise ValueError("expected an audit_record dictionary")
    lines = [
        "labplan calibration record",
        f"  model:      {record['model']}",
        f"  source:     {record['model_reference']}",
        f"  when (UTC): {record['timestamp_utc']}",
    ]
    if record.get("operator"):
        lines.append(f"  operator:   {record['operator']}")
    lines.append("  parameters:")
    for name, pv in record["parameters"].items():
        lines.append(f"    {name} = {pv['value']:.9g} "
                     f"+/- {pv['sigma']:.3g}")
    if record["chi2"] is not None:
        lines.append(f"  chi2 / dof: {record['chi2']:.4g} / "
                     f"{record['chi2_dof']} (near 1 per dof means "
                     "model and stated errors agree)")
    else:
        lines.append("  chi2:       not available (no measurement "
                     "errors were supplied; error bars are scaled "
                     "from the residual scatter)")
    lines.append(f"  points:     {record['n_points']}")
    lines.append(f"  condition:  {record['condition_number']:.3g} "
                 "(unit-free; large means barely identifiable)")
    lines.append(f"  data sha256: {record['data_sha256']}")
    sw = record["software"]
    lines.append(f"  software:   labplan {sw['labplan']}, "
                 f"numpy {sw['numpy']}")
    if record.get("note"):
        lines.append(f"  note:       {record['note']}")
    return "\n".join(lines)

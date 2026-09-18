"""labplan: plan the measurement, fit the model, keep the record --
for any lab model.

Nine research packages in this organization ended up needing the same
loop: predict the error bars of a planned measurement before spending
the instrument time, choose the most informative settings, fit the
model to the data with honest uncertainties, and refuse -- with an
explanation -- whenever the design cannot identify the parameters.
labplan is that loop extracted into one dependency-light package that
works with ANY forward model you can write as a Python function: a
spin resonance versus field, a critical current versus temperature, a
transmittance versus wavelength, a sensor reading versus anything.

The promise that makes planning meaningful: the planner's error bars
and the fitter's error bars are the SAME matrix, (J^T W J)^-1 -- so
what is promised before the measurement is what is reported after,
exactly, whenever the model describes the data. On top of the
model-based statistics, split conformal prediction supplies
distribution-free intervals with an exact finite-sample guarantee,
for the day the model is wrong in ways nobody modeled. And every fit
can be turned into an audit record that states what was fitted, to
exactly which data (sha256), by which software versions, when.

Statistics sources, stated once: weighted least squares and Fisher
information are textbook (e.g. Cox & Hinkley, Theoretical Statistics
(1974)); D-optimal design follows F. Pukelsheim, Optimal Design of
Experiments, SIAM (2006); split conformal prediction follows Vovk,
Gammerman & Shafer (2005), Lei et al., J. Am. Stat. Assoc. 113, 1094
(2018) and Angelopoulos & Bates, arXiv:2107.07511. Every statistical
claim in the test suite is pinned to a closed form, an exact
identity, or seeded simulation against an exact formula -- never a
stored number.
"""
from .model import Model
from .fit import FitResult, fit
from .plan import design, information, repeats_for
from .conformal import (conformal_interval, conformal_quantile,
                        coverage_exact)
from .report import audit_record, report_text
from .records import load_measurements_csv, save_measurements_csv

__version__ = "0.1.0"
__all__ = [
    "Model", "FitResult", "fit",
    "information", "design", "repeats_for",
    "conformal_quantile", "conformal_interval", "coverage_exact",
    "audit_record", "report_text",
    "save_measurements_csv", "load_measurements_csv",
    "__version__",
]

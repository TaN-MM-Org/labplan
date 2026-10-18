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
for the day the model is wrong in ways nobody modeled; `propagate`
carries the parameter covariance into the error bar of any number
computed from the parameters (JCGM 100:2008, sections 5.1.2 and
5.2); `goodness_of_fit` turns a fit's chi-squared into the
probability of a value at least that large; `load_columns` reads the
named columns of an instrument's delimited text export. Since 0.4.0,
`inverse_predict` reads a calibration curve backwards (the setting of
a new reading, with its error bar from the reading and from the
calibration), and `runs_test` checks whether the residuals come in
long blocks of one sign, a misfit chi-squared cannot see. Since 0.5.0,
`fit_xy` fits when the settings have error bars too (errors in
variables; for a straight line the method of York et al., Am. J.
Phys. 72, 367 (2004)). And every fit
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
from .fit import FitResult, fit, goodness_of_fit
from .fitxy import fit_xy
from .plan import design, information, repeats_for
from .propagate import propagate
from .inverse import inverse_predict
from .runs import runs_test
from .conformal import (conformal_interval, conformal_quantile,
                        coverage_exact)
from .report import audit_record, report_text
from .records import (load_columns, load_measurements_csv,
                      save_measurements_csv)

__version__ = "0.5.0"
__all__ = [
    "Model", "FitResult", "fit", "fit_xy", "goodness_of_fit",
    "information", "design", "repeats_for", "propagate",
    "inverse_predict", "runs_test",
    "conformal_quantile", "conformal_interval", "coverage_exact",
    "audit_record", "report_text",
    "save_measurements_csv", "load_measurements_csv", "load_columns",
    "__version__",
]

# labplan

[![PyPI](https://img.shields.io/pypi/v/labplan.svg)](https://pypi.org/project/labplan/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22826765-blue)](https://doi.org/10.5281/zenodo.22826765) [![tests](https://github.com/TaN-MM-Org/labplan/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/labplan/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Every measurement campaign asks the same four questions. Can the
measurements I am about to take determine the numbers I care about --
and how well? Which settings are worth the instrument time? Once the
data exist, what are the numbers, with error bars that mean it? And a
year from now, can anyone trace exactly what was fitted, to which
data, by what? `labplan` answers all four for ANY instrument or
experiment you can describe with a Python function -- and refuses,
with an explanation, whenever the honest answer is "this design
cannot tell".

Nine research packages in this organization -- covering colour-centre
spins, squeezed light, spin-squeezed clocks, Raman maps,
single-photon emitters, superconducting detectors, band structure,
semiconductor heterostructures and photonic fabrication -- each grew
the same planning-and-calibration loop for its own physics. `labplan`
is that loop extracted, generalized, and hardened into a single
dependency-light package (NumPy only, Python 3.9-3.14): the tenth
package is the pattern itself.

## Install

```
pip install labplan        # NumPy only
```

## The loop in one example

```python
import numpy as np
from labplan import (Model, information, design, fit,
                     repeats_for, audit_record, report_text)

# 1. Your model: anything that predicts a reading from parameters
#    and settings. Here, a sensor line y = gain * x + offset.
m = Model("sensor line",
          lambda th, x: th[0] * x[:, 0] + th[1],
          param_names=("gain", "offset"),
          reference="sensor manual rev. 3, eq. (2)")

# 2. Before measuring: would 12 planned settings determine the
#    parameters, and how well, at 0.05 units of reading noise?
x_planned = np.linspace(0.5, 5.0, 12)[:, None]
plan = information(m, theta=[2.0, 0.0], x=x_planned, sigmas=0.05)
print(plan["identifiable"], plan["sigma"])

# ... or let labplan pick the best 5 of the settings you can reach,
#     and price a target error bar in repeats (a closed form):
pick = design(m, [2.0, 0.0], x_planned, n_pick=5, sigmas=0.05)
r, predicted = repeats_for({"gain": 0.01}, plan)

# 3. After measuring: fit, with the SAME matrix the plan promised.
res = fit(m, x_planned, y_measured, theta0=[1.0, 0.0], sigmas=0.05)
print(res.values, res.sigma, res.chi2)

# 4. Keep the record: what, to which data (sha256), by what, when.
rec = audit_record(res, operator="T. Mahim", note="bench 2, warm-up ok")
print(report_text(rec))
```

The central promise: the planner's error bars and the fit's error
bars are the same matrix, so what is promised before the measurement
is what is reported after -- exactly, whenever the model describes
the data. The tests assert that equality to machine precision.

## When the model might be wrong

Model-based error bars assume the model is right. For the day it is
not, `conformal_quantile` supplies distribution-free prediction
intervals from held-out calibration data, with an exact finite-sample
guarantee that holds whatever the model and whatever the noise (split
conformal prediction; Vovk, Gammerman and Shafer, Algorithmic
Learning in a Random World, Springer (2005); Lei et al., J. Am.
Stat. Assoc. 113, 1094 (2018); Angelopoulos and Bates,
arXiv:2107.07511):

```python
from labplan import conformal_quantile, conformal_interval

q = conformal_quantile(abs_errors_heldout, alpha=0.1)   # 90% level
lo, hi = conformal_interval(new_predictions, q)
```

Its honest limits are stated in the docstring rather than hidden:
the guarantee is marginal, and it needs the calibration data to be
exchangeable with the new measurement -- last month's instrument
state does not certify next month's drift.

## What is inside

- **`Model`**: your forward function with named parameters and a
  mandatory `reference` -- provenance travels with every prediction,
  the same rule every package in this organization applies.
- **`information` / `design` / `repeats_for`**: predicted error bars
  from the design alone (the Fisher information of independent
  Gaussian measurements; any statistics text, under "Cramer-Rao
  bound"); greedy D-optimal selection of the most informative
  settings (Pukelsheim, Optimal Design of Experiments, SIAM (2006));
  and the exact 1/sqrt(repeats) law, inverted in closed form.
- **`fit`**: Levenberg-Marquardt weighted least squares in pure
  NumPy, with exact known-noise covariance and a chi-squared check
  when measurement errors are supplied, and residual-scaled error
  bars (stated as such) when they are not.
- **`conformal_quantile` / `conformal_interval` / `coverage_exact`**:
  distribution-free intervals with the exact finite-sample coverage
  formula exposed for checking.
- **`audit_record` / `report_text`**: a JSON-serializable statement
  of record -- model, source, values, error bars, goodness of fit,
  identifiability, the sha256 of the exact data arrays, software
  versions, UTC timestamp, operator -- and its human-readable
  rendering.
- **`save_measurements_csv` / `load_measurements_csv`**: a plain,
  checked file contract whose round trip is bit-exact.

## Refusals, not guesses

A design that cannot tell the parameters apart is refused with an
explanation, in the planner, the fit and the design tool alike --
judged on a unit-free (correlation-scaled) information matrix, so
mixed units can never fake or hide a degeneracy. Too few points, a
non-converging fit, an uncertifiable conformal level, a malformed
data file: each refuses with the reason and, where one exists, the
remedy.

## How it is checked

14 tests (Python 3.9-3.14, run in CI on every push), every
statistical claim pinned to a closed form, an exact identity, or
seeded simulation against an exact formula -- never a stored number.
Highlights: on a linear model the fit covariance equals the textbook
closed form sigma^2 (X^T X)^-1 exactly, and the planner promises the
same matrix; 400 seeded Monte-Carlo experiments match the reported
error bars; an exactly degenerate model is refused via an exact rank
argument; the greedy design obeys the rank-one determinant identity,
reproduces its own rule, never loses to a random subset, and -- for
the two-point line design -- matches the classical optimum found by
exhaustion; the repeat law is asserted by tiling the design; the
conformal quantile is the exact rank formula and seeded simulation
matches the exact closed-form coverage inside the published
two-sided guarantee; the audit record survives JSON round trip
exactly and its digest pins the exact data; file round trips are
bit-exact.

## Honest limits

Deliberate scope, designed out with reasons: the Gaussian
error-bar machinery is exact for independent Gaussian measurement
errors and first-order-accurate otherwise (the conformal tools are
the assumption-free complement, and their own limits are stated);
the greedy design is a transparent heuristic, not a proof of global
optimality; no physics ships in this package at all -- your model
and its `reference` carry the physics, and the nine physics packages
of this organization remain the place where specific instruments'
models live, each already wired into this same loop.

## Support and governance

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and takes the final decision on scope and releases.
Design questions are discussed in the open in issues and pull
requests, and the standing rule of
[CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly as it
binds contributors: a change that touches the statistics arrives with
a test, and a claim arrives with its source.

Support runs through the
[issue tracker](https://github.com/TaN-MM-Org/labplan/issues). Usage
questions are welcome alongside bug reports; a docstring that left a
unit or a convention unclear is treated as a documentation bug, not
user error. While the version is below 1.0 the API may still move
between minor versions; such changes are called out in the release
notes.

## License

Apache-2.0. Every release is archived on Zenodo under the concept DOI
[10.5281/zenodo.22826765](https://doi.org/10.5281/zenodo.22826765),
which always resolves to the latest version.

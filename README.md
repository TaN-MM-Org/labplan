# labplan

[![PyPI](https://img.shields.io/pypi/v/labplan.svg)](https://pypi.org/project/labplan/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22826765-blue)](https://doi.org/10.5281/zenodo.22826765) [![tests](https://github.com/TaN-MM-Org/labplan/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/labplan/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

`labplan` is a small Python package for planning a measurement,
fitting a model to the results, and keeping a record of what was
done. It works with any instrument or experiment whose readings you
can predict with a Python function of some unknown numbers (the
parameters) and the settings you measure at. It helps with four
questions:

- **Before measuring:** can the measurements I plan pin down the
  numbers I care about, and how large will the error bars be? Which
  settings are worth the instrument time, and how many repeats do I
  need for a target error bar?
- **After measuring:** what are the fitted numbers, and their error
  bars?
- **If the model may be wrong:** how wide a band around the
  prediction still covers new measurements at a chosen rate?
- **Later:** can anyone trace what was fitted, to exactly which data,
  with which software, and when?

When a planned measurement cannot separate the parameters, the
package says so instead of returning numbers: the planner reports it,
and the fit and the design tool stop with an error message that
explains why.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install](#install)
- [Examples](#examples) (each with the output it prints)
- [What is in the package](#what-is-in-the-package)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections in earlier versions](#corrections-in-earlier-versions)
- [Limits](#limits)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

- **Model** -- your prediction function `f(theta, x)`. It takes the
  **parameters** `theta` (the unknown numbers, such as a gain and an
  offset) and the **settings** `x` (what you set on the instrument,
  one row per measurement), and returns the predicted **readings**.
- **Error bar** -- the one-standard-deviation ("1-sigma")
  uncertainty of a number. `sigmas` are the error bars of the
  individual readings; `sigma` in results are the error bars of the
  fitted parameters.
- **Design** -- the list of settings you plan to measure at.
- **Information matrix** -- a table that measures how strongly the
  readings of a design respond to each parameter, weighted by the
  reading error bars. In formula form it is `J^T W J`, where `J` holds
  the slopes of each prediction with respect to each parameter and
  `W = diag(1/sigma^2)`. Its inverse is the **covariance** of the
  parameters: the diagonal holds the squared error bars, the rest says
  how the parameter errors move together. The planner and the fit use
  this same formula, which is why the planned error bars and the
  fitted ones agree when the model describes the data (see the
  checks below).
- **Identifiable** -- the design can tell the parameters apart. It
  cannot when some combination of parameters changes nothing the
  design can see (for example, two parameters that only ever appear
  added together).
- **Condition number** -- how close the information matrix is to
  "cannot tell apart"; large means barely identifiable. `labplan`
  computes it after rescaling the matrix so that the units of the
  parameters do not matter, and treats a design as not identifiable
  above 1e10.
- **D-optimal design** -- choosing the settings that make the
  determinant of the information matrix largest, which shrinks the
  joint uncertainty of all parameters.
- **Chi-squared (chi2)** -- the sum over all readings of
  ((reading - prediction) / sigma) squared. Divided by the **degrees
  of freedom** (number of readings
  minus number of parameters) it should be near 1 when the model and
  the stated reading errors agree.
- **Conformal prediction** -- a way to put a band around predictions
  from held-out data alone, without trusting the model. The
  **calibration scores** are the absolute errors
  `|measured - predicted|` on measurements not used in the fit.
  **alpha** is the allowed miss rate (0.1 means a 90 % band). The
  guarantee needs the calibration data and the new measurement to be
  **exchangeable**: drawn the same way, in any order.
- **sha256 digest** -- a 64-character fingerprint of the exact data
  arrays. Any change to the data changes it.

## Install

```
pip install labplan
```

It needs Python 3.9 or newer and NumPy 1.22 or newer, and nothing
else. The package has no built-in units: settings, readings, error
bars and parameters are in whatever units your model uses, and
results come back in those units. Settings `x` are an `(n, d)` array,
one row per measurement; a 1-D array is read as one setting per
measurement. Parameters are in the order of `param_names`.

Runnable worked examples live in [`examples/`](examples/):
`01_plan_then_fit.py` plans, simulates, fits and records a
calibration of a first-order low-pass filter (it picks 6 of 25
candidate frequencies and asks for 14 repeats of that sweep to reach
an error bar of 5 Hz on the corner frequency), and
`02_conformal_certificate.py` wraps a conformal band around a
deliberately wrong straight-line model and measures its coverage
over 300 simulated rounds (0.899, against the exact 0.9000 for its
79 calibration points).

## Examples

Each example below runs as written, and the output shown is what it
printed with labplan 0.1.1. The model, the numbers and the noise are
illustrative; the random numbers come from fixed seeds, so the
output is repeatable.

### 1. Before measuring: will this design work, and how well?

```python
import numpy as np
from labplan import Model, information

# A sensor whose reading is a straight line in the setting x:
# reading = gain * x + offset. All numbers here are illustrative.
line = Model("sensor line",
             lambda th, x: th[0] * x[:, 0] + th[1],
             param_names=("gain", "offset"),
             reference="illustrative straight-line sensor model")

x_planned = np.linspace(0.5, 5.0, 12)     # 12 planned settings
plan = information(line, theta=[2.0, 0.0], x=x_planned, sigmas=0.05)
print("identifiable:", plan["identifiable"])
for name, s in plan["sigma"].items():
    print(f"expected error bar on {name}: {s:.5f}")
```

```
identifiable: True
expected error bar on gain: 0.01022
expected error bar on offset: 0.03160
```

`theta` is your best guess of the parameters (from a datasheet or a
previous calibration); the answer is computed there. `sigmas=0.05`
says each reading will have an error bar of 0.05. Every `Model`
needs a `reference` string saying where the model comes from.

### 2. Choose the settings, and count the repeats

```python
import numpy as np
from labplan import Model, design, information, repeats_for

line = Model("sensor line",
             lambda th, x: th[0] * x[:, 0] + th[1],
             param_names=("gain", "offset"),
             reference="illustrative straight-line sensor model")
candidates = np.linspace(0.5, 5.0, 12)

pick = design(line, [2.0, 0.0], candidates, n_pick=4, sigmas=0.05)
print("chosen settings:", candidates[pick["indices"]])

plan = information(line, [2.0, 0.0], candidates, sigmas=0.05)
r, predicted = repeats_for({"gain": 0.002}, plan)
print("repeats of the 12-point sweep needed:", r)
print(f"predicted error bar on gain: {predicted['gain']:.5f}")
```

```
chosen settings: [5.         0.5        0.90909091 4.59090909]
repeats of the 12-point sweep needed: 27
predicted error bar on gain: 0.00197
```

`design` picks settings one at a time, each time the candidate that
adds the most information, and lists them in the order picked. For a
straight line the ends of the range are the most informative, so it
takes those first. `repeats_for` uses the rule that repeating a
whole design `r` times divides every error bar by `sqrt(r)`: here
0.01022 / 0.002 = 5.1, squared 26.1, rounded up to 27.

### 3. After measuring: fit the model

```python
import numpy as np
from labplan import Model, fit

line = Model("sensor line",
             lambda th, x: th[0] * x[:, 0] + th[1],
             param_names=("gain", "offset"),
             reference="illustrative straight-line sensor model")
x = np.linspace(0.5, 5.0, 12)

# Simulated measurements: true gain 2.0, offset 0.1, noise 0.05.
rng = np.random.default_rng(1)
y = 2.0 * x + 0.1 + 0.05 * rng.standard_normal(x.size)

res = fit(line, x, y, theta0=[1.0, 0.0], sigmas=0.05)
for name in line.param_names:
    print(f"{name} = {res.values[name]:.4f} +/- {res.sigma[name]:.4f}")
print(f"chi2 = {res.chi2:.2f} for {res.chi2_dof} degrees of freedom")
```

```
gain = 2.0005 +/- 0.0102
offset = 0.1104 +/- 0.0316
chi2 = 4.10 for 10 degrees of freedom
```

The fitted error bars are the ones example 1 planned for the same
settings (0.0102 and 0.0316). `theta0` is the starting guess. If you
leave out `sigmas`, the error bars are estimated from the scatter of
the residuals (the differences between the readings and the fitted
predictions) instead, and `chi2` is `None`.

### 4. A design that cannot work is refused

```python
import numpy as np
from labplan import Model, information, fit

# Two parameters that only ever appear as their sum: no measurement
# can tell them apart.
bad = Model("sum only", lambda th, x: (th[0] + th[1]) * x[:, 0],
            param_names=("a", "b"),
            reference="illustrative degenerate model")
x = np.linspace(0.5, 5.0, 12)

plan = information(bad, [1.0, 2.0], x, sigmas=0.1)
print("identifiable:", plan["identifiable"], "| error bars:", plan["sigma"])
try:
    fit(bad, x, 3.0 * x, [1.0, 2.0], sigmas=0.1)
except ValueError as err:
    print("fit refused:", err)
```

```
identifiable: False | error bars: None
fit refused: these measurements cannot tell the parameters apart (singular or near-singular information matrix): some combination of parameters changes nothing the design can see. Measure settings that respond differently to each parameter -- `information` shows which designs work before you spend the instrument time
```

### 5. Keep a record of the fit

```python
import json
import numpy as np
from labplan import Model, fit, audit_record

line = Model("sensor line",
             lambda th, x: th[0] * x[:, 0] + th[1],
             param_names=("gain", "offset"),
             reference="illustrative straight-line sensor model")
x = np.linspace(0.5, 5.0, 12)
rng = np.random.default_rng(1)
y = 2.0 * x + 0.1 + 0.05 * rng.standard_normal(x.size)
res = fit(line, x, y, theta0=[1.0, 0.0], sigmas=0.05)

rec = audit_record(res, operator="bench 2", note="illustrative run")
print(sorted(rec))
print(rec["data_sha256"][:16], "...")
print("survives JSON round trip:", json.loads(json.dumps(rec)) == rec)
```

```
['chi2', 'chi2_dof', 'condition_number', 'covariance', 'data_sha256', 'model', 'model_reference', 'n_points', 'note', 'operator', 'parameters', 'record', 'software', 'timestamp_utc']
dbf50525b0aff6ea ...
survives JSON round trip: True
```

The record is a plain dictionary: model name and reference, each
value with its error bar, the covariance, chi2, the number of points,
the condition number, the sha256 digest of the fitted data (settings,
readings and reading errors), the labplan and NumPy versions, a UTC
timestamp, the operator and a note. `report_text(rec)` turns it into
readable lines for a logbook. (The example prints only part of the
record because the timestamp changes on every run.)

### 6. A band that holds even if the model is wrong

```python
import numpy as np
from labplan import conformal_quantile, conformal_interval, coverage_exact

# Absolute errors |measured - predicted| on 39 held-out measurements
# (simulated here; in practice they come from your own data).
rng = np.random.default_rng(3)
scores = np.abs(0.1 * rng.standard_normal(39))

q = conformal_quantile(scores, alpha=0.1)          # aim: 90 % coverage
lo, hi = conformal_interval(np.array([4.2, 5.0]), q)
print(f"half-width q = {q:.4f}")
print("intervals:", np.round(lo, 4), "to", np.round(hi, 4))
print("exact coverage for n = 39:", coverage_exact(39, 0.1))

try:
    conformal_quantile(scores[:5], alpha=0.1)
except ValueError as err:
    print("refused:", err)
```

```
half-width q = 0.2041
intervals: [3.9959 4.7959] to [4.4041 5.2041]
exact coverage for n = 39: 0.9
refused: 5 calibration scores cannot certify level 0.9: the required rank 6 exceeds n. Collect at least 9 calibration measurements, or lower the confidence
```

`q` is the `k`-th smallest score, with `k = ceil((n + 1)(1 - alpha))`.
A new measurement that is exchangeable with the calibration
measurements then lands within `q` of its prediction with probability
at least `1 - alpha`, whatever the model and whatever the noise. For
scores without ties that probability is exactly
`k / (n + 1)`, which `coverage_exact` returns. The guarantee is an
average over many calibration sets and new points, not a promise for
each single setting.

### 7. Save and load measurements

```python
import os, tempfile
import numpy as np
from labplan import save_measurements_csv, load_measurements_csv

x = np.array([[0.5], [1.0], [1.5]])
y = np.array([1.1, 2.1, 3.1])
path = os.path.join(tempfile.mkdtemp(), "run.csv")
save_measurements_csv(path, x, y, sigmas=0.05)
print(open(path).read())
x2, y2, s2 = load_measurements_csv(path)
print(np.array_equal(x, x2), np.array_equal(y, y2), s2)
```

```
x1,y,sigma
0.5,1.1,0.05
1.0,2.1,0.05
1.5,3.1,0.05

True True [0.05 0.05 0.05]
```

Columns are the settings `x1, x2, ...`, the reading `y` and, if
given, the reading error `sigma`. Values are written in full
precision (`repr`), so they read back as the identical numbers.

## What is in the package

**The model**

- `Model(name, f, param_names, reference, units="")` -- your
  prediction function `f(theta, x)` with a name, one name per
  parameter, and a required `reference` (a paper, a manual, your own
  derivation note). `units` is an optional free-text note kept on the
  model. `Model.predict(theta, x)` evaluates `f` with shape and
  finiteness checks; `Model.jacobian(theta, x)` gives the slope of
  each prediction with respect to each parameter by central
  differences (it nudges one parameter a small step up and down and
  divides the change in the predictions by the distance between the
  two parameter values). The step is 1e-6 times the parameter's size;
  when that is too small to change the predictions measurably (a
  parameter that is zero, or tiny compared with the rest of the
  prediction), it uses the step of version 0.1.0 instead: 1e-6 times
  the larger of the parameter's size and 1e-3.
  `Model.n_params` counts the parameters.

**Planning**

- `information(model, theta, x, sigmas=None)` -- the information
  matrix of a planned design at your guessed parameters, whether it
  is identifiable, its condition number, the expected error bar of
  each parameter and the covariance. Without `sigmas` the error bars
  are per unit reading error.
- `design(model, theta, candidates, n_pick, sigmas=None)` -- picks
  `n_pick` of the candidate settings, one at a time, each time taking
  the candidate that makes the determinant of the information matrix
  largest (greedy D-optimal selection; F. Pukelsheim, Optimal Design
  of Experiments, SIAM (2006)). Returns the chosen indices in pick
  order, the information matrix, the condition number and the error
  bars.
- `repeats_for(target_sigma, plan)` -- the number of repeats `r` of a
  planned design needed to bring every error bar down to a target,
  from the rule that `r` repeats divide the error bars by `sqrt(r)`.
  The target is one number for all parameters, or a dictionary
  `{name: target}`. Returns `r` and the error bars at `r` repeats.

**Fitting**

- `fit(model, x, y, theta0, sigmas=None, max_iter=200, tol=1e-12)` --
  weighted least squares by the Levenberg-Marquardt method (a
  standard step-by-step search that adjusts the parameters until the
  weighted sum of squared differences between readings and
  predictions stops decreasing), written in NumPy. With `sigmas`, the covariance is the inverse information
  matrix and chi2 is reported; without them, the covariance is scaled
  by the residual scatter (chi2 / degrees of freedom), which needs at
  least one more reading than parameters.
- `FitResult` -- what `fit` returns: `values` and `sigma`
  (dictionaries by parameter name), `theta` and `cov` (arrays),
  `chi2`, `chi2_dof`, `n_points`, `condition_number`, `n_iter`,
  `model_name`, `reference` and `data_digest` (the sha256 of the
  settings, readings and reading errors).

**When the model may be wrong** (split conformal prediction)

- `conformal_quantile(scores, alpha=0.1)` -- the half-width `q` from
  held-out absolute errors.
- `conformal_interval(prediction, q)` -- `prediction - q` and
  `prediction + q`.
- `coverage_exact(n, alpha)` -- the exact coverage `k / (n + 1)` for
  scores without ties; it always lies between `1 - alpha` and
  `1 - alpha + 1/(n + 1)`.

**Records**

- `audit_record(result, operator="", note="")` -- a JSON-ready
  dictionary describing one fit (see example 5).
- `report_text(record)` -- the same record as readable text.
- `save_measurements_csv(path, x, y, sigmas=None)` and
  `load_measurements_csv(path)` -- a plain CSV file of measurements;
  loading returns `(x, y, sigmas)`, with `sigmas` `None` when the file
  has no sigma column.
- `__version__` -- the package version.

Each function's docstring (`help(labplan.fit)`, for example) gives
its inputs and conventions.

## When it refuses, and why

`labplan` raises an error instead of guessing when:

- a `Model` has an empty name, a prediction function that cannot be
  called, missing or repeated parameter names, or a `reference`
  shorter than 8 characters;
- the prediction function returns the wrong number of values, or
  values that are not finite (infinite or NaN);
- `theta` has the wrong number of entries, or settings, readings or
  reading errors are not finite, or reading errors are not positive;
- the design cannot tell the parameters apart (condition number above
  1e10 after rescaling, or a parameter that no reading responds to):
  `fit` refuses; `design` refuses when even the full candidate list
  cannot; `information` does not raise but reports
  `identifiable: False` with no error bars, and also does so when
  there are fewer settings than parameters;
- `fit` gets fewer readings than parameters, or, without `sigmas`,
  no spare reading to estimate the scatter from;
- `fit` does not converge within `max_iter` steps (a `RuntimeError`);
- `design` is asked for fewer picks than parameters or more picks
  than candidates;
- `repeats_for` gets a plan that is not identifiable (no number of
  repeats can fix that), an unknown parameter name, or a target that
  is not positive;
- `conformal_quantile` gets negative or non-finite scores, an alpha
  outside 0 to 1, or too few scores for the level (the message names
  the minimum); `coverage_exact` refuses the same levels;
  `conformal_interval` refuses a negative or non-finite `q`;
- `audit_record` gets an operator or note that is not text, or
  `report_text` gets something that is not an audit record;
- a measurement file is empty, has an unexpected header, has no data
  rows, has a row with the wrong number of values, or has a value that
  is not a number.

## How the results are checked

16 automated tests run on every push and pull request, on Python 3.9,
3.10, 3.11, 3.12, 3.13 and 3.14, and once more on Python 3.9 with the
oldest NumPy the package allows (1.22.0, with pytest 7.0.0). The
numerical checks compare against a formula, an identity or seeded
simulation computed in the test itself, not against numbers stored
from an earlier run. What the tests assert:

**Planning and fitting**

- On noiseless straight-line data with reading error 0.05, the fit
  recovers both parameters to 1e-8, chi2 is below 1e-12 with 10
  degrees of freedom, the covariance matches the textbook formula
  `sigma^2 (X^T X)^-1` (NumPy `allclose` with relative tolerance 1e-6
  and absolute 1e-15), and the planned error bars from `information`
  equal the fitted ones to 1 part in 10^6.
- In 400 seeded simulated experiments (reading error 0.2), the
  spread of the fitted parameters matches the reported error bars
  within 15 %.
- A nonlinear model (exponential decay) is fitted from a poor start
  and recovers both parameters to 1e-6 on noiseless data.
- A model whose two parameters only appear as a sum is reported not
  identifiable by `information` (the smallest singular value of the
  rescaled matrix is below 1e-10 of the largest), and is refused by
  `fit` and by `design`.
- The same RC charging model written with its time constant in
  seconds (1e-9) and in nanoseconds (1) gives the same planned error
  bars to 1 part in 10^6 and the same noiseless fit (values to
  1 part in 10^6), and the slopes of a model that divides by a
  1e-9 parameter are finite and match the exact derivative to 1 part
  in 10^6 (added in 0.1.1; see below).
- A parameter near zero still gets usable slopes: for the straight
  line with slope 1e-12, 1e-300 or 0 (intercept 1), `Model.jacobian`
  matches the exact slopes to 1 part in 10^6, and a noiseless fit of
  a flat line (true slope 0) recovers both parameters to 1e-8 with
  the textbook covariance (relative tolerance 1e-6; added in 0.1.1).
- Repeating a design 9 times divides every planned error bar by 3, to
  1 part in 10^9; the `r` that `repeats_for` returns meets the target
  and `r - 1` repeats would not.

**Design**

- `design` returns 5 distinct picks from 15 candidates, and their
  information determinant is at least that of each of 30 random
  5-point subsets.
- Asked for 2 picks on the straight line, it returns the pair with
  the largest determinant among all 105 pairs, found by checking each
  one.
- The identity `det(A + g g^T) = det(A) (1 + g^T A^-1 g)`, which
  explains why each greedy pick can only add information, holds to
  1 part in 10^9 on 20 random matrices. (This checks the identity,
  not the `design` function.)

**Conformal prediction**

- `conformal_quantile` equals the `k`-th smallest score, compared with
  `==`, for alpha 0.05, 0.1 and 0.25.
- `coverage_exact(29, 0.1)` lies between `1 - alpha` and
  `1 - alpha + 1/(n + 1)`, and 4000 seeded simulated trials hit it
  within 4 binomial standard deviations.
- Too few scores, alpha outside 0 to 1 and negative scores are
  refused; `conformal_interval` gives `prediction -/+ q`.

**Records and files**

- An audit record is unchanged, compared with `==`, after
  `json.dumps` and `json.loads`; its digest is the fit's; changing one
  reading by 1e-9 changes the digest; `report_text` contains the
  expected lines and refuses a non-record.
- Saving and loading a CSV file with 3 setting columns and sigmas
  gives back identical arrays (`np.array_equal`); a file with 1
  setting column and no sigmas loads back with the right shape and
  `sigmas` `None`; a wrong header is refused.
- The version in the package matches `pyproject.toml` and
  `CITATION.cff`, and every name in `__all__` exists.

The input refusals of `Model`, `fit` and `design` (short reference,
non-callable function, repeated names, too few points, negative
sigmas, too few picks, infinite parameters) are each checked to fire.
`Model.jacobian` is compared with exact derivatives only in the two
0.1.1 tests above.

## Corrections in earlier versions

**0.1.1 fixed a unit-dependence in the derivative step.** In 0.1.0,
`Model.jacobian` (used by `information`, `design` and `fit`) never
used a step smaller than 1e-9 in absolute terms. For a parameter
smaller than 1e-3 in its own units that step was too coarse, and for
a time constant of 1e-9 s it was as large as the parameter itself.
The planned error bars then depended on the units the parameter was
written in, a noiseless fit could stop away from the truth (0.987 and
0.938 ns instead of 1 and 1 ns in the new test's model), and a model
that divides by such a parameter was refused as non-finite. The step
is now 1e-6 times the parameter's size, falling back to the 0.1.0
step only when that relative step is too small to change the
predictions measurably (a parameter that is exactly zero, or one that
is tiny next to the rest of the prediction, such as a slope of 1e-12
beside an offset of 1). Results for parameters of size 1e-3 or more
do not change. If you used 0.1.0 with parameters smaller than 1e-3
in their own units, please re-run those results.

Also in 0.1.1: CI now runs Python 3.10 (claimed but not tested
before) and the oldest allowed NumPy. The 0.1.0 release notes said
the tests recompute the greedy rule (they do not; the checks are the
ones listed above) and that planner and fit agree exactly (the tests
assert 1 part in 10^6).

The full history is in [CHANGELOG.md](CHANGELOG.md).

## Limits

- The error bars assume independent, Gaussian ("bell-curve") reading
  errors. For a model that is not a straight-line function of its
  parameters, they come from a linear approximation around the
  guessed or fitted parameters and are approximate; the planned error
  bars also depend on how good your guess is.
- Slopes are computed by finite differences, not exact derivatives.
  A parameter that is exactly zero, or too small for the relative
  step to show in the predictions, gets an absolute step of 1e-9,
  which can be too large or too small for a parameter whose natural
  size is far from 1 (for example a time offset of zero written in
  seconds when the times involved are nanoseconds); write such a
  parameter in units where its natural size is near 1, or start it
  away from zero.
- `design` is a greedy heuristic: it adds the best single setting at
  each step and does not prove that the chosen set is the best
  possible. It picks each candidate at most once; use `repeats_for`
  for repeats.
- The conformal guarantee is an average over calibration sets and new
  measurements, not a promise at each setting, and it needs the
  calibration data to be exchangeable with the new measurement:
  calibration data from last month's instrument state do not certify
  next month's drift. The tighter bound `1 - alpha + 1/(n + 1)`
  assumes scores without ties.
- The `units` note of a `Model` is not written into the audit record
  or the text report.
- No physics ships with the package. Your model and its `reference`
  carry the physics.

## Where it comes from

Nine research packages in this organization -- covering colour-centre
spins, squeezed light, spin-squeezed clocks, Raman maps, single-photon
emitters, superconducting detectors, band structure, semiconductor
heterostructures and photonic fabrication -- each grew the same
planning-and-calibration loop for its own physics. `labplan` is that
loop extracted and generalized into one package that depends only on
NumPy.

The statistics are standard. Weighted least squares and the
information matrix are textbook material (e.g. Cox & Hinkley,
Theoretical Statistics (1974); any statistics text under
"Cramer-Rao bound"). D-optimal design follows F. Pukelsheim, Optimal
Design of Experiments, SIAM (2006). Split conformal prediction follows
Vovk, Gammerman and Shafer, Algorithmic Learning in a Random World,
Springer (2005); Lei et al., J. Am. Stat. Assoc. 113, 1094 (2018);
and Angelopoulos and Bates, arXiv:2107.07511.

## Citing, support and license

If `labplan` helps your work, please cite it with the concept DOI
[10.5281/zenodo.22826765](https://doi.org/10.5281/zenodo.22826765),
which always resolves to the latest release.
[CITATION.cff](CITATION.cff) has the details.

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and takes the final decision on scope and releases.
Design questions are discussed in the open in issues and pull
requests. Usage questions and bug reports are welcome in the
[issue tracker](https://github.com/TaN-MM-Org/labplan/issues). A
docstring that leaves a unit or a convention unclear counts as a
documentation bug, not user error. The standing rule of
[CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly as it
binds contributors: a change that touches the statistics arrives with
a test, and a claim arrives with its source. While the version is
below 1.0 the programming interface may still change between minor
versions; such changes are called out in the release notes.

Licensed under Apache-2.0.

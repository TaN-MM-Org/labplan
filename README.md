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
  bars? Are the readings consistent with the model and their stated
  errors? Do the residuals wander in a pattern the model misses? What
  is the error bar of anything computed from them, such as a predicted
  reading? And, reading the calibration backwards, which setting
  produced a new reading, and how sure is that? The readings can come
  straight from an instrument's exported text table.
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
- [Tutorials](#tutorials) (complete scripts, from data to result)
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
- **Slopes (Jacobian)** -- how much each prediction changes per unit
  change of each parameter. `labplan` computes them numerically by
  default, or uses exact ones you supply.
- **Propagation of uncertainty** -- turning the error bars (and
  covariance) of the parameters into the error bar of something
  computed from them, to first order: exact when that quantity is a
  straight-line function of the parameters, approximate otherwise.
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
- **p-value**: here, the probability that chi2 would come out at
  least as large as it did if the model and the stated reading errors
  were right. A very small p-value means one of them is wrong.
- **Residuals and runs**: the residuals are the differences
  `reading - prediction`. Ordered by the setting, a **run** is a block
  of residuals with the same sign. A model that misses a bend or a
  drift leaves long runs, so too few runs is a warning sign.
- **Inverse prediction**: reading a calibration curve backwards. A
  new reading `y0` is turned into the setting `x0` at which the model
  predicts `y0`; its error bar counts both the error of the new
  reading and the uncertainty of the calibration.
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
79 calibration points). Two longer tutorials, also in `examples/`,
are described in [Tutorials](#tutorials) below.

## Examples

Each example below runs as written, and the output shown is what it
printed with labplan 0.4.0. The model, the numbers and the noise are
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
['chi2', 'chi2_dof', 'condition_number', 'covariance', 'data_sha256', 'model', 'model_reference', 'model_units', 'n_points', 'note', 'operator', 'parameters', 'record', 'software', 'timestamp_utc']
dbf50525b0aff6ea ...
survives JSON round trip: True
```

The record is a plain dictionary: model name, reference and units
note (the model's `units`, empty here), each value with its error bar, the covariance, chi2, the number of points,
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

### 8. Error bars of predictions and derived numbers

```python
import numpy as np
from labplan import Model, fit, propagate

line = Model("sensor line",
             lambda th, x: th[0] * x[:, 0] + th[1],
             param_names=("gain", "offset"),
             reference="illustrative straight-line sensor model")
x = np.linspace(0.5, 5.0, 12)
rng = np.random.default_rng(1)
y = 2.0 * x + 0.1 + 0.05 * rng.standard_normal(x.size)
res = fit(line, x, y, theta0=[1.0, 0.0], sigmas=0.05)

# The predicted reading at settings 2.5 and 7.0 (7.0 is outside the
# measured range, so its error bar is larger).
pred = propagate(lambda th: line.predict(th, [2.5, 7.0]),
                 res.theta, res.cov)
print("predictions:", np.round(pred["value"], 4),
      "+/-", np.round(pred["sigma"], 4))

# The setting at which the sensor reads 5.0: x = (5.0 - offset) / gain.
x5 = propagate(lambda th: (5.0 - th[1]) / th[0], res.theta, res.cov)
print(f"setting for a reading of 5.0: {x5['value']:.4f} +/- {x5['sigma']:.4f}")
```

```
predictions: [ 5.1116 14.114 ] +/- [0.0147 0.0458]
setting for a reading of 5.0: 2.4442 +/- 0.0074
```

`propagate(func, theta, cov)` takes any function of the parameters
and returns its value, error bar and covariance, using the
parameter covariance (and so the correlation between gain and
offset, which a simple "add the error bars" rule would miss). The
predictions are straight-line functions of the parameters, so their
error bars are exact; the setting `(5.0 - offset) / gain` is not, so
its error bar is the usual first-order approximation. Pass
`information(...)["cov"]` instead of `res.cov` to get the planned
error bar of a derived number before measuring.

### 9. Better picks by swapping, and repeats of a chosen design

```python
import numpy as np
from labplan import Model, design, repeats_for

decay = Model("decay with background",
              lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]) + th[2],
              param_names=("amplitude", "tau", "background"),
              reference="illustrative exponential decay with background")
guess = [1.0, 1.0, 0.2]
candidates = np.linspace(0.0, 3.0, 31)       # 0.0, 0.1, ..., 3.0

greedy = design(decay, guess, candidates, n_pick=3, sigmas=0.01)
swapped = design(decay, guess, candidates, n_pick=3, sigmas=0.01,
                 exchange=True)
for name, out in (("greedy", greedy), ("exchange", swapped)):
    print(f"{name:8s} settings {np.sort(candidates[out['indices']])}"
          f"  error bar on tau {out['sigma']['tau']:.4f}")

r, predicted = repeats_for({"tau": 0.01}, swapped)
print("repeats of the 3-point exchange design for tau +/- 0.01:", r)
```

```
greedy   settings [0.  1.1 3. ]  error bar on tau 0.0481
exchange settings [0.  0.8 3. ]  error bar on tau 0.0451
repeats of the 3-point exchange design for tau +/- 0.01: 21
```

The greedy picks are made one at a time and never revisited.
`exchange=True` then tries replacing each chosen setting by each
unused one and keeps the best swap, until no swap increases the
information determinant. Here it moves the middle setting from 1.1
to 0.8; for this example, checking all 4495 possible triples one
by one finds the same best triple (a test asserts this). The output of `design` can be
passed straight to `repeats_for`.

### 10. Exact slopes, and a parameter that starts at zero

```python
import numpy as np
from labplan import Model, information

# A Gaussian pulse, with times written in seconds: center t0, width w.
def pulse(th, x):
    return th[0] * np.exp(-0.5 * ((x[:, 0] - th[1]) / th[2]) ** 2)

def pulse_slopes(th, x):              # d(pulse)/d(A, t0, w), by hand
    u = (x[:, 0] - th[1]) / th[2]
    e = np.exp(-0.5 * u ** 2)
    return np.column_stack([e, th[0] * e * u / th[2],
                            th[0] * e * u ** 2 / th[2]])

numeric = Model("pulse", pulse, ("A", "t0", "w"),
                reference="illustrative Gaussian pulse")
exact = Model("pulse", pulse, ("A", "t0", "w"),
              reference="illustrative Gaussian pulse", jac=pulse_slopes)

t = np.linspace(-3e-9, 3e-9, 25)        # 25 samples from -3 ns to 3 ns
guess = [1.0, 0.0, 1e-9]                # center exactly zero, width 1 ns
diff = np.abs(exact.jacobian(guess, t) - numeric.numeric_jacobian(guess, t))
print("largest slope difference / largest slope:",
      f"{diff.max() / np.abs(exact.jacobian(guess, t)).max():.1e}")
for m in (numeric, exact):
    s = information(m, guess, t, sigmas=0.01)["sigma"]["t0"]
    print(f"planned error bar on t0 ({'exact' if m.jac else 'numeric'} slopes): {s:.6e} s")
```

```
largest slope difference / largest slope: 3.1e-07
planned error bar on t0 (numeric slopes): 5.311778e-12 s
planned error bar on t0 (exact slopes): 5.311777e-12 s
```

`jac=` gives the model exact slopes, which `information`, `design`
and `fit` then use; `numeric_jacobian` always computes the numerical
ones, so you can compare the two once, as here, to catch a mistake
in the hand-written slopes. The numerical slopes are also good here,
although the center `t0` is exactly zero and there is no natural
step size for it (see "Corrections" below: version 0.1.1 printed
6.686976e-12 s, 26 % too large, for the numeric line).

### 11. Are the readings consistent with the model?

```python
import numpy as np
from labplan import Model, fit, goodness_of_fit

line = Model("sensor line",
             lambda th, x: th[0] * x[:, 0] + th[1],
             param_names=("gain", "offset"),
             reference="illustrative straight-line sensor model")
x = np.linspace(0.5, 5.0, 12)
rng = np.random.default_rng(1)
y = 2.0 * x + 0.1 + 0.05 * rng.standard_normal(x.size)   # true error 0.05

for stated in (0.05, 0.02):            # the second is understated
    res = fit(line, x, y, theta0=[1.0, 0.0], sigmas=stated)
    g = goodness_of_fit(res)
    print(f"sigmas={stated}: chi2 = {g['chi2']:.2f} for {g['dof']} "
          f"degrees of freedom, p = {g['p_value']:.3g}, "
          f"Birge ratio = {g['birge_ratio']:.2f}")

# A curved response fitted with a straight line, with the right errors:
y_curved = 2.0 * x + 0.1 + 0.05 * x ** 2 + 0.05 * rng.standard_normal(x.size)
g = goodness_of_fit(fit(line, x, y_curved, [1.0, 0.0], sigmas=0.05))
print(f"curved data: chi2 = {g['chi2']:.1f}, p = {g['p_value']:.2g}")
```

```
sigmas=0.05: chi2 = 4.10 for 10 degrees of freedom, p = 0.943, Birge ratio = 0.64
sigmas=0.02: chi2 = 25.62 for 10 degrees of freedom, p = 0.00429, Birge ratio = 1.60
curved data: chi2 = 49.0, p = 4e-07
```

`goodness_of_fit` needs a fit made with `sigmas`. Its `p_value` is
the probability of a chi2 at least this large if the model and the
stated errors are both right. With the true reading error (0.05) the
fit is unremarkable. With an error stated 2.5 times too small, chi2 is
6.25 times larger and p falls to 0.4 %. A straight line fitted to
readings that bend gives p = 4e-07 although the errors are right. The
test cannot say which of the two is wrong, only that something is.
`p_value_low` (also returned) is the opposite tail: a tiny value
means the readings agree better than their errors allow, so the
errors are probably overstated. The Birge ratio `sqrt(chi2 / dof)` is
the factor by which the stated errors would have to change for chi2
to equal its degrees of freedom. You can also call
`goodness_of_fit(chi2=..., dof=...)` with numbers from any other fit.

### 12. Read an instrument's exported table

```python
import os, tempfile
from labplan import Model, fit, load_columns

# A sweep as an instrument might export it (illustrative numbers):
# a comment line, a header, semicolons, decimal commas, extra columns.
export = """# leakage sweep, bench 2
Time (s);Bias (V);Current (A);Current error (A);Status
0;0,5;1,04E-12;5E-14;OK
2;1,0;2,01E-12;5E-14;OK
4;1,5;3,05E-12;5E-14;OK
6;2,0;3,98E-12;5E-14;OK
8;2,5;5,02E-12;5E-14;OK
"""
path = os.path.join(tempfile.mkdtemp(), "sweep.csv")
with open(path, "w") as fh:
    fh.write(export)

x, y, s = load_columns(path, x="Bias (V)", y="Current (A)",
                       sigma="Current error (A)", decimal=",")
print("bias:", x[:, 0], " current:", y)

leak = Model("leakage", lambda th, x: th[0] * x[:, 0] + th[1],
             param_names=("G", "I0"),
             reference="illustrative ohmic leakage with an offset current",
             units={"G": "S", "I0": "A"})
res = fit(leak, x, y, theta0=[0.0, 0.0], sigmas=s)
print(f"G = {res.values['G']:.4e} +/- {res.sigma['G']:.1e} S")

# A cell the instrument marked as overload is refused, not skipped:
with open(path, "a") as fh:
    fh.write("10;3,0;OVLD;5E-14;OVERLOAD\n")
try:
    load_columns(path, x="Bias (V)", y="Current (A)", decimal=",")
except ValueError as err:
    print("refused:", err)
```

```
bias: [0.5 1.  1.5 2.  2.5]  current: [1.04e-12 2.01e-12 3.05e-12 3.98e-12 5.02e-12]
G = 1.9860e-12 +/- 3.2e-14 S
refused: line 8, column 'Current (A)': 'OVLD' is not a number
```

`load_columns` reads the columns you name from a delimited text file
with a header row: comma, tab, semicolon or space separated (found
from the header, or given with `delimiter=`), with `.` or `,` as the
decimal mark. Comment lines (starting with `#`) and blank lines are
skipped, and `skip_rows=` skips a preamble that is not marked as
comment. Other columns may hold anything and are not read. Every
value it returns was read as a finite number from the cell you named.
An empty cell, text such as `OVLD`, `nan` or `inf`, or a row with
the wrong number of fields stops it with the line number, because
skipping such rows silently would change the data you fit. The
result `(x, y, sigmas)` goes straight into `fit`.

### 13. Read the calibration backwards: which setting gave this reading?

```python
import numpy as np
from labplan import Model, fit, inverse_predict

line = Model("sensor line",
             lambda th, x: th[0] * x[:, 0] + th[1],
             param_names=("gain", "offset"),
             reference="illustrative straight-line sensor model")
x = np.linspace(0.5, 5.0, 12)
rng = np.random.default_rng(1)
y = 2.0 * x + 0.1 + 0.05 * rng.standard_normal(x.size)
res = fit(line, x, y, theta0=[1.0, 0.0], sigmas=0.05)

# An unknown gives the reading 5.0 (error bar 0.05): which setting?
out = inverse_predict(line, res, reading=5.0, bracket=(0.5, 5.0),
                      reading_sigma=0.05)
print(f"setting = {out['value']:.4f} +/- {out['sigma']:.4f}")
print(f"  from the reading: {out['sigma_from_reading']:.4f}, "
      f"from the calibration: {out['sigma_from_parameters']:.4f}")

# A reading the calibrated range does not reach is refused:
try:
    inverse_predict(line, res, reading=12.0, bracket=(0.5, 5.0))
except ValueError as err:
    print("refused:", err)
```

```
setting = 2.4442 +/- 0.0261
  from the reading: 0.0250, from the calibration: 0.0074
refused: the model does not reach the reading 12 for settings between 0.5 and 5 (there it runs from 1.11061 to 10.1129); widen the bracket, or check the reading
```

`inverse_predict` (new in 0.4.0) solves `f(theta, x0) = reading` for
`x0` inside the `bracket` you give, normally the calibrated range. The
error bar is first order: `u(x0)^2 = (u(y0)^2 + a^T C a) / s^2`, with
`s` the slope of the curve at `x0` and `a` the slopes with respect to
the parameters. For a straight line this is the propagation-of-error
formula of the NIST/SEMATECH e-Handbook of Statistical Methods,
section 2.3.6.7.3. The two parts show what limits the result: here the
reading (0.0250) far more than the calibration (0.0074), so repeating
the reading of the unknown would help, and recalibrating would not.
The value is the setting example 8 found with `propagate`; the error
bar is larger because it now includes the reading's own error. Pass
an array of readings to get the covariance of several unknowns (they
share the calibration, so their errors are correlated), or
`theta=` and `cov=` from `information` to plan before calibrating. A
reading that the model reaches twice in the bracket (a peak), a curve
that is flat there, and a curve that bends so much within one error
bar that the first-order error bar is unreliable are refused.

### 14. Do the residuals wander? The runs test

```python
import numpy as np
from labplan import Model, fit, goodness_of_fit, runs_test

line = Model("sensor line",
             lambda th, x: th[0] * x[:, 0] + th[1],
             param_names=("gain", "offset"),
             reference="illustrative straight-line sensor model")
x = np.linspace(0.5, 5.0, 24)
rng = np.random.default_rng(2)
# Readings that bend a little, with errors stated 3 times too large:
y = 2.0 * x + 0.1 + 0.05 * x ** 2 + 0.05 * rng.standard_normal(x.size)
res = fit(line, x, y, theta0=[1.0, 0.0], sigmas=0.15)
print(f"chi2 test: p = {goodness_of_fit(res)['p_value']:.3f}")

r = runs_test(y - line.predict(res.theta, x), order=x)
print(f"signs: {r['n_positive']} +, {r['n_negative']} -, "
      f"{r['runs']} runs (expected {r['expected_runs']:.2f} "
      f"+/- {r['sigma_runs']:.2f}), p = {r['p_value']:.4f}")
```

```
chi2 test: p = 0.994
signs: 9 +, 15 -, 5 runs (expected 12.25 +/- 2.24), p = 0.0010
```

The readings bend a little, and their stated errors are 3 times too
large, so chi2 is small and its test sees nothing wrong (p = 0.994).
The residuals of the straight line, ordered by `x`, tell another
story: positive at both ends and negative in the middle, they form
only 5 runs where about 12 are expected. `runs_test` (new in 0.4.0)
gives the probability of so few runs if the signs were in random order
(the runs test of A. Wald and J. Wolfowitz, Ann. Math. Statist. 11,
147 (1940)), computed exactly. `p_value_high` is the opposite tail
(signs that alternate too often). For residuals of a fit the signs are
not exactly independent; in the tests this made the test more
cautious, not less: on 4000 seeded correct straight-line fits each
with 12 and with 30 readings it raised a false alarm at the 5 % level
in 0.7 % and 2.8 % of them (3.6 % for 30 independent random signs,
where the exact test, whose runs come in whole numbers, stays at or
below 5 %).

## Tutorials

Two complete scripts in [`examples/`](examples/) go from raw data
(simulated with a fixed seed, so the output is repeatable) to a final
result with error bars. Each runs in well under a second, and a test
checks that it still prints exactly what is shown here.

[`tutorial_calibration_curve.py`](examples/tutorial_calibration_curve.py)
calibrates an illustrative pressure sensor with a curved response
(`V = c0 + c1 P + c2 P^2`, 11 reference pressures from 0 to 100 kPa),
plans how well an unknown will be read before calibrating, checks the
calibration with chi-squared and the runs test, and then reads an
unknown (5 readings, true pressure 50 kPa) off the curve:

```
planned error bar at 50 kPa from one reading: 0.0161 kPa
fitted c0 = 0.09988 +/- 0.00046 V
fitted c1 = 0.039960 +/- 2.1e-05 V/kPa
fitted c2 = 9.9567e-06 +/- 2e-07 V/kPa^2
chi2 = 6.93 for 8 degrees of freedom, p = 0.544
runs of residual signs: 8 (expected 6.5), p = 0.911
unknown: mean reading 2.12274 V +/- 0.00027 V (5 readings)
pressure = 49.9988 +/- 0.0094 kPa
  from the reading noise:       0.0066 kPa
  from the calibration curve:   0.0067 kPa
```

[`tutorial_decay_on_background.py`](examples/tutorial_decay_on_background.py)
measures a 10 mV decay with a time constant near 1 ns on a steady 1 V
level, in seconds and volts. It plans how many traces to average for
an error bar of 10 ps on the time constant, fits the averaged trace,
checks the fit and reports the decay rate `1/tau` with its error bar.
labplan 0.3.0 refused this model at the natural first guess
`tau = 1e-9` s (see Corrections):

```
planned error bar on tau from one trace: 3.57e-11 s
traces to average for tau +/- 1e-11 s: 13
A   = 0.0101 +/- 5.9e-05 V
tau = 9.9383e-10 +/- 9.8e-12 s
B   = 0.999993 +/- 1.3e-05 V
chi2 = 150.0 for 197 degrees of freedom, p = 0.995
runs of residual signs: 105 (expected 101.0), p = 0.740
decay rate 1/tau = 1.006e+09 +/- 9.9e+06 1/s
```

## What is in the package

**The model**

- `Model(name, f, param_names, reference, units="", jac=None)` --
  your prediction function `f(theta, x)` with a name, one name per
  parameter, and a required `reference` (a paper, a manual, your own
  derivation note). `units` is an optional note, free text or a
  dictionary such as `{"fc": "Hz"}`, that is copied into fit
  results, audit records and text reports. `jac` is an optional
  function `jac(theta, x)` returning the exact slopes as an `(n, p)`
  array (one column per parameter); when given it replaces the
  numerical slopes everywhere.
- `Model.predict(theta, x)` evaluates `f` with shape and finiteness
  checks. `Model.jacobian(theta, x)` gives the slope of each
  prediction with respect to each parameter: from `jac` if the model
  has one, otherwise from `Model.numeric_jacobian(theta, x)`.
- `Model.numeric_jacobian(theta, x)` computes the slopes by central
  differences (it nudges one parameter a small step up and down and
  divides the change in the predictions by the distance between the
  two parameter values). The step is 1e-6 times the parameter's
  size. When that is too small to change the predictions measurably
  (a parameter that is zero, or tiny compared with the rest of the
  prediction), it uses the step of version 0.1.0 instead: 1e-6 times
  the larger of the parameter's size and 1e-3. Since 0.2.0 that
  fallback step is checked against a step 10 times smaller; while
  the two disagree by more than 1e-6 of the largest slope (plus an
  allowance for rounding), the smaller step is taken, at most 8
  times. Since 0.4.0, when that fallback step reaches or crosses zero
  and the model returns values that are not finite there (a time
  constant of 1e-9 s, say), it is divided by 10 until they are finite,
  at most 8 times, and then checked the same way. `rel_step` (default
  1e-6) must be a positive number. `Model.n_params` counts the
  parameters.

**Planning**

- `information(model, theta, x, sigmas=None)` -- the information
  matrix of a planned design at your guessed parameters, whether it
  is identifiable, its condition number, the expected error bar of
  each parameter and the covariance. Without `sigmas` the error bars
  are per unit reading error.
- `design(model, theta, candidates, n_pick, sigmas=None,
  exchange=False)` -- picks `n_pick` of the candidate settings, one
  at a time, each time taking the candidate that makes the
  determinant of the information matrix largest (greedy D-optimal
  selection; F. Pukelsheim, Optimal Design of Experiments, SIAM
  (2006)). With `exchange=True` it then swaps one chosen setting for
  one unused setting while that increases the determinant (the
  exchange idea of V. V. Fedorov, Theory of Optimal Experiments,
  Academic Press (1972)). Returns the chosen indices (in pick order,
  a swapped-in setting taking the place of the one it replaced), the
  information matrix, `identifiable`, the condition number, the error
  bars and the covariance; the result can be passed to
  `repeats_for`. Since 0.4.0 the candidates are scored in batches,
  which gives the same picks, bit for bit, 9 to 24 times faster for
  400 to 3000 candidates (medians of 7 runs on a shared 2-CPU
  machine).
- `repeats_for(target_sigma, plan)` -- the number of repeats `r` of a
  planned design needed to bring every error bar down to a target,
  from the rule that `r` repeats divide the error bars by `sqrt(r)`.
  The target is one number for all parameters, or a dictionary
  `{name: target}`. Returns `r` and the error bars at `r` repeats.
  `plan` is the output of `information` or `design`.

**Error bars of derived numbers**

- `propagate(func, theta, cov)` -- the value, error bar and
  covariance of `func(theta)` (one number or an array of numbers
  computed from the parameters), by first-order propagation
  `J C J^T`, where `C` is the parameter covariance and `J` holds the
  numerical slopes of `func`. Use the covariance of a fit, or of a
  plan for the error bars you should expect. Values and error bars
  come back as plain numbers when `func` returns one number. Since
  0.3.0 a covariance that is not positive semidefinite (one that
  would give some combination of parameters a negative variance) is
  refused.

**Fitting**

- `fit(model, x, y, theta0, sigmas=None, max_iter=200, tol=1e-12)` --
  weighted least squares by the Levenberg-Marquardt method (a
  standard step-by-step search that adjusts the parameters until the
  weighted sum of squared differences between readings and
  predictions stops decreasing), written in NumPy. With `sigmas`, the
  covariance is the inverse information matrix and chi2 is reported;
  without them, the covariance is scaled by the residual scatter
  (chi2 / degrees of freedom), which needs at least one more reading
  than parameters. The search stops when a step changes the weighted
  sum of squares by at most `tol` times (one unit + the sum) and
  moves no parameter by more than 1e-10 times (1 + the largest
  parameter). With `sigmas` the unit is 1; without them, since 0.3.0,
  it is the residual variance (at most 1), so the result no longer
  depends on the units of the readings (see Corrections).
- `FitResult` -- what `fit` returns: `values` and `sigma`
  (dictionaries by parameter name), `theta` and `cov` (arrays),
  `chi2`, `chi2_dof`, `n_points`, `condition_number`, `n_iter`,
  `model_name`, `reference`, `units` (the model's note) and
  `data_digest` (the sha256 of the settings, readings and reading
  errors).
- `goodness_of_fit(result)` or `goodness_of_fit(chi2=..., dof=...)`
  (new in 0.3.0): the chi-squared test. It returns `chi2`, `dof`,
  `reduced_chi2` (chi2 / dof), `p_value` (probability of a chi2 at
  least this large if the model and the stated errors are right),
  `p_value_low` (probability of one at most this large) and
  `birge_ratio` (`sqrt(chi2 / dof)`; R. T. Birge, Phys. Rev. 40, 207
  (1932)). The probabilities are the regularized incomplete gamma
  functions `Q(dof/2, chi2/2)` and `P(dof/2, chi2/2)`, computed by
  the series and continued fraction of Press et al., Numerical
  Recipes, 3rd ed. (2007), section 6.2, so that small values keep
  their full relative precision. See example 11.
- `runs_test(residuals, order=None)` (new in 0.4.0): the runs test of
  Wald and Wolfowitz on the signs of the residuals, sorted by `order`
  (normally the settings). Returns `runs`, `n_positive`, `n_negative`,
  `n_zero` (zero residuals have no sign and are left out),
  `expected_runs`, `sigma_runs`, `p_value` (probability of this few
  runs or fewer when the signs are in random order) and `p_value_high`
  (this many or more). See example 14.

**Reading a calibration backwards**

- `inverse_predict(model, result, reading, bracket, reading_sigma=0.0,
  theta=None, cov=None, n_scan=1001, max_slope_change=0.1)` (new in
  0.4.0): the setting `x0` at which the model predicts `reading`,
  searched in `bracket = (low, high)` (scanned at `n_scan` points, then
  refined by bisection), with its error bar from the reading's own
  error `reading_sigma` and from the parameter covariance (of `result`,
  or `theta=` and `cov=`). Returns `value`, `sigma`, `cov` (for several
  readings), `slope` (`df/dx` at `x0`), `sigma_from_reading`,
  `sigma_from_parameters` and `slope_change`. The settings must be one
  number per measurement. See example 13.

**When the model may be wrong** (split conformal prediction)

- `conformal_quantile(scores, alpha=0.1)` -- the half-width `q` from
  held-out absolute errors.
- `conformal_interval(prediction, q)` -- `prediction - q` and
  `prediction + q`.
- `coverage_exact(n, alpha)` -- the exact coverage `k / (n + 1)` for
  scores without ties; it always lies between `1 - alpha` and
  `1 - alpha + 1/(n + 1)`.
- `alpha` is read as the decimal number you typed: `k` is computed so
  that `alpha=0.7` means exactly 7/10 (see Corrections below).

**Records**

- `audit_record(result, operator="", note="")` -- a JSON-ready
  dictionary describing one fit (see example 5), including the
  model's units note under `model_units`.
- `report_text(record)` -- the same record as readable text. Records
  written by 0.1.x, which have no `model_units`, still render.
- `save_measurements_csv(path, x, y, sigmas=None)` and
  `load_measurements_csv(path)` -- a plain CSV file of measurements;
  loading returns `(x, y, sigmas)`, with `sigmas` `None` when the file
  has no sigma column.
- `load_columns(path, x, y, sigma=None, delimiter=None, decimal=".",
  comment="#", skip_rows=0, header=True)` (new in 0.3.0): reads
  the named columns (header names, or positions counted from 0) of
  any delimited text table, such as an instrument's or a
  spreadsheet's export, and returns `(x, y, sigmas)` like
  `load_measurements_csv`. `x` can be one column or a list of
  columns. See example 12.
- `__version__` -- the package version.

Each function's docstring (`help(labplan.fit)`, for example) gives
its inputs and conventions.

## When it refuses, and why

`labplan` raises an error instead of guessing when:

- a `Model` has an empty name, a prediction function (or `jac`)
  that cannot be called, missing or repeated parameter names, or a
  `reference` shorter than 8 characters;
- `jac` returns an array of the wrong shape or values that are not
  finite;
- the prediction function returns the wrong number of values, or
  values that are not finite (infinite or NaN);
- `theta` has the wrong number of entries, or settings, readings or
  reading errors are not finite, or reading errors are not positive,
  or there is neither one reading error for all readings nor one per
  reading (since 0.2.0 the message says so, instead of NumPy's
  broadcasting error);
- the design cannot tell the parameters apart (condition number above
  1e10 after rescaling, or a parameter that no reading responds to):
  `fit` refuses; `design` refuses when even the full candidate list
  cannot; `information` does not raise but reports
  `identifiable: False` with no error bars, and also does so when
  there are fewer settings than parameters;
- `fit` gets fewer readings than parameters, or, without `sigmas`,
  no spare reading to estimate the scatter from;
- `fit` does not converge within `max_iter` steps (a `RuntimeError`),
  or gets a `max_iter` below 1 or a negative `tol` (since 0.3.0);
- `design` is asked for fewer picks than parameters or more picks
  than candidates, or (since 0.3.0) for a number of picks that is not
  a whole number, such as 2.5, which 0.2.0 silently cut to 2;
- `repeats_for` gets a plan that is not identifiable (no number of
  repeats can fix that), an unknown parameter name, a target that
  is not positive, or (since 0.3.0, with a clear message) an empty
  dictionary of targets;
- `conformal_quantile` gets negative or non-finite scores, an alpha
  outside 0 to 1, or too few scores for the level (the message names
  the minimum); `coverage_exact` refuses the same levels;
  `conformal_interval` refuses a negative or non-finite `q`;
- `inverse_predict` finds no setting in the bracket at which the
  model reaches the reading (the message gives the range the model
  covers there), finds more than one (narrow the bracket to one
  branch), finds the model flat at the solution or equal to the
  reading over a whole range, or finds that the slope `df/dx` changes
  by more than `max_slope_change` (10 %) within one error bar of the
  solution; or gets a bracket that is not two finite numbers with
  low < high, a reading that is not finite, a negative
  `reading_sigma`, a `FitResult` of another model, both a result and
  `theta=`/`cov=`, or a covariance that `propagate` would refuse;
- `runs_test` gets residuals that are not finite or not a 1-D array,
  an `order` of the wrong length, fewer than 2 nonzero residuals, or
  residuals that all have the same sign (no runs to count);
- `Model.numeric_jacobian` gets a `rel_step` that is not a finite
  positive number (since 0.4.0; 0.3.0 failed with an
  `UnboundLocalError`), and `coverage_exact` gets an `n` that is not
  a whole number (since 0.4.0; 0.3.0 cut 29.9 to 29);
- `propagate` gets no covariance (`None`, as from a plan that is
  not identifiable), one of the wrong size, one that is not
  symmetric or has a negative diagonal, one that is not positive
  semidefinite (since 0.3.0), or a `func` that cannot be
  called, returns non-finite values, or returns an array with more
  than one dimension (a table instead of a list of numbers);
- `audit_record` gets an operator or note that is not text, or
  `report_text` gets something that is not an audit record;
- `goodness_of_fit` gets a fit made without `sigmas` (it has no
  chi2), zero degrees of freedom, a negative or non-finite chi2, a
  `dof` that is not a whole number, or both a fit and numbers;
- a measurement file is empty, has an unexpected header, has no data
  rows, has a row with the wrong number of values, or has a value that
  is not a number;
- `load_columns` is asked for a column name that is not in the header
  (the message lists the names that are), a name that appears twice,
  a position outside the table, or the same column twice; or finds a
  row with a different number of fields than the header, an empty
  cell, text, NaN or infinity in a requested column, a reading error
  that is not positive, a header in which more than one of comma, tab
  and semicolon occurs (pass `delimiter=`), a `.` in a number when
  `decimal=","` (a thousands separator?), or no data rows. Each
  message names the line.

## How the results are checked

96 automated tests run on every push and pull request, on Python 3.9,
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
  and `r - 1` repeats would not, and equals `ceil((sigma / target)^2)`
  for the largest ratio, with the predicted error bars `sigma /
  sqrt(r)` (relative 1e-12).
- Without `sigmas`, the fitted covariance equals the ordinary
  least-squares formula `s^2 (X^T X)^-1` with `s^2` = residual sum of
  squares / (n - p), and the values equal `np.linalg.lstsq` (relative
  tolerances 1e-6 and 1e-8). `information` without `sigmas` gives
  error bars exactly 1/0.05 of those with `sigmas=0.05` (relative
  1e-12), and fewer settings than parameters are reported not
  identifiable.
- Slopes of an exponential decay match the exact derivatives to
  relative 1e-6 (added in 0.2.0).

**Slopes: parameters at zero and exact slopes** (added in 0.2.0)

- A Gaussian pulse written in seconds with center `t0 = 0` and width
  1 ns: every column of the numerical slopes matches the exact
  derivative to 1e-5 of the column's largest value; the planned error
  bars match the exact `(J^T W J)^-1` to relative 1e-5, and equal
  those of the same model written in nanoseconds to relative 1e-5; a
  noiseless fit in seconds reports the exact error bar on `t0` to
  relative 1e-5. (With 0.1.1 the slope of `t0` was 31 % off and its
  planned error bar 26 % too large; this test fails on 0.1.1.)
- Where the 0.1.0 fallback step was already accurate (a slope of 0 or
  1e-300 beside an intercept of 1, a phase of 0 in a sine, a rate of
  0 in an exponential), the slopes are bit-for-bit the central
  difference with step 1e-9, recomputed by hand in the test; and a
  model that loses digits in its own arithmetic does not get worse
  slopes than that step gave.
- With `jac=`, `jacobian` returns exactly the supplied array, the
  numerical slopes agree with it to 1e-5 of the largest slope, the
  planned covariance equals the exact `(J^T W J)^-1` to relative
  1e-9 (entries compared on the scale `sqrt(C_ii C_jj)`), and a
  noiseless fit recovers the amplitude to 1e-9 and the width to 1
  part in 10^9. Wrong shapes, non-finite values and a non-callable
  `jac` are refused.

**Error bars of derived numbers** (added in 0.2.0)

- `propagate` of straight-line predictions at three settings equals
  the closed form `A C A^T` (relative 1e-6); at setting 0 it equals
  the offset's own error bar.
- For a product `a b` it equals the textbook first-order formula
  `b^2 C_aa + a^2 C_bb + 2 a b C_ab` (relative 1e-6).
- In 300 seeded simulated fits (reading error 0.2) the scatter of
  `4 gain + offset` matches the propagated error bar within 15 %.
- Propagating with the planned covariance equals propagating with the
  covariance of a noiseless fit (relative 1e-6).

**Design**

- `design` returns 5 distinct picks from 15 candidates, and their
  information determinant is at least that of each of 30 random
  5-point subsets.
- Asked for 2 picks on the straight line, it returns the pair with
  the largest determinant among all 105 pairs, found by checking each
  one; asked for 4 of 12 settings on a line, its first two picks are
  the two ends.
- With `exchange=True`, on 24 seeded problems (a decay with
  background, 3 to 5 picks from 12 candidates) the result has the
  largest determinant of all subsets, found by checking each one, in
  all 24 cases, while the greedy picks alone miss it in 6; the result
  is never worse than the greedy one, no single swap improves it, and
  the returned information matrix belongs to the returned indices;
  for example 9 above, the exchange triple is the best of all 4495
  triples and the greedy triple is not (added in 0.2.0).
- A `design` result gives the same error bars and covariance as
  `information` on the chosen settings (relative 1e-9), and
  `repeats_for` returns the same `r` for either (added in 0.2.0; it
  was refused in 0.1.1).
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
- For every alpha from 0.001 to 0.999 in steps of 0.001 and every n
  from 1 to 120, `coverage_exact` equals `k / (n + 1)` with `k`
  computed in exact fractions from the decimal alpha, and refuses
  exactly when that `k` exceeds n; for alpha 0.01, 0.05, 0.1, 0.2,
  0.3 and 0.45 the minimum number of scores named in the refusal is
  the exact one (added in 0.2.0).
- For n from 1 to 199 and six levels of alpha, `coverage_exact` lies
  between `1 - alpha` and `1 - alpha + 1/(n + 1)`.

**Records and files**

- An audit record is unchanged, compared with `==`, after
  `json.dumps` and `json.loads`; its digest is the fit's; changing one
  reading by 1e-9 changes the digest; `report_text` contains the
  expected lines and refuses a non-record.
- The model's units note (text or a dictionary) reaches the fit
  result and the audit record, survives the JSON round trip and
  appears in `report_text`; a record without it (as written by 0.1.x)
  still renders (added in 0.2.0).
- Saving and loading a CSV file with 3 setting columns and sigmas
  gives back identical arrays (`np.array_equal`); a file with 1
  setting column and no sigmas loads back with the right shape and
  `sigmas` `None`; a wrong header is refused. Awkward values (1/3,
  1e-300, -2.5e17, pi) come back identical, and an empty file, a file
  with no data rows, a row of the wrong length, a non-number, a
  negative sigma and a non-finite reading are each refused.
- The version in the package matches `pyproject.toml` and
  `CITATION.cff`, and every name in `__all__` exists.

**Added in 0.3.0**

- A straight line fitted without `sigmas` to readings scaled by 1,
  1e-3, 1e-6, 1e-9, 1e-12 and 1e-15 gives the values of
  `np.linalg.lstsq` to 1e-6 of their standard errors and the error
  bars of `s^2 (X^T X)^-1` to relative 1e-6, at every scale (0.2.0
  was 1.3 standard errors off at 1e-12; this test fails on 0.2.0).
  Noiseless lines and exponential decays at scales 1, 1e-6 and 1e-12
  still converge to the truth (relative 1e-12 and 1e-9). A weighted
  fit of the same small readings equals weighted least squares from
  `np.linalg.lstsq`.
- `propagate` accepts a fitted covariance and gives the closed form
  `a C a^T` (relative 1e-6), but refuses the same covariance copied to
  3 significant digits (it has a negative eigenvalue; 0.2.0 returned
  an error bar of 0) and a covariance with a correlation of 2. It
  accepts a parameter held fixed (zero variance and covariances) and
  a planned covariance with condition number 4e9, just inside the
  limit.
- `goodness_of_fit`: for every number of degrees of freedom from 1 to
  80 and twelve values of chi2 from 1e-8 to 1200, `p_value` equals
  the closed form computed in the test with the standard library (the
  finite Poisson sum for even degrees of freedom; `erfc` plus the
  recurrence `Q(a + 1, x) = Q(a, x) + x^a e^-x / Gamma(a + 1)` for odd
  ones) to relative 1e-12, and `p_value + p_value_low = 1`. Small
  lower tails match `-expm1(-chi2/2)` (2 degrees of freedom) and
  `erf(sqrt(chi2/2))` (1 degree of freedom) to relative 1e-13.
  `p_value_low` equals Simpson's-rule integration of the chi-squared
  density to 1e-10 for 20 cases, and `p_value` matches the fraction
  of 20000 seeded chi-squared draws above three quantiles within 4.5
  binomial standard errors. For 400 seeded straight-line fits with
  correct Gaussian errors the p-values are uniform: each tenth of
  (0, 1) holds 40 of them within 4.5 binomial standard deviations,
  and their mean is 1/2 within 4.5 standard errors.
- `load_columns` reads a commented, comma-separated export with a
  quoted text column and a blank line back to the numbers typed into
  it (`np.array_equal`), and agrees exactly with `np.genfromtxt` on
  the same file; tab, semicolon and space separated tables agree with
  the written numbers and with `np.genfromtxt`; a semicolon file with
  decimal commas and a byte-order mark is read correctly; a file
  written by `save_measurements_csv` reads back identically; a
  preamble is skipped with `skip_rows`, and a file without a header
  is read by column positions. Each refusal in the list above fires
  with the expected message.

**Added in 0.4.0**

- A decay of 10 on a background of 1000, with its time constant
  `tau = 1e-9` written in seconds (samples from `t = 0`): the slopes
  match the exact derivatives to 1e-5 of each column's largest value,
  the planned error bars match the exact `(J^T W J)^-1` to relative
  1e-5, and a fit from that guess equals the fit with exact slopes to
  1e-4 of an error bar (0.3.0 refused all three; this test fails on
  0.3.0). Where the fallback step already gave finite values the
  slopes are unchanged bit for bit. A `rel_step` of 0, a negative,
  NaN, infinite or non-numeric one, and a fractional `n` in
  `coverage_exact` are refused.
- `design` (greedy, and with `exchange=True`) returns the same
  indices and the same information matrix, compared with `==`, as the
  0.3.0 loops, kept in the test as the reference, on 21 problems (a
  decay, a Gaussian peak with a sloped background and a plane in two
  settings, with 2 to 8 picks from 9 to 60 candidates, including
  exact ties from repeated candidates), also when the batches are cut
  to 7 candidates.
- `inverse_predict` on a straight line equals the NIST/SEMATECH
  e-Handbook formula (section 2.3.6.7.3), typed into the test, for
  the value (relative 1e-12) and the error bar (relative 1e-7), and
  the covariance of three readings equals the closed form. On an
  exponential decay with background it equals the closed-form inverse
  `-tau ln((y - B) / A)` and the error bar from its partial
  derivatives (relative 1e-6). For 20000 seeded replicate
  calibrations (least squares done in the test with NumPy) followed
  by a new reading, the scatter of the inverse values matches the
  planned error bar within 3 %. On a saturating response the
  first-order error bar matches seeded simulation within 2 % where it
  is accepted; where it is refused (the slope changes by 65 % within
  one error bar) the simulated scatter is more than 25 % larger. Each
  refusal fires.
- `runs_test`: for every n1 and n2 from 1 to 7 the tail probabilities,
  the mean and the variance of the number of runs equal the counts
  over all arrangements of the signs (relative 1e-12); for n1, n2 up
  to 900 and 700 the tails equal exact integer arithmetic (relative
  1e-9). On 4000 seeded sets of random signs and of residuals of
  correct straight-line fits the false-alarm rate at the 5 % level is
  at most 5 % plus 4.5 binomial standard errors; on curved readings
  fitted with a line and errors stated 3 times too large, chi-squared
  gives p = 0.994 while the runs test gives p = 0.001.
- The two tutorials run and print exactly the output shown in their
  docstrings and in this README; their results agree with the values
  used to simulate the data within 3 error bars, and their printed
  numbers agree with each other (error bar parts adding in
  quadrature, the repeat count, the error bar of `1/tau`).

Every refusal listed in "When it refuses, and why" is checked to
fire, including a fit that does not converge within `max_iter` steps
(a `RuntimeError`) and a parameter that no reading responds to (an
all-zero slope column: `information` reports it not identifiable,
`fit` and `design` refuse). A `Model` whose `units` note is a
dictionary can be hashed (used as a dictionary key or in a set), and
equal models hash equally.

## Corrections in earlier versions

**0.4.0 fixed one wrong refusal.**

- *A small parameter whose effect is small next to the largest
  reading could be refused as "non-finite".* For a parameter whose
  relative step is too small to show next to the largest prediction,
  the slopes fall back to the 0.1.0 step `1e-6 * max(|theta|, 1e-3)`,
  which is 1e-9 for any parameter below 1e-3. For a time constant of
  1e-9 s that step is the parameter itself: `tau - h = 0`. A decay of
  10 on a background of 1000 (any amplitude below about 1/73 of the
  background) with `tau = 1e-9` and samples from `t = 0` then divided
  0 by 0, and `information`, `design` and `fit` stopped with "the
  forward function returned non-finite predictions; check theta and
  x", although the model is finite at that `tau`. 0.4.0 divides such
  a step by 10 until the model is finite and then checks it as
  before: the planned error bar on `tau` is 1.4189183e-10 s, equal to
  the exact value from the analytic slopes to 6e-9 relative. Results
  that 0.3.0 could compute are unchanged.

Also in 0.4.0: `Model.numeric_jacobian(rel_step=0)` (or a negative
step) failed with an `UnboundLocalError` and now explains the
problem; `coverage_exact(29.9, 0.1)` returned the answer for n = 29
and now refuses.

**0.3.0 fixed two wrong answers.**

- *A fit without `sigmas` could stop too early when the readings are
  small numbers in their own units.* Without `sigmas` the quantity
  being minimized is the sum of squared residuals, in the readings'
  units squared. The search stopped once a step changed it by less
  than `tol * (1 + sum)`, and that fixed 1 is enormous for readings
  of order 1e-12 (a current of a few picoamperes written in
  amperes): every change was below it, and the fit stopped after its
  first, still damped, step. For a straight line through 12 such
  readings (gain 2e-12 A/V, noise 5e-14 A) 0.2.0 returned gain
  1.99116e-12 and offset 1.35943e-13 after 1 step, 1.30 and 1.15
  standard errors away from the least-squares answer (2.00051e-12
  and 1.10355e-13), with error bars about 10 % too large (7.18e-15
  instead of 6.54e-15 on the gain). With readings of order 1e-9 and
  relative noise 1e-5 it stopped 0.60 standard errors off. 0.3.0
  measures the change in units of the residual variance (at most 1)
  and gives the least-squares answer to 1e-6 of a standard error at
  every scale tested. Fits with `sigmas` are unchanged bit for bit;
  fits without them whose residual variance is below 1 may take a
  step or two more and then change far below their error bars (in
  300 random straight-line fits with readings from 1e-3 to 1e3, 253
  gave identical results and the largest change was 3.4e-6 of a
  standard error, the size of the rounding differences between
  `fit` and `np.linalg.lstsq` themselves). If you used 0.2.0
  or earlier without `sigmas` on readings much smaller than 1, please
  re-run those fits.
- *`propagate` returned an error bar of 0 for an invalid covariance.*
  A covariance that is not positive semidefinite gives some
  combination of parameters a negative variance. That happens, for
  example, when the covariance of a strongly correlated fit is copied
  to a few digits: a line measured at settings 100 to 101 has slope
  and offset correlated at -0.999995, and its covariance rounded to
  3 significant digits has a negative eigenvalue. With it, the error
  bar of the predicted reading at 100.5 came out as 0.0 (the right
  value is 0.0144), with a negative variance in the returned `cov`.
  Such a matrix is now refused with an explanation.

**0.2.0 fixed three wrong answers.**

- *Slopes for a parameter at zero.* When a parameter was exactly zero
  (or too small for its relative step to show), 0.1.1 used a fixed
  step of 1e-9 in the parameter's own units, which can be far too
  large. For a Gaussian pulse written in seconds, with center 0 and
  width 1 ns, that step equals the width: the slope with respect to
  the center was 31 % wrong, and both the planned and the fitted
  error bar on the center came out as 6.687e-12 s instead of the
  exact 5.312e-12 s (26 % too large). The step is now checked
  against a 10 times smaller one and reduced while they disagree;
  0.2.0 gives 5.312e-12 s. Where the fixed step was already accurate
  the slopes are unchanged, bit for bit. If you used 0.1.x with a
  parameter at or near zero whose natural size is far from 1 in its
  units, please re-run those results.
- *`repeats_for` refused the output of `design`* with the message
  "the plan is not identifiable", because `design` did not report
  `identifiable`. It now does (and also returns `cov`).
- *Conformal rank for some alpha.* In binary floating point
  `1 - 0.7` is slightly more than 0.3, so for alpha = 0.7 and n = 9
  the rank `ceil(10 * 0.3)` came out as 4 instead of 3:
  `conformal_quantile` returned the 4th smallest score and
  `coverage_exact(9, 0.7)` returned 0.4 instead of 0.3. In a scan of
  all alphas 0.001 to 0.999 (steps of 0.001) and n up to 3000 this
  happened for 255 of the 999 alphas (3595 alpha/n pairs), none of
  them 0.01, 0.02, 0.05, 0.1, 0.2 or 0.25. In every case the rank was
  one too large, so the band was slightly wider than needed, never
  narrower.

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
- Unless you give `jac`, slopes are computed by finite differences,
  not exact derivatives. For a parameter at zero the step is found by
  shrinking a 1e-9 step until two successive steps agree; this
  assumes the model is smooth in that parameter. It checks only
  shrinking, so a 1e-9 step that is too small for a parameter whose
  natural size is much larger than 1 (a zero offset written in
  nanoseconds when the times are seconds) is not enlarged; write such
  a parameter in units where its natural size is near 1, or give
  `jac`.
- `labplan` cannot check a hand-written `jac`; a wrong one gives wrong
  error bars. Compare it once with `Model.numeric_jacobian`, as in
  example 10.
- `propagate` is first order: exact for straight-line functions of
  the parameters, an approximation otherwise that is good when the
  function is close to a straight line within about one error bar.
  It refuses a covariance that is not positive semidefinite, but it
  cannot tell when a valid one was rounded: copy covariances at full
  precision.
- `goodness_of_fit` assumes independent Gaussian reading errors of the
  stated size. Its chi-squared distribution is exact for models that
  are straight-line functions of their parameters and approximate
  otherwise. A small p-value says that the model or the stated errors
  are wrong, not which one, and a large one does not prove the model
  right: with few readings or large errors the test has little power.
- `design` is a greedy heuristic: it adds the best single setting at
  each step and does not prove that the chosen set is the best
  possible. `exchange=True` improves it until no single swap helps,
  which in the tests reached the best subset every time, but that is
  still not a proof. Each swap pass tries every chosen setting
  against every unused one; since 0.4.0 these are scored in batches
  (3000 candidates take a fraction of a second), but the work still
  grows with the number of picks times the number of candidates for
  every pass. It picks each candidate at most once; use `repeats_for`
  for repeats.
- `inverse_predict` needs settings that are one number per
  measurement and a bracket in which the model reaches the reading
  exactly once. It scans the bracket at `n_scan` points, so two
  crossings closer together than the scan spacing are not seen. Its
  error bar is first order, like `propagate`; it refuses when the
  slope changes by more than 10 % within one error bar, and below that
  the error bar can still be a few percent too small where the curve
  bends (in the tests the simulated scatter was 1.014 times the error
  bar at a slope change of 10.5 %).
- `runs_test` looks only at the signs, so it needs enough residuals
  to have power (with 10 residuals, 5 of each sign, the smallest
  possible `p_value` is 2/252 = 0.008). For
  residuals of a fit it is approximate (the residuals are not exactly
  independent); in the tests that made it more cautious, not less.
- The conformal guarantee is an average over calibration sets and new
  measurements, not a promise at each setting, and it needs the
  calibration data to be exchangeable with the new measurement:
  calibration data from last month's instrument state do not certify
  next month's drift. The tighter bound `1 - alpha + 1/(n + 1)`
  assumes scores without ties. `alpha` is read to about 9
  significant digits: an alpha that differs from a "round" value by
  less than 1e-9 relative is treated as that value.
- `save_measurements_csv` and `load_measurements_csv` use `labplan`'s
  own CSV layout (`x1, ..., y, sigma`). Other delimited text tables
  are read with `load_columns` (since 0.3.0), which needs one header
  line (or none), one row per line (no line breaks inside quoted
  fields) and names given exactly as in the header. It does not read
  binary instrument formats or spreadsheet files (`.xlsx`); export
  those as text first.
- No physics ships with the package. Your model and its `reference`
  carry the physics.

## Where it comes from

Nine research packages in this organization -- covering color-center
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
Design of Experiments, SIAM (2006), and the swap refinement the
exchange idea of V. V. Fedorov, Theory of Optimal Experiments,
Academic Press (1972). The chi-squared test computes the regularized
incomplete gamma functions (NIST Digital Library of Mathematical
Functions, chapter 8) by the series and continued fraction of W. H.
Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery,
Numerical Recipes, 3rd ed., Cambridge University Press (2007),
section 6.2; the Birge ratio is from R. T. Birge, Phys. Rev. 40, 207
(1932). The runs test is A. Wald and J. Wolfowitz, "On a test whether
two samples are from the same population", Ann. Math. Statist. 11,
147-162 (1940), doi:10.1214/aoms/1177731909. The straight-line case of
`inverse_predict` is the propagation-of-error formula of the
NIST/SEMATECH e-Handbook of Statistical Methods, section 2.3.6.7.3
(https://itl.nist.gov/div898/handbook/mpc/section3/mpc3673.htm).
First-order propagation of uncertainty with
covariances is the "law of propagation of uncertainty" of JCGM
100:2008, Evaluation of measurement data -- Guide to the expression
of uncertainty in measurement (sections 5.1.2 and 5.2). Split
conformal prediction follows Vovk, Gammerman and Shafer, Algorithmic
Learning in a Random World, Springer (2005); Lei et al., J. Am. Stat.
Assoc. 113, 1094 (2018); and Angelopoulos and Bates,
arXiv:2107.07511.

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

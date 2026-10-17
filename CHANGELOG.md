# Changelog

Every statistical claim added in any release is pinned by a test
against a closed form, an exact identity, or seeded simulation
against an exact formula; the release notes on GitHub carry the full
anchor lists.

## 0.4.0 (2026-10-17)

### Added

- `inverse_predict(model, result, reading, bracket, reading_sigma=0.0,
  theta=None, cov=None, n_scan=1001, max_slope_change=0.1)`: reads a
  calibration curve backwards. It finds the setting `x0` at which the
  model predicts a new reading inside the given bracket (a scan of
  `n_scan` points for sign changes, then bisection) and its error bar
  by first-order propagation through the implicit function,
  `u(x0)^2 = (u(y0)^2 + a^T C a) / s^2` (JCGM 100:2008, section 5.2),
  split into the part from the reading and the part from the
  calibration. Several readings give the full covariance (they share
  the calibration). For a straight line it is the propagation-of-error
  formula of the NIST/SEMATECH e-Handbook of Statistical Methods,
  section 2.3.6.7.3
  (https://itl.nist.gov/div898/handbook/mpc/section3/mpc3673.htm).
  Refuses a reading the model does not reach in the bracket or reaches
  more than once, a model that is flat at the solution, and a
  solution where the slope changes by more than `max_slope_change`
  (10 %) within one error bar, where a first-order error bar is not
  reliable.
- `runs_test(residuals, order=None)`: the runs test of A. Wald and
  J. Wolfowitz, Ann. Math. Statist. 11, 147-162 (1940),
  doi:10.1214/aoms/1177731909, on the signs of the residuals ordered
  by the setting. Finds a systematic misfit (a bend, a drift) that
  chi-squared misses when the reading errors are overstated. Exact
  tail probabilities of the number of runs, `p_value` (too few runs)
  and `p_value_high` (too many), with the mean and standard deviation
  of the number of runs.
- Tutorials `examples/tutorial_calibration_curve.py` and
  `examples/tutorial_decay_on_background.py` (see Tutorials below).
- README examples 13 (`inverse_predict`) and 14 (`runs_test`), and a
  Tutorials section, each with output obtained by running it.

### Fixed

- `Model.numeric_jacobian` (and so `information`, `design`, `fit` and
  `propagate`) refused a valid model at a valid parameter as
  "non-finite". A parameter whose relative step does not change the
  predictions by 1e-8 of their largest value falls back to the 0.1.0
  step `1e-6 * max(|theta|, 1e-3)`, which is 1e-9 for a parameter
  below 1e-3: for a time constant of 1e-9 s it is the parameter
  itself. A decay of amplitude 10 on a background of 1000 (any
  amplitude below about 1/73 of the background), `tau = 1e-9` s,
  samples from t = 0: `tau - h = 0`, the model divided 0 by 0, and
  before the fix `information`, `design` and `fit` from that guess
  raised "the forward function returned non-finite predictions".
  After: the step is divided by 10 until the model is finite (at most
  8 times) and then checked as before; the planned error bar on `tau`
  (reading error 1) is 1.4189183e-10 s, the exact value from the
  analytic slopes to 6e-9 relative, and the fit equals the fit with
  exact slopes to 1e-4 of an error bar. Steps that already gave finite
  predictions are used as before, so every result 0.3.0 could compute
  is unchanged (test `test_fallback_step_that_reaches_zero_is_cut`,
  which fails on 0.3.0, and
  `test_fallback_cut_does_not_touch_working_steps`).

### Changed

- Clearer refusals: `Model.numeric_jacobian` (which `Model.jacobian`
  uses when the model has no `jac`) refuses a `rel_step` that is not a finite positive number (0.3.0
  raised `UnboundLocalError` for 0 or a negative step);
  `coverage_exact` refuses an `n` that is not a whole number (0.3.0
  returned 0.9 for `coverage_exact(29.9, 0.1)`, the answer for
  n = 29).
- Packaging (from the unreleased notes after 0.3.0): the license is
  given as the SPDX expression `license = "Apache-2.0"` with
  `license-files = ["LICENSE"]`; the old license table and license
  classifier are gone. Building no longer prints setuptools'
  deprecation warnings (their support ends on 2027-02-18). Building
  from source needs setuptools 77 or newer; installing the package is
  unchanged.
- README: the new functions, refusals, tests, the 0.4.0 correction,
  limits and sources; a word guide entry for residuals and runs and
  one for inverse prediction. The printed output of README examples 1
  to 12 and of `examples/01_plan_then_fit.py` and
  `examples/02_conformal_certificate.py` is unchanged (apart from the
  version and the timestamp in the record that example 01 prints).
- Version 0.4.0 in `pyproject.toml`, `labplan.__version__` and
  `CITATION.cff` (released 2026-10-17).

### Performance

- `design` scores all candidates of a greedy step, and all swaps of
  one chosen setting in `exchange=True`, with one batched call of
  `np.linalg.slogdet` (in batches of at most 8 MB) instead of one
  call per candidate. Each matrix is formed by the same floating-point
  operations and factorized by the same LAPACK routine, and the scan
  for the best one keeps the 0.3.0 order and thresholds, so picks and
  information matrices are identical bit for bit (test
  `test_batched_design_equals_the_030_loops`, which keeps the 0.3.0
  loops as the reference, on 21 problems including exact ties).
  Measured speedup, median of 7 interleaved runs of each version on a
  shared 2-CPU machine: decay model, 400 candidates, 10 picks: 12.6x
  (greedy) and 17.1x (exchange); Gaussian peak with sloped background,
  1000 candidates, 12 picks: 13.0x and 16.7x; decay model, 3000
  candidates, 6 picks: 19.1x and 14.9x (a second run on the loaded
  machine gave 8.7x to 24.1x for the same six cases). For example,
  the 1000-candidate exchange design took 1.56 s in 0.3.0 and 0.094 s
  in 0.4.0.

### Tutorials

- `examples/tutorial_calibration_curve.py`: plans, calibrates and
  checks an illustrative quadratic pressure-sensor curve (11 reference
  pressures), then reads an unknown (5 readings) off the curve with
  `inverse_predict`, its error bar split into reading noise and
  calibration. Runs in about 0.2 s.
- `examples/tutorial_decay_on_background.py`: a 10 mV decay with a
  time constant near 1 ns on a 1 V level, in seconds and volts (the
  case 0.3.0 refused): planning the number of averaged traces, fit,
  chi-squared and runs checks, and the decay rate `1/tau` with
  `propagate`. Runs in about 0.2 s.

### Tests

- 27 new tests (96 in total, about 6 s; the new ones about 3 s):
  `tests/test_fixes_040.py` (the fix against exact slopes and the
  exact information matrix, the clearer refusals, and `design` against
  the 0.3.0 loops), `tests/test_inverse_runs.py` (`inverse_predict`
  against the NIST handbook formula, the closed-form inverse of an
  exponential decay, the closed-form covariance of several readings,
  20000 seeded replicate calibrations and simulation of the
  slope-change refusal; `runs_test` against enumeration of all sign
  arrangements, exact integer arithmetic, seeded false-alarm rates
  and a bend that chi-squared misses) and `tests/test_tutorials.py`
  (both tutorials print exactly the output in their docstrings and the
  README, and their numbers agree with the simulated truth and with
  each other).
- The suite was also run with Python 3.9, NumPy 1.22.0 and pytest
  7.0.0 (the CI oldest-dependencies job): 96 passed.

## 0.3.0 (2026-10-10)

### Added

- `goodness_of_fit(result)` or `goodness_of_fit(chi2=..., dof=...)`:
  the chi-squared test of a fit made with `sigmas`. Returns `chi2`,
  `dof`, `reduced_chi2`, `p_value` (probability of a chi2 at least
  this large if the model and the stated errors are right, the
  regularized upper incomplete gamma function `Q(dof/2, chi2/2)`),
  `p_value_low` (`P(dof/2, chi2/2)`, the opposite tail) and
  `birge_ratio` (`sqrt(chi2/dof)`; R. T. Birge, Phys. Rev. 40, 207
  (1932)). The incomplete gamma functions are computed in pure Python
  by the series and continued fraction of Press, Teukolsky,
  Vetterling and Flannery, Numerical Recipes, 3rd ed., Cambridge
  University Press (2007), section 6.2, as
  `labplan.stats.gamma_regularized`; small tail probabilities keep
  their full relative precision. Refuses a fit without `sigmas`, zero
  degrees of freedom, and a negative or non-finite chi2.
- `load_columns(path, x, y, sigma=None, ...)`: reads the named
  columns (header names, or positions) of a delimited text table such
  as an instrument or spreadsheet export: comma, tab, semicolon or
  space separated (detected from the header, or `delimiter=`),
  decimal point or decimal comma, `#` comment lines, blank lines, a
  preamble (`skip_rows=`), files with or without a header row, and
  the byte-order mark some spreadsheet programs write. Returns
  `(x, y, sigmas)` for `fit`. Refuses, naming the line and column,
  every cell it would otherwise have to guess at: empty cells, text,
  NaN and infinity, rows of the wrong length, ambiguous delimiters,
  unknown or repeated column names, and reading errors that are not
  positive. This lifts the listed limit that files with other column
  names had to be converted first.
- README examples 11 (goodness of fit) and 12 (reading an exported
  table), each with output obtained by running it.

### Fixed

- `fit` without `sigmas` stopped too early when the readings are
  small numbers in their own units. The stopping test compared the
  change of the sum of squared residuals, which has the readings'
  units squared, with `tol * (1 + sum)`; for readings of order 1e-12
  every change was below it and the fit returned its first, still
  damped, Levenberg-Marquardt step. Straight line through 12
  readings of order 1e-12 (gain 2e-12, offset 1e-13, noise 5e-14):
  before, gain 1.99116e-12 and offset 1.35943e-13 after 1 step,
  1.30 and 1.15 standard errors from least squares, with the gain
  error bar 7.18e-15; after, gain 2.00051e-12 and offset
  1.10355e-13, equal to `np.linalg.lstsq` within 1e-6 of a standard
  error, gain error bar 6.54e-15. Readings of order 1e-9 with relative
  noise 1e-5: before, 0.60 standard errors off; after, within 1e-6.
  The unit in the test is now the residual variance `cost / (n - p)`,
  capped at 1, so no fit stops earlier than before. Fits with
  `sigmas` are unchanged bit for bit (test
  `test_unweighted_fit_does_not_depend_on_reading_units`, and
  `test_unweighted_fit_of_small_readings_with_tiny_noise`).
- `propagate` accepted a covariance that is not positive
  semidefinite and returned an error bar of 0 with a negative
  variance in `cov`. Example: the covariance of a straight-line fit
  at settings 100 to 101 (slope and offset correlated at -0.999995)
  copied to 3 significant digits; the error bar of the prediction at
  100.5 was 0.0 before (0.0144 with the full covariance), and a
  covariance with a correlation of 2 gave 0.0 for the difference of
  the two parameters. Such a matrix is now refused; the test is made
  on the eigenvalues of the correlation matrix (tolerance -1e-10), so
  it does not depend on the units of the parameters, and it accepts
  every covariance labplan produces, including a planned one with
  condition number 4e9 (test
  `test_propagate_refuses_a_covariance_that_is_not_positive`).

### Changed

- Fits without `sigmas` whose residual variance is below 1 may take
  a step or two more than in 0.2.0. In 300 random straight-line fits
  with readings from 1e-3 to 1e3, 253 gave identical results and the
  largest change was 3.4e-6 of a standard error. The printed output of
  every README example and of `examples/` is unchanged.
- Clearer refusals: `design` refuses an `n_pick` that is not a whole
  number (0.2.0 silently cut 2.5 to 2); `repeats_for({})` says the
  dictionary of targets is empty (0.2.0 raised the bare error of
  Python's `max()` on an empty sequence); `fit` refuses `max_iter` below 1 (0.2.0 raised
  "did not converge") and a negative or non-finite `tol`.
- Type hints on the public functions and `Model` methods.
- README: American spelling in the prose that was touched ("center",
  "color"); a p-value entry in the word guide; the new functions,
  refusals, tests, corrections, limits and sources.
- Version 0.3.0 in `pyproject.toml`, `labplan.__version__` and
  `CITATION.cff` (released 2026-10-10).

### Tests

- 26 new tests (69 in total, about 4 s): `tests/test_fixes_030.py`
  (the two fixes, against `np.linalg.lstsq` and the closed form
  `a C a^T`, and the clearer refusals) and `tests/test_lab_data.py`
  (`goodness_of_fit` against closed forms of the chi-squared tail,
  numerical integration of its density, seeded chi-squared draws and
  the uniform distribution of p-values of correct fits; `load_columns`
  against the typed numbers and `np.genfromtxt`, and every refusal).
  The fix tests fail on 0.2.0.

## v0.2.0 - 2026-09-30

### Added

- `propagate(func, theta, cov)`: value, 1-sigma error bar and
  covariance of any quantity computed from the parameters (a
  predicted reading, a ratio, a time constant), by first-order
  propagation `J C J^T` (JCGM 100:2008, sections 5.1.2 and 5.2).
  Works with a fit's covariance or with a plan's, for expected error
  bars before measuring. Exact for quantities linear in the
  parameters; first-order approximation otherwise. A `func` that
  returns an array with more than one dimension is refused.
- `Model(..., jac=...)`: optional exact slopes `jac(theta, x) -> (n,
  p)`, used by `information`, `design` and `fit` instead of finite
  differences; shape and finiteness are checked.
  `Model.numeric_jacobian` always returns the finite-difference
  slopes, so a hand-written `jac` can be compared with them.
- `design(..., exchange=True)`: after the greedy picks, swaps one
  chosen candidate for one unused candidate while that increases the
  information determinant (Fedorov's exchange idea, V. V. Fedorov,
  Theory of Optimal Experiments, Academic Press (1972)). The default
  (`exchange=False`) gives the same picks as before.
- The model's `units` note (text or a dictionary) is now carried into
  `FitResult.units`, the audit record (`model_units`) and
  `report_text` (a `units:` line). This lifts a listed limit.
- `design` now also returns `identifiable` and `cov`.

### Fixed

- Finite-difference slopes for a parameter that is exactly zero (or
  too small for its relative step to show) used a fixed step of 1e-9
  in the parameter's units, which can be as large as the parameter's
  natural scale. That step is now compared with a 10 times smaller
  one and reduced while the two disagree by more than 1e-6 of the
  column's largest slope plus a rounding allowance (at most 8 times;
  the estimate with the smallest disagreement is kept if rounding or
  noise in the model takes over). Where the old step was accurate the
  slopes are bit-for-bit unchanged.
- `repeats_for(target, design(...))` was refused with "the plan is
  not identifiable" because `design` did not report `identifiable`.
- The conformal rank `k = ceil((n + 1)(1 - alpha))` was computed in
  floating point, where `1 - 0.7` is slightly above 0.3; it came out
  one too large for 255 of the alphas 0.001, 0.002, ..., 0.999 at some
  n up to 3000 (3595 alpha/n pairs; none at alpha 0.01, 0.02, 0.05,
  0.1, 0.2 or 0.25). A product within 1e-9 (relative) above an
  integer is now taken to be that integer, which matches exact
  rational arithmetic for all 999 alphas and every n up to 3000.
- A `sigmas` array of the wrong length now raises a `ValueError` that
  says a single value or one per reading is needed, instead of
  NumPy's broadcasting message.
- A `Model` with a dictionary `units` note (as in `examples/01`)
  raised `TypeError: unhashable type: 'dict'` when hashed, because
  `Model` is a frozen dataclass. `units` is now left out of the hash
  (it still counts for `==`).

### Behaviour changes

- Gaussian pulse in seconds (centre 0, width 1 ns, 25 samples from -3
  to 3 ns, reading error 0.01): planned and fitted error bar on the
  centre 6.686976e-12 s in 0.1.1, 5.311778e-12 s in 0.2.0 (exact
  value from the analytic slopes: 5.311777e-12 s).
- `conformal_quantile(scores, 0.7)` with 9 scores: 4th smallest score
  in 0.1.1, 3rd smallest in 0.2.0; `coverage_exact(9, 0.7)`: 0.4 in
  0.1.1, 0.3 in 0.2.0. Results for alpha 0.01, 0.02, 0.05, 0.1, 0.2
  and 0.25 do not change.
- `audit_record` has one more key, `model_units` (empty string when
  the model has no units note); `report_text` prints a `units:` line
  when it is not empty.
- `design` returns two more keys (`identifiable`, `cov`).
- `sigmas` given as any array with exactly one element (for example
  shape `(1, 1)`) is now read as one value for all readings; in 0.1.1
  only shapes that NumPy could broadcast were accepted.

### Tests

- 27 new tests (43 in total, 2 to 4 s). New files
  `tests/test_derivatives_design.py` (zero-parameter slopes against
  analytic derivatives and against the same model in nanoseconds;
  bit-identical slopes where the old step was accurate; exact `jac`;
  exchange against exhaustive search over all subsets, including
  README example 9; `design`
  output in `repeats_for`), `tests/test_propagate_records.py`
  (`propagate` against closed forms, the first-order product formula
  and seeded simulation; units in records; the conformal rank against
  Python `fractions` for 999 alphas and n up to 120) and
  `tests/test_documented_claims.py` (claims the README already made
  without a test: the covariance of a fit without `sigmas` against
  `np.linalg.lstsq` and `s^2 (X^T X)^-1`, per-unit planning, the
  closed form of `repeats_for`, slopes of a nonlinear model against
  exact derivatives, coverage bounds over many n and alpha, every
  documented refusal including non-convergence and a parameter no
  reading responds to, and hashing a model with dictionary units).
- The suite was also run with Python 3.9, NumPy 1.22.0 and pytest
  7.0.0 (the CI oldest-dependencies job): 43 passed.

### Changed

- README: new examples 8 to 10 (propagation, exchange design, exact
  slopes), each with output obtained by running it; example 5 output
  shows the new `model_units` key; "Corrections", "Limits" and the
  test descriptions updated.

## v0.1.1 - 2026-09-22

### Fixed

- `Model.jacobian` (used by `information`, `design` and `fit`) took
  its finite-difference step as `1e-6 * max(|theta|, 1e-3)`. For a
  parameter smaller than 1e-3 in its own units the step, 1e-9, was
  too coarse; for a time constant of 1e-9 s it was the size of the
  parameter itself. The planned error bars then depended on the
  units the parameter was written in, a noiseless fit could stop
  away from the truth (V0 = 0.987 and tau = 0.938 ns instead of 1
  and 1 ns in the new test's RC model), and a model that divides by
  such a parameter was refused as "non-finite". The step is now
  `1e-6 * |theta|`. When that step changes the predictions by less
  than 1e-8 of their size (always for a parameter that is exactly
  zero, and for one that is tiny next to the rest of the prediction,
  such as a slope of 1e-12 beside an offset of 1), the 0.1.0 step
  `1e-6 * max(|theta|, 1e-3)` is used instead: a purely relative step
  would there be lost in floating-point rounding and give wrong
  slopes, wrong error bars and false "cannot tell the parameters
  apart" refusals. Results for parameters of size 1e-3 or more are
  unchanged.

### Tests

- New `test_small_parameters_unit_invariant`: the same RC model with
  tau in seconds and in nanoseconds gives the same planned error bars
  (to 1 part in 10^6) and the same noiseless fit, and the slopes of a
  model that divides by a 1e-9 parameter are finite and exact to
  1 part in 10^6. It fails on 0.1.0.
- New `test_near_zero_parameter_keeps_a_resolvable_step`: for a
  straight line with slope 1e-12, 1e-300 or 0, the slopes match the
  exact ones to 1 part in 10^6, and a noiseless fit of a flat line
  returns the textbook covariance. It guards the fallback above.
- The CI matrix now includes Python 3.10, which the classifiers
  claim but CI did not run, and a new `oldest-dependencies` job runs
  the suite on Python 3.9 with NumPy 1.22.0 and pytest 7.0.0, the
  lowest versions `pyproject.toml` allows.

### Changed

- README rewritten in plain language, with worked examples whose
  printed output is checked, a list of the actual refusals, and each
  test described with the tolerance it really uses.

### Corrections to the v0.1.0 notes

- "the recomputed greedy rule": no test recomputes the greedy
  selection step by step. The tests check that the picks are
  distinct, that they beat 30 random subsets, and that the two-point
  pick equals the best pair found by exhaustion.
- "linear-model covariance equals the textbook closed form exactly,
  planner and fit agree as one matrix": the tests assert this to a
  relative tolerance of 1e-6, not exactly.

## v0.1.0 - 2026-09-18

First release: the measurement-planning, calibration and audit-trail
loop that nine physics packages of this organization each grew for
their own instrument, extracted and generalized to ANY forward model
a lab can write as a Python function.

- `model`: the `Model` container -- a named forward function with
  named parameters and a mandatory `reference`.
- `plan`: `information` (predicted error bars from the design alone
  -- the same (J^T W J)^-1 matrix the fit reports), `design` (greedy
  D-optimal selection; Pukelsheim, SIAM (2006)), `repeats_for` (the
  exact 1/sqrt(r) law inverted in closed form).
- `fit`: pure-NumPy Levenberg-Marquardt weighted least squares with
  exact known-noise covariance, chi-squared checks, and
  scale-invariant identifiability refusals.
- `conformal`: split conformal prediction -- distribution-free
  intervals with the exact finite-sample guarantee (Vovk et al.
  (2005); Lei et al., JASA 113, 1094 (2018); Angelopoulos & Bates,
  arXiv:2107.07511), with the exact coverage formula exposed.
- `report`: JSON-serializable audit records (values, error bars,
  goodness of fit, identifiability, sha256 of the exact data,
  software versions, UTC timestamp, operator) and their
  human-readable rendering.
- `records`: a checked CSV contract with bit-exact round trips.
- Anchors: linear-model covariance equals the textbook closed form
  exactly, planner and fit agree as one matrix; 400 seeded
  Monte-Carlo runs match reported error bars; exact-degeneracy
  refusals; the rank-one determinant identity and the recomputed
  greedy rule; the two-point line design matched to the classical
  optimum by exhaustion; exact conformal coverage
  ceil((n+1)(1-alpha))/(n+1) verified by seeded simulation inside
  the published two-sided bound; audit-record JSON round trip and
  data-digest pinning; bit-exact file round trips.

# Changelog

Every statistical claim added in any release is pinned by a test
against a closed form, an exact identity, or seeded simulation
against an exact formula; the release notes on GitHub carry the full
anchor lists.

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

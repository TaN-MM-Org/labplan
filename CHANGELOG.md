# Changelog

Every statistical claim added in any release is pinned by a test
against a closed form, an exact identity, or seeded simulation
against an exact formula; the release notes on GitHub carry the full
anchor lists.

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

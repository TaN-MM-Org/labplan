"""Error bounds that hold even when the model is wrong.

    pip install labplan
    python 02_conformal_certificate.py

A fit's covariance error bars trust the model. Split-conformal
intervals do not: calibrate them on held-out residuals and their
finite-sample coverage is guaranteed by exchangeability alone, no
matter how wrong the model is.  Here we deliberately fit a straight
line to gently curved data and show that the conformal band still
covers at its promised rate while staying honest about its width.
"""
import numpy as np

from labplan import (Model, coverage_exact,
                     conformal_interval, conformal_quantile, fit)

rng = np.random.default_rng(11)

# The world is curved; our model is a straight line (on purpose).
def world(x):
    return 1.0 + 2.0 * x + 0.8 * x ** 2 + 0.05 * rng.standard_normal(len(x))

model = Model(
    name="line-through-curved-world",
    f=lambda th, x: th[0] + th[1] * x[:, 0],
    param_names=("b", "m"),
    reference="straight-line model, deliberately misspecified for this demo",
)

# Fit on one batch...
x_fit = rng.uniform(0.0, 1.0, 60).reshape(-1, 1)
result = fit(model, x_fit, world(x_fit[:, 0]), [0.0, 1.0])

# ...calibrate the conformal band on a SECOND, held-out batch...
x_cal = rng.uniform(0.0, 1.0, 79).reshape(-1, 1)
scores = np.abs(world(x_cal[:, 0])
                - model.predict(result.theta, x_cal))
q = conformal_quantile(scores, alpha=0.1)
print(f"calibrated half-width q = {q:.3f}  "
      f"(exact coverage {coverage_exact(len(scores), 0.1):.4f})")

# ...and check the guarantee. The promise is MARGINAL: averaged over
# calibration draws, coverage is at least 90%. Any single calibration
# set scatters around that, so we repeat the whole calibrate-then-test
# loop and average -- which is exactly the statement of the theorem.
covs = []
for _ in range(300):
    xc = rng.uniform(0.0, 1.0, 79).reshape(-1, 1)
    s = np.abs(world(xc[:, 0]) - model.predict(result.theta, xc))
    qi = conformal_quantile(s, alpha=0.1)
    xn = rng.uniform(0.0, 1.0, 200).reshape(-1, 1)
    lo, hi = conformal_interval(model.predict(result.theta, xn), qi)
    yn = world(xn[:, 0])
    covs.append(np.mean((lo <= yn) & (yn <= hi)))
print(f"coverage over 300 calibrate-and-test rounds: "
      f"{np.mean(covs):.3f}  "
      f"(theory: exactly {coverage_exact(79, 0.1):.4f} for this n)")
print("the model is wrong -- the guarantee holds anyway; the price "
      "is a wider band than an honest model would need.")

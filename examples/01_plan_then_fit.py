"""Plan a calibration before taking it, then fit and write the record.

A worked end-to-end example, runnable as-is:

    pip install labplan
    python 01_plan_then_fit.py

The instrument here is a first-order low-pass filter whose gain at
frequency f is  G(f) = A / sqrt(1 + (f/fc)^2)  -- two unknowns, the
DC gain A and the corner frequency fc.  Everything below works the
same for YOUR model: replace `f` and the parameter names.
"""
import numpy as np

from labplan import (Model, audit_record, design, fit, information,
                     repeats_for, report_text)

# ---------------------------------------------------------------- model
model = Model(
    name="lowpass-gain",
    f=lambda th, x: th[0] / np.sqrt(1.0 + (x[:, 0] / th[1]) ** 2),
    param_names=("A", "fc"),
    reference=("first-order low-pass response, e.g. Horowitz & Hill, "
               "The Art of Electronics, 3rd ed. (2015), ch. 1"),
    units={"A": "V/V", "fc": "Hz"},
)

guess = [2.0, 1.0e3]          # rough prior values, good enough to plan
sigma = 0.02                  # each gain reading has a 2% abs. error

# ------------------------------------------------- plan the measurement
# Which 6 of these 25 candidate frequencies buy the most information?
candidates = np.geomspace(10.0, 1.0e5, 25).reshape(-1, 1)
picked = design(model, guess, candidates, n_pick=6, sigmas=sigma)
chosen = candidates[picked["indices"]]
print("measure at (Hz):", np.round(np.sort(chosen.ravel()), 1))

# How many repeats of that 6-point sweep until sigma(fc) <= 5 Hz?
plan = information(model, guess, chosen, sigmas=sigma)
r, predicted = repeats_for({"fc": 5.0}, plan)
print(f"repeats needed: {r}   "
      f"(predicted sigma_fc = {predicted['fc']:.2f} Hz)")

# ------------------------------------------- simulate taking the data
rng = np.random.default_rng(7)
x = np.tile(chosen, (r, 1))
truth = [2.05, 980.0]
y = model.predict(truth, x) + sigma * rng.standard_normal(len(x))

# ------------------------------------------------------------- fit it
result = fit(model, x, y, guess, sigmas=sigma)
for name in model.param_names:
    print(f"{name} = {result.values[name]:.4g} "
          f"+/- {result.sigma[name]:.2g} {model.units.get(name, '')}")

# ------------------------------------------------- audit-ready record
record = audit_record(result, operator="example-user",
                      note="worked example 01 of the labplan repo")
print()
print(report_text(record))

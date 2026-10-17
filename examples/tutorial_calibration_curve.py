"""Tutorial: calibrate a sensor, check the calibration, and read an
unknown off the calibration curve, with its error bar.

    pip install labplan
    python tutorial_calibration_curve.py

The sensor here is illustrative: its output voltage V depends on an
applied pressure P (in kPa) along a gently curved line,
V = c0 + c1 P + c2 P^2. The "measured" readings are simulated with a
fixed seed, so the output below is repeatable. Swap in your own model
and data; every step stays the same.

Steps:
1. Before calibrating: plan 11 reference pressures, and ask how well
   an unknown near 50 kPa would then be determined.
2. Calibrate: fit the curve to the readings of the 11 references.
3. Check the calibration: chi-squared (are the residuals the right
   size?) and the runs test (do they wander systematically?).
4. Measure an unknown 5 times and read its pressure off the curve,
   with an error bar that counts both the reading noise and the
   calibration's own uncertainty.

Output (labplan 0.4.0):

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
"""
import numpy as np

from labplan import (Model, fit, goodness_of_fit, information,
                     inverse_predict, runs_test)

sensor = Model(
    name="pressure sensor",
    f=lambda th, x: th[0] + th[1] * x[:, 0] + th[2] * x[:, 0] ** 2,
    param_names=("c0", "c1", "c2"),
    reference="illustrative quadratic calibration curve (second-degree "
              "polynomial in the applied pressure)",
    units={"c0": "V", "c1": "V/kPa", "c2": "V/kPa^2"},
)
reading_error = 0.0006            # V, one reading (from the datasheet)
pressures = np.linspace(0.0, 100.0, 11)          # reference pressures

# 1. Plan: expected error bars at a rough guess of the curve
guess = [0.1, 0.04, 1e-5]
plan = information(sensor, guess, pressures, sigmas=reading_error)
planned = inverse_predict(sensor, reading=sensor.predict(guess, [50.0])[0],
                          bracket=(0.0, 100.0), reading_sigma=reading_error,
                          theta=guess, cov=plan["cov"])
print(f"planned error bar at 50 kPa from one reading: "
      f"{planned['sigma']:.4f} kPa")

# 2. Calibrate (simulated readings of the 11 references)
rng = np.random.default_rng(2026)
truth = [0.1002, 0.03995, 1.002e-5]
volts = sensor.predict(truth, pressures) \
    + reading_error * rng.standard_normal(pressures.size)
cal = fit(sensor, pressures, volts, guess, sigmas=reading_error)
print(f"fitted c0 = {cal.values['c0']:.5f} +/- {cal.sigma['c0']:.2g} V")
print(f"fitted c1 = {cal.values['c1']:.6f} +/- {cal.sigma['c1']:.2g} V/kPa")
print(f"fitted c2 = {cal.values['c2']:.4e} +/- {cal.sigma['c2']:.2g} "
      "V/kPa^2")

# 3. Check the calibration
gof = goodness_of_fit(cal)
print(f"chi2 = {gof['chi2']:.2f} for {gof['dof']} degrees of freedom, "
      f"p = {gof['p_value']:.3f}")
residuals = volts - sensor.predict(cal.theta, pressures)
runs = runs_test(residuals, order=pressures)
print(f"runs of residual signs: {runs['runs']} "
      f"(expected {runs['expected_runs']:.1f}), p = {runs['p_value']:.3f}")

# 4. An unknown, read 5 times; its true pressure is 50 kPa here
readings = sensor.predict(truth, [50.0])[0] \
    + reading_error * rng.standard_normal(5)
mean = readings.mean()
u_mean = reading_error / np.sqrt(readings.size)
print(f"unknown: mean reading {mean:.5f} V +/- {u_mean:.2g} V "
      f"({readings.size} readings)")
p = inverse_predict(sensor, cal, mean, bracket=(0.0, 100.0),
                    reading_sigma=u_mean)
print(f"pressure = {p['value']:.4f} +/- {p['sigma']:.4f} kPa")
print(f"  from the reading noise:       {p['sigma_from_reading']:.4f} kPa")
print(f"  from the calibration curve:   "
      f"{p['sigma_from_parameters']:.4f} kPa")

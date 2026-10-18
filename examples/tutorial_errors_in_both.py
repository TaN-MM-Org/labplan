"""Tutorial: calibrate against reference values that have error bars of
their own, then read an unknown off the calibration.

    pip install labplan
    python tutorial_errors_in_both.py

A temperature sensor gives a voltage V that is (here) a straight line
in temperature, V = c0 + c1 T. It is calibrated by placing it next to
a reference thermometer at 15 temperatures. The reference is good, but
not perfect: each reference temperature has an error bar of 0.05 K,
and the sensor's sensitivity (about 4 mV/K) turns that into 0.2 mV,
as much as the sensor's own reading error. The numbers are
illustrative, and the "measured" values are simulated with a fixed
seed, so the output below is repeatable.

Steps:
1. Fit the line as if the reference temperatures were exact (`fit`).
   Half of the scatter then has no error bar to account for it:
   chi-squared averages twice its degrees of freedom, and the error
   bars of the line are too small (by a factor of about 1.4 here).
2. Fit again with the errors of both coordinates (`fit_xy`, new in
   labplan 0.5.0): it also fits the true temperatures, chi-squared
   averages its degrees of freedom, and the error bars count both
   sources.
3. Check how far the true temperatures were moved, in error bars.
4. Measure an unknown 5 times and read its temperature off the
   calibration with `inverse_predict`, with an error bar from the
   reading noise and from the calibration.

Output (labplan 0.5.0):

    reference points: 15, from 10.0 to 80.0 degC
    x exact:   c1 = 4.0003 +/- 0.0024 mV/K, chi2 = 20.8 for 13 dof, p = 0.076
    x and y:   c1 = 4.0004 +/- 0.0034 mV/K, chi2 = 10.4 for 13 dof, p = 0.66
    reference temperatures moved by 0.59 error bars (rms), at most 1.25
    unknown: mean reading 212.051 mV +/- 0.089 mV (5 readings)
    temperature = 50.013 +/- 0.029 degC
      from the reading noise:   0.022 degC
      from the calibration:     0.019 degC
"""
import numpy as np

from labplan import Model, fit, fit_xy, goodness_of_fit, inverse_predict

sensor = Model(
    name="temperature sensor",
    f=lambda th, x: th[0] + th[1] * x[:, 0],
    param_names=("c0", "c1"),
    reference="illustrative linear sensor response V = c0 + c1 T",
    units={"c0": "mV", "c1": "mV/K"},
)
u_ref = 0.05         # K, error bar of each reference temperature
u_volt = 0.2         # mV, error bar of one sensor reading

# The calibration run (simulated): true temperatures, the reference
# thermometer's readings of them, and the sensor's readings.
rng = np.random.default_rng(2026)
t_true = np.linspace(10.0, 80.0, 15)
t_ref = t_true + u_ref * rng.standard_normal(t_true.size)
volts = sensor.predict([12.0, 4.0], t_true) \
    + u_volt * rng.standard_normal(t_true.size)
print(f"reference points: {t_ref.size}, from {t_true[0]:.1f} to "
      f"{t_true[-1]:.1f} degC")

# 1. The reference temperatures taken as exact
naive = fit(sensor, t_ref, volts, [0.0, 1.0], sigmas=u_volt)
g = goodness_of_fit(naive)
print(f"x exact:   c1 = {naive.values['c1']:.4f} +/- "
      f"{naive.sigma['c1']:.2g} mV/K, chi2 = {g['chi2']:.1f} for "
      f"{g['dof']} dof, p = {g['p_value']:.2g}")

# 2. Both coordinates with their error bars
cal = fit_xy(sensor, t_ref, volts, [0.0, 1.0], sigmas=u_volt,
             x_sigmas=u_ref)
g = goodness_of_fit(cal)
print(f"x and y:   c1 = {cal.values['c1']:.4f} +/- "
      f"{cal.sigma['c1']:.2g} mV/K, chi2 = {g['chi2']:.1f} for "
      f"{g['dof']} dof, p = {g['p_value']:.2g}")

# 3. How far were the reference temperatures moved?
moved = (cal.x_adjusted[:, 0] - t_ref) / u_ref
print(f"reference temperatures moved by {np.sqrt(np.mean(moved ** 2)):.2f}"
      f" error bars (rms), at most {np.max(np.abs(moved)):.2f}")

# 4. An unknown at 50 degC, read 5 times
readings = sensor.predict([12.0, 4.0], [50.0])[0] \
    + u_volt * rng.standard_normal(5)
mean = readings.mean()
u_mean = u_volt / np.sqrt(readings.size)
print(f"unknown: mean reading {mean:.3f} mV +/- {u_mean:.2g} mV "
      f"({readings.size} readings)")
out = inverse_predict(sensor, cal, mean, bracket=(10.0, 80.0),
                      reading_sigma=u_mean)
print(f"temperature = {out['value']:.3f} +/- {out['sigma']:.3f} degC")
print(f"  from the reading noise:   {out['sigma_from_reading']:.3f} degC")
print(f"  from the calibration:     "
      f"{out['sigma_from_parameters']:.3f} degC")

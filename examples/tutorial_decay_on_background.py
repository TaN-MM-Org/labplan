"""Tutorial: a small, fast decay on a large steady level, in SI units.

    pip install labplan
    python tutorial_decay_on_background.py

The signal here is illustrative: a detector voltage that sits at a
steady 1 V and, after a short excitation at t = 0, carries a decay of
10 mV with a time constant near 1 ns:

    V(t) = A exp(-t / tau) + B,   A ~ 0.01 V, tau ~ 1e-9 s, B ~ 1 V.

Everything is written in seconds and volts. labplan 0.3.0 refused
this model at the natural first guess tau = 1e-9 s (the numerical
slope of tau used a step as large as tau itself, and the model
divided by zero); 0.4.0 computes it.

Steps:
1. Plan: with 200 samples from 0 to 10 ns and 0.5 mV noise per
   sample, how many traces must be averaged for tau +/- 10 ps?
2. Measure (simulated, fixed seed): average that many traces.
3. Fit, and check the fit with chi-squared and the runs test.
4. Report the decay rate 1/tau with its error bar.

Output (labplan 0.4.0):

    planned error bar on tau from one trace: 3.57e-11 s
    traces to average for tau +/- 1e-11 s: 13
    A   = 0.0101 +/- 5.9e-05 V
    tau = 9.9383e-10 +/- 9.8e-12 s
    B   = 0.999993 +/- 1.3e-05 V
    chi2 = 150.0 for 197 degrees of freedom, p = 0.995
    runs of residual signs: 105 (expected 101.0), p = 0.740
    decay rate 1/tau = 1.006e+09 +/- 9.9e+06 1/s
"""
import numpy as np

from labplan import (Model, fit, goodness_of_fit, information,
                     propagate, repeats_for, runs_test)

decay = Model(
    name="decay on a steady level",
    f=lambda th, x: th[0] * np.exp(-x[:, 0] / th[1]) + th[2],
    param_names=("A", "tau", "B"),
    reference="illustrative single-exponential decay on a constant "
              "background",
    units={"A": "V", "tau": "s", "B": "V"},
)
t = np.linspace(0.0, 10e-9, 200)          # s
noise = 5e-4                              # V per sample, one trace
guess = [0.01, 1e-9, 1.0]

# 1. Plan
plan = information(decay, guess, t, sigmas=noise)
print(f"planned error bar on tau from one trace: "
      f"{plan['sigma']['tau']:.3g} s")
n_traces, _ = repeats_for({"tau": 10e-12}, plan)
print(f"traces to average for tau +/- 1e-11 s: {n_traces}")

# 2. Measure: the average of n_traces traces has noise / sqrt(n)
rng = np.random.default_rng(7)
truth = [0.0101, 1.004e-9, 1.0]
sigma_avg = noise / np.sqrt(n_traces)
v = decay.predict(truth, t) + sigma_avg * rng.standard_normal(t.size)

# 3. Fit and check
res = fit(decay, t, v, guess, sigmas=sigma_avg)
print(f"A   = {res.values['A']:.4f} +/- {res.sigma['A']:.2g} V")
print(f"tau = {res.values['tau']:.4e} +/- {res.sigma['tau']:.2g} s")
print(f"B   = {res.values['B']:.6f} +/- {res.sigma['B']:.2g} V")
gof = goodness_of_fit(res)
print(f"chi2 = {gof['chi2']:.1f} for {gof['dof']} degrees of freedom, "
      f"p = {gof['p_value']:.3f}")
runs = runs_test(v - decay.predict(res.theta, t), order=t)
print(f"runs of residual signs: {runs['runs']} "
      f"(expected {runs['expected_runs']:.1f}), p = {runs['p_value']:.3f}")

# 4. A derived number with its error bar
rate = propagate(lambda th: 1.0 / th[1], res.theta, res.cov)
print(f"decay rate 1/tau = {rate['value']:.4g} +/- {rate['sigma']:.2g} 1/s")

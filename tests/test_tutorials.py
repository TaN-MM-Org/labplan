"""The tutorials in examples/ run, print what their docstrings (and
the README) say they print, and their key numbers hold up against
independent checks done here: the truth used to simulate the data,
closed forms, and arithmetic on the printed numbers."""
import math
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _run(name):
    path = EXAMPLES / name
    if not path.exists():               # e.g. tests run from an sdist
        pytest.skip(f"{path} is not here")
    out = subprocess.run([sys.executable, str(path)], capture_output=True,
                         text=True, check=True, timeout=120).stdout
    doc = re.search(r"Output \(labplan [0-9.]+\):\n\n(.*?)\n\"\"\"",
                    path.read_text(), re.S).group(1)
    documented = "\n".join(line[4:] for line in doc.split("\n"))
    return out, documented


def _num(pattern, text):
    return [float(v) for v in re.search(pattern, text).groups()]


def test_tutorial_calibration_curve():
    out, documented = _run("tutorial_calibration_curve.py")
    assert out.rstrip("\n") == documented
    readme = (ROOT / "README.md").read_text()
    assert documented in readme
    p, sig = _num(r"pressure = ([0-9.]+) \+/- ([0-9.]+) kPa", out)
    (s_read,) = _num(r"from the reading noise: +([0-9.]+)", out)
    (s_cal,) = _num(r"from the calibration curve: +([0-9.]+)", out)
    # the unknown was simulated at 50 kPa
    assert abs(p - 50.0) < 3 * sig
    # the two parts add in quadrature (printed to 4 decimals)
    assert abs(math.hypot(s_read, s_cal) - sig) < 2e-4
    # reading part: 0.0006 V / sqrt(5) divided by the slope dV/dP at
    # 50 kPa, c1 + 2 c2 P, from the printed coefficients
    c1, = _num(r"c1 = ([0-9.e-]+) \+/-", out)
    c2, = _num(r"c2 = ([0-9.e-]+) \+/-", out)
    assert abs(s_read - 0.0006 / math.sqrt(5) / (c1 + 2 * c2 * p)) < 1e-4
    chi2, dof, pval = _num(r"chi2 = ([0-9.]+) for ([0-9]+) degrees of "
                           r"freedom, p = ([0-9.]+)", out)
    assert dof == 11 - 3 and pval > 0.01
    runs, expected, p_runs = _num(r"runs of residual signs: ([0-9]+) "
                                  r"\(expected ([0-9.]+)\), p = ([0-9.]+)",
                                  out)
    assert p_runs > 0.01


def test_tutorial_decay_on_background():
    out, documented = _run("tutorial_decay_on_background.py")
    assert out.rstrip("\n") == documented
    readme = (ROOT / "README.md").read_text()
    assert documented in readme
    (one,) = _num(r"from one trace: ([0-9.e+-]+) s", out)
    (n,) = _num(r"tau \+/- 1e-11 s: ([0-9]+)", out)
    # repeats_for's closed form, from the printed planned error bar
    # (printed to 3 digits, so allow one either way)
    assert abs(n - math.ceil((one / 1e-11) ** 2)) <= 1
    tau, s_tau = _num(r"tau = ([0-9.e+-]+) \+/- ([0-9.e+-]+) s", out)
    # simulated with tau = 1.004e-9 s; the fitted error bar is the
    # planned one for n averaged traces
    assert abs(tau - 1.004e-9) < 3 * s_tau
    assert abs(s_tau - one / math.sqrt(n)) < 0.1 * s_tau
    rate, s_rate = _num(r"1/tau = ([0-9.e+-]+) \+/- ([0-9.e+-]+) 1/s",
                        out)
    assert rate == pytest.approx(1 / tau, rel=1e-3)
    assert s_rate == pytest.approx(s_tau / tau ** 2, rel=0.02)


def test_tutorial_errors_in_both():
    out, documented = _run("tutorial_errors_in_both.py")
    assert out.rstrip("\n") == documented
    readme = (ROOT / "README.md").read_text()
    assert documented in readme
    c1_n, s_n, chi_n, dof_n, _ = _num(
        r"x exact: +c1 = ([0-9.]+) \+/- ([0-9.]+) mV/K, chi2 = ([0-9.]+) "
        r"for ([0-9]+) dof, p = ([0-9.]+)", out)
    c1, s_c1, chi, dof, p = _num(
        r"x and y: +c1 = ([0-9.]+) \+/- ([0-9.]+) mV/K, chi2 = ([0-9.]+) "
        r"for ([0-9]+) dof, p = ([0-9.]+)", out)
    assert dof == dof_n == 15 - 2 and p > 0.01
    # simulated with c1 = 4 mV/K
    assert abs(c1 - 4.0) < 3 * s_c1
    # For a straight line with the same errors everywhere, fit_xy
    # weighs each residual with 1 / (u_V^2 + c1^2 u_T^2); fit with
    # 1 / u_V^2. So the error bars differ by the factor
    # sqrt(1 + (c1 u_T / u_V)^2) (here 1.414; printed to 2 digits),
    # and fit's chi2, at its own minimum, is at most fit_xy's chi2
    # times that factor squared (printed to 0.1).
    k2 = 1 + (c1 * 0.05 / 0.2) ** 2
    lo = (s_c1 - 0.00005) / (s_n + 0.00005)
    hi = (s_c1 + 0.00005) / (s_n - 0.00005)
    assert lo <= math.sqrt(k2) <= hi
    assert chi_n <= k2 * (chi + 0.05) + 0.05
    t, s_t = _num(r"temperature = ([0-9.]+) \+/- ([0-9.]+) degC", out)
    (s_read,) = _num(r"from the reading noise: +([0-9.]+)", out)
    (s_cal,) = _num(r"from the calibration: +([0-9.]+)", out)
    # the unknown was simulated at 50 degC
    assert abs(t - 50.0) < 3 * s_t
    assert abs(math.hypot(s_read, s_cal) - s_t) < 2e-3
    # reading part: 0.2 mV / sqrt(5) divided by the slope c1
    assert abs(s_read - 0.2 / math.sqrt(5) / c1) < 1e-3

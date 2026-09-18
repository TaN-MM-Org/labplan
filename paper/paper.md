---
title: 'labplan: exact measurement planning, calibration and audit trails for any lab model'
tags:
  - Python
  - metrology
  - experimental design
  - calibration
  - conformal prediction
  - uncertainty quantification
authors:
  - name: Tanvir Mahmud Mahim
    orcid: 0000-0002-4550-3248
    affiliation: 1
affiliations:
  - name: Department of Electrical and Electronic Engineering, BRAC University, Dhaka, Bangladesh
    index: 1
date: 18 September 2026
bibliography: paper.bib
---

# Summary

Every measurement campaign asks the same four questions. Can the
measurements about to be taken determine the numbers of interest, and
how well? Which settings are worth the instrument time? Once the data
exist, what are the numbers, with error bars that mean something? And
a year later, can anyone trace exactly what was fitted, to which data,
by what software? `labplan` answers all four for any instrument or
experiment that can be described by a Python function mapping
parameters and settings to predicted readings.

The package chains four tools around such a model. Fisher-information
analysis [@Fedorov1972] predicts, before any data are taken, the error
bars a weighted least-squares fit will report for a given measurement
plan, and a greedy D-optimal selector picks the most informative
subset of candidate settings using the rank-one determinant update. An
exact closed form converts a target error bar into the number of
repeats it costs, since $r$ identical repeats multiply the information
matrix by $r$. A dependency-free Levenberg--Marquardt fitter
[@Marquardt1963] then estimates the parameters from measured data and
reports the exact covariance of the weighted least-squares problem.
Split-conformal prediction [@Vovk2005; @Lei2018] wraps any resulting
predictor in intervals whose finite-sample coverage is guaranteed by
exchangeability alone, independent of whether the model is correct.
Finally, an audit-record writer produces a JSON document containing
the fitted values with uncertainties, the covariance, an SHA-256
digest of the raw data arrays, the software versions, a UTC timestamp
and the operator, which is the record a laboratory notebook or an
external audit actually requires.

`labplan` is built on NumPy alone [@Harris2020], supports Python 3.9
through 3.14, and requires a literature reference as a mandatory field
of every model rather than as an optional comment.

# Statement of need

Measurement planning, fitting, and uncertainty certification are
usually served by separate tools with separate conventions:
optimal-design theory lives in statistics packages, fitting in
`scipy.optimize` or `lmfit`, conformal prediction in machine-learning
libraries, and provenance in electronic lab notebooks. A working
experimentalist who wants to know "how many repeats until my error bar
is small enough?" before spending beam time, and to hand an auditor a
traceable record afterwards, has to assemble and cross-validate that
chain by hand. `labplan` ships the chain as one small, tested engine
with a deliberate refusal policy: when a design cannot identify the
requested parameters -- an exactly degenerate design, an
ill-conditioned information matrix, a calibration set too small for
the requested conformal level -- the package raises an explanatory
error instead of returning one of many possible answers.

Every statistical claim in the package is pinned by its test suite to
a closed form or an independent code path: the linear-model covariance
$\sigma^2 (X^\top X)^{-1}$ is reproduced exactly, planned error bars
are matched against seeded Monte Carlo, the greedy design is matched
against exhaustive search, the repeats law is verified by literally
tiling designs, and the conformal coverage $\lceil (n+1)(1-\alpha)
\rceil / (n+1)$ is matched against simulation. The same
planning-and-calibration pattern is deployed across the nine
domain-specific research packages of the TaN-MM-Org organization,
covering colour-centre spins, squeezed light, Raman spectroscopy,
superconducting detectors and photonic fabrication; `labplan` is that
pattern extracted, generalized and hardened into a standalone package.

# Acknowledgements

This software is developed at the Department of Electrical and
Electronic Engineering, BRAC University, and released under the
Apache-2.0 license. Each release is archived on Zenodo under the
concept DOI 10.5281/zenodo.22826765.

# References

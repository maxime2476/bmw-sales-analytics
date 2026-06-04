# ADR-0006 — Statistical signal audit & positive control

- **Status:** Accepted
- **Date:** 2026-06-04
- **Author:** Maxime GOURGUECHON
- **Related:** ADR-0002, `reports/signal_audit.md`

## Context

ADR-0002 established (via correlation, ANOVA and MI) that the dataset is signal-
free. Two objections remain that a rigorous reviewer would raise:

1. *"Absence of evidence isn't evidence of absence — prove the null formally."*
2. *"Maybe your pipeline is just broken."*

## Decision

Add a dataset-agnostic `bmw_sales.audit` package providing **falsifiable,
transferable** evidence:

- **Permutation (label-shuffle) test** — compares the real held-out score to a
  null distribution from shuffled labels; a one-sided p-value quantifies the
  probability of seeing the real score by chance. Result: p ≈ 0.90 (regression),
  p ≈ 0.81 (classification) → indistinguishable from chance.
- **Positive control** — runs the *identical* pipeline on a synthetic target that
  is a known function of the features. Result: R² ≈ 0.86 (synthetic) vs ≈ 0
  (real) → the pipeline is sound; the data is empty.
- **KS uniformity** and **chi-squared independence** — fingerprint the synthetic,
  uniform data-generating process.

The audit is surfaced in `reports/signal_audit.md` (via `make audit`) and as a
"Statistical proof" panel in the dashboard's Data Integrity tab.

## Rationale

This converts the project's central claim from an assertion into a **reusable
research instrument**: the same module audits *any* dataset for exploitable
signal. The positive control is the single most convincing artefact — it
pre-empts the "your model is broken" objection with a number.

## Consequences

- **+** Decisive, reproducible evidence; a transferable tool beyond this dataset.
- **+** Strengthens the credibility of every downstream null result.
- **−** Permutation testing is compute-heavy; mitigated by sub-sampling and a fast
  `HistGradientBoosting` estimator, and cached in the app.

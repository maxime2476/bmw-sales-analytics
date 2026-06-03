# ADR-0002 — Data integrity finding & honest-modelling strategy

- **Status:** Accepted
- **Date:** 2026-06-03
- **Author:** Maxime GOURGUECHON
- **Supersedes:** —
- **Related:** ADR-0001, `reports/data_integrity_report.md`

## Context

Before committing to a modelling approach, I ran a formal data-integrity analysis
(`bmw_sales.data.validation`). It produced three reproducible findings:

1. **Structurally pristine** — 50,000 rows, 0 nulls, 0 duplicates.
2. **No predictive signal** — every feature is statistically independent of the
   targets:
   - Max absolute off-diagonal Pearson correlation = **0.0087**.
   - One-way ANOVA of `Sales_Volume` across each categorical: **all p > 0.4**
     (Model 0.87, Region 0.50, Color 0.42, Fuel 0.73, Transmission 0.74).
   - Max mutual information across all features = **0.0042 nats** (captures
     non-linear dependence too → none exists).
   - Mean `Price_USD` is flat across models (74.4k–75.6k), which is economically
     impossible for a real luxury line-up → confirms a synthetic, uniform DGP.
3. **Target leakage** — `Sales_Classification = High ⟺ Sales_Volume ≥ 7000`, with
   zero class overlap. The label is a deterministic threshold on the regression
   target.

## Decision

Adopt an **honest-modelling strategy** with three pillars:

1. **Report honest baselines.** We build the full ML/DL/econometric pipeline and
   report real metrics, expecting **R² ≈ 0** (regression) and **ROC-AUC ≈ 0.5**
   (leakage-free classification). We do not tune toward, or cherry-pick, spurious
   gains.
2. **Demonstrate the leakage explicitly.** We include `Sales_Volume` in one
   classification run to show the trivial ~100% accuracy, then exclude it as the
   correct, leakage-free setup — turning the flaw into a teaching artefact.
3. **Deliver business value through a labelled Scenario Simulator.** Because the
   *data* cannot support forecasting, decision value comes from an explicit
   simulation grounded in (a) price elasticities from the automotive economics
   literature and (b) **real** macro/fuel/FX/CO₂ data pulled from external APIs.
   It is clearly labelled as a what-if decision-support tool, never presented as
   a fit to the historical data.

## Alternatives considered

- **Inflate metrics / overfit** — rejected: dishonest, and trivially detectable
  by any senior reviewer (the held-out R² collapses).
- **Silently re-synthesise a signal into the target** — rejected: would mislead;
  only acceptable if explicitly labelled (which is exactly what the Simulator is).
- **Abandon predictive modelling** — rejected: the pipeline itself is the
  portfolio artefact (engineering, MLOps, testing), independent of dataset signal.

## Consequences

- **+** The deliverable showcases senior judgement: detecting, proving and
  communicating a data problem most pipelines would mask.
- **+** Reproducible: the report regenerates from raw data via `make eda`.
- **−** Headline predictive metrics are intentionally unimpressive; mitigated by
  framing and the Scenario Simulator. This trade-off is the whole point.

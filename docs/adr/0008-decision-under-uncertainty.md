# ADR-0008 — Decision-making under uncertainty in the Scenario Simulator

- **Status:** Accepted
- **Date:** 2026-06-04
- **Author:** Maxime GOURGUECHON
- **Related:** ADR-0002 (Scenario Simulator), `bmw_sales.simulation.uncertainty`

## Context

The Scenario Simulator (ADR-0002) initially returned a **point estimate** of
projected demand (e.g. "+15.8%"). But the elasticities driving it are themselves
**uncertain** — they are priors taken from the literature, not facts. A point
estimate hides that uncertainty and invites false confidence; a strategy team
plans against a *range*, not a single number.

## Decision

Add `bmw_sales.simulation.uncertainty`: place **Gaussian priors** on each
elasticity (`ElasticityPriors`, mean ± sd) and propagate them through the
constant-elasticity demand model with a seeded **Monte-Carlo** simulation
(`simulate_mc`, 5,000 draws). The output is a full predictive **distribution**
(`ScenarioDistribution`) exposing **80% and 95% credible intervals**.

The dashboard surfaces this directly: the projected-volume and net-change KPIs
now carry their 80% CI, and a predictive-distribution chart (with P10 / median /
P90 markers) replaces the illusion of a single answer.

## Rationale

- **Honest decision support.** Reporting "+15.8% [80% CI +6%, +27%]" is what a
  real consultancy delivers; it makes the uncertainty an explicit, plan-able
  quantity rather than hiding it.
- **Transparent and reproducible.** Priors are explicit and user-adjustable in
  the UI; sampling is seeded. This is a deliberately lightweight,
  Bayesian-*flavoured* treatment — full MCMC (e.g. PyMC) would add a heavy
  dependency for no extra decision value at this fidelity, so Monte-Carlo
  propagation of priors was chosen instead.
- **Consistent with the project's spine.** The simulator is, and remains,
  explicitly labelled as a what-if tool — never a fit to the (signal-free) data.

## Consequences

- **+** Outputs are decision-grade ranges with credible intervals.
- **+** No heavy probabilistic-programming dependency; pure NumPy, fast, seeded.
- **−** Gaussian priors can in principle yield implausible tail draws; bounded in
  practice by realistic prior sds and the multiplicative model form.

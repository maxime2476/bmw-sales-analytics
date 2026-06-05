# Predictive Capability — *can the pipeline predict when there IS signal?*

*Generated: 2026-06-05 · Author: Maxime GOURGUECHON*

> Counterpart to the [Signal Audit](signal_audit.md): the same production pipeline, run on a **clearly-labelled signal-bearing target**, validated honestly. Reproduce with `make capability`.

## Result

| Metric | Value |
|---|---|
| **Cross-validated R²** (5-fold) | **0.849 ± 0.003** |
| **Held-out test R²** | **0.854** |
| Verdict | SKILFUL — predicts the known signal |

The CV and held-out scores agree (no overfitting), and the score is stable across folds (low σ) — this is a *validated* model, not a lucky split.

## Learning curve (test R² vs training size)

| Train size | Test R² |
|---|---|
| 559 | 0.809 |
| 1,819 | 0.826 |
| 3,079 | 0.835 |
| 4,339 | 0.843 |
| 5,599 | 0.847 |

Performance rises monotonically with data — the model genuinely learns.

## SHAP — the model recovers the TRUE drivers

The synthetic target was built from region, premium tier, engine size, price and electrification. SHAP ranks exactly those at the top:

| Feature | mean \|SHAP\| |
|---|---|
| num__price_per_litre_engine | 401.9 |
| cat__is_premium_tier_0 | 294.0 |
| cat__Region_Africa | 279.0 |
| cat__is_electrified_0 | 234.6 |
| cat__Region_Europe | 169.6 |
| cat__Region_South America | 158.0 |

## The point

On signal-bearing data the pipeline reaches **R² ≈ 0.85**; on the real BMW data it scores **≈ 0** (see [model_benchmark.md](model_benchmark.md)). The pipeline is sound and predictively competent — the null result is a property of the *data*, proven, not a failure of the modelling.

# ADR-0004 — Deep learning: tested, not assumed

- **Status:** Accepted
- **Date:** 2026-06-03
- **Author:** Maxime GOURGUECHON
- **Related:** ADR-0002, `reports/dl_vs_ml.md`, `reports/model_benchmark.md`

## Context

The brief asks for a deep-learning model *and* a justification of its use versus
classical ML. On 50k tabular rows, gradient-boosted trees are the well-evidenced
default; deep nets typically need more data and tuning to compete. We also know
(ADR-0002) the data is signal-free, which caps achievable performance for *any*
model at "no skill".

## Decision

Implement a regularised **PyTorch tabular MLP** (BatchNorm + dropout + early
stopping) and **benchmark it head-to-head** against the tuned XGBoost/LightGBM/
CatBoost models, using the identical preprocessing and splits. Let the held-out
metrics, not preference, decide.

### Result (held-out)

| Task | Tabular MLP | Best booster |
|---|---|---|
| Regression R² | ≈ −0.005 | ≈ −0.001 |
| Classification ROC-AUC | ≈ 0.49 | ≈ 0.51 |

Both sit at no-skill. The MLP's early stopping firing after ~10 epochs confirms
there is no structure to fit.

## Rationale for the final choice

- **Boosting is the right default** for tabular data of this size: equal (null)
  accuracy here, but cheaper to train, more robust, and directly explainable
  with tree SHAP (used in the app).
- **DL is not justified for this dataset.** We keep the implementation in the
  repo as evidence of the comparison, but the production model is a booster.
- **Reproducibility:** a Windows/Anaconda OpenMP clash (`libiomp5md.dll`) is
  neutralised in-code (`KMP_DUPLICATE_LIB_OK`) so the DL path runs everywhere.

## Consequences

- **+** The DL decision is defensible with numbers, not hand-waving.
- **+** Demonstrates breadth (PyTorch training loop, early stopping) without
  over-engineering the deployed artefact.
- **−** Carries a `torch` dev dependency; kept out of the slim runtime image.

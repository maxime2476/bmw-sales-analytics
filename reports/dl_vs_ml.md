# Deep Learning vs Gradient Boosting — BMW Sales

*Generated: 2026-06-03 · Author: Maxime GOURGUECHON*

> Empirical justification for the modelling choice (see **ADR-0004**).

## Why test deep learning at all?

Gradient-boosted trees are the strong default for tabular data at this scale. Rather than assume DL is unnecessary, we train a regularised MLP (BatchNorm + dropout, early stopping) and let the held-out metrics decide.

## Head-to-head (held-out test set)

### Regression — `Sales_Volume` (higher R² is better)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Tabular MLP (PyTorch) | -0.0054 | 2865.7 | 2479.2 |
| Best gradient booster | -0.0006 | — | — |

### Classification — `Sales_Classification` (higher ROC-AUC is better)

| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| Tabular MLP (PyTorch) | 0.4864 | 0.6951 | 0.0000 |
| Best gradient booster | 0.5078 | — | — |

- MLP size: 14,977 parameters · early-stopped after 10 (reg) / 9 (clf) epochs.

## Conclusion

Both approaches land at **no skill** (R² ≈ 0, ROC-AUC ≈ 0.5) — neither can extract signal that the data does not contain. The MLP's early stopping firing within a handful of epochs is itself evidence: there is no learnable structure to fit.

**Decision:** gradient boosting is the correct default here — comparable (null) accuracy, but far cheaper to train, easier to explain (SHAP on trees), and more robust on 50k tabular rows. Deep learning is **not justified** for this dataset, and we say so rather than ship a heavier model for appearances.

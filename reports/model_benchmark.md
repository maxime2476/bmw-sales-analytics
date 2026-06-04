# Model Benchmark — BMW Sales (2010–2024)

*Generated: 2026-06-03 · Author: Maxime GOURGUECHON*

> Gradient-boosting benchmark on the API-enriched dataset. Metrics are held-out (test split). Read with the [Data Integrity Report](data_integrity_report.md).

## Executive summary

Three tuned gradient-boosting families (XGBoost, LightGBM, CatBoost) are benchmarked. As predicted by the data-integrity analysis, **regression R² ≈ 0** and **leakage-free classification ROC-AUC ≈ 0.5**: there is no signal to learn. The contrast with the leakage run (ROC-AUC ≈ 1.0) validates that our honest setup correctly excludes the leaked column.

## 1. Regression — target `Sales_Volume`

|          |      r2 |    rmse |     mae |
|:---------|--------:|--------:|--------:|
| XGBoost  | -0.0018 | 2860.53 | 2476.98 |
| LightGBM | -0.0024 | 2861.44 | 2477.53 |
| CatBoost | -0.0006 | 2858.88 | 2475.67 |

**Best:** CatBoost (R² = -0.0006). An R² at/below zero means the models do not beat predicting the mean — the honest, expected outcome on noise.

## 2. Classification — target `Sales_Classification` (leakage-free)

|          |   roc_auc |   accuracy |     f1 |
|:---------|----------:|-----------:|-------:|
| XGBoost  |    0.5078 |     0.6951 | 0      |
| LightGBM |    0.5071 |     0.6948 | 0      |
| CatBoost |    0.4952 |     0.6947 | 0.0009 |

**Best:** XGBoost (ROC-AUC = 0.5078). ROC-AUC ≈ 0.5 confirms no discriminative signal once the leaked `Sales_Volume` is correctly removed.

## 3. Leakage demonstration — `Sales_Volume` left in as a feature

|          |   roc_auc |   accuracy |     f1 |
|:---------|----------:|-----------:|-------:|
| XGBoost  |         1 |     1      | 1      |
| LightGBM |         1 |     0.9973 | 0.9956 |
| CatBoost |         1 |     0.9993 | 0.9989 |

**Result:** XGBoost reaches ROC-AUC = 1.0000. This near-perfect score is **not a success** — it is the signature of target leakage (the label is a deterministic threshold on this feature) and is shown here only to make the failure mode explicit.

## Takeaways

- Tuning cannot manufacture signal that the data does not contain.
- A >0.99 classification score on this data is a **red flag**, not a win.
- Decision value is delivered by the Scenario Simulator, not these in-sample models — see ADR-0002.

# Signal Audit — is there anything to learn?

*Generated: 2026-06-04 · Author: Maxime GOURGUECHON*

> A dataset-agnostic, falsifiable audit. Reproduce with `python -m bmw_sales.audit.report`.

## 1. Positive control — *does the pipeline even work?*

We run the **identical** pipeline on a synthetic target engineered to be a known function of the features.

| Target | Held-out R² (LightGBM) |
|---|---|
| Real `Sales_Volume` | **-0.0444** |
| Synthetic signal-bearing target | **+0.8559** |

> Pipeline VALIDATED: it recovers a known signal (R²=0.856) but scores ~0 on the real target (R²=-0.044) — the null result is a property of the data, not a modelling failure.

## 2. Permutation (label-shuffle) test

The strongest test for exploitable signal: compare the real held-out score to a null distribution from shuffled labels.

- **regression (R²)** — observed = **-0.1110**, null = -0.0931 ± 0.0163 over 30 shuffles, **p = 0.903** → NO SIGNAL (indistinguishable from chance)
- **classification (ROC-AUC)** — observed = **+0.4817**, null = +0.4991 ± 0.0181 over 30 shuffles, **p = 0.806** → NO SIGNAL (indistinguishable from chance)

## 3. Kolmogorov–Smirnov uniformity test

Are the numeric features drawn from a Uniform distribution (a synthetic-data fingerprint)?

| Feature | KS stat | p-value | Verdict |
|---|---|---|---|
| Year | 0.0685 | 0.000 | not uniform |
| Engine_Size_L | 0.0174 | 0.000 | not uniform |
| Mileage_KM | 0.0045 | 0.259 | ✓ uniform |
| Price_USD | 0.0019 | 0.992 | ✓ uniform |
| Sales_Volume | 0.0050 | 0.157 | ✓ uniform |

## 4. Chi-squared independence (categoricals)

| Pair | p-value | Verdict |
|---|---|---|
| Model vs Region | 0.724 | independent |
| Fuel_Type vs Transmission | 0.659 | independent |
| Region vs Color | 0.432 | independent |

## Conclusion

The positive control proves the pipeline recovers signal when it exists. The permutation test shows the real targets are **indistinguishable from shuffled labels**, and the KS tests are consistent with **uniformly generated** features. Together these are decisive, transferable evidence that the dataset is synthetic noise — see ADR-0002.

# Econometric Analysis — BMW Sales (2010–2024)

*Generated: 2026-06-03 · Author: Maxime GOURGUECHON*

> Inference-focused models with HC3 robust standard errors. Read alongside the [Data Integrity Report](data_integrity_report.md).

## Executive summary

We estimate a hedonic price model, a demand model augmented with real macro drivers, and the price elasticity of demand. Consistent with the data-integrity finding, **no specification achieves meaningful explanatory power** (R² ≈ 0, effects insignificant). This is reported transparently: the rigour is in the method (robust SE, VIF diagnostics, F-tests), and forward-looking business value is delivered separately by the Scenario Simulator using literature elasticities.

## 1. Price elasticity of demand

- **Elasticity (∂log Q / ∂log P)** = -0.0010 (95% CI [-0.021, +0.019], p = 0.923, N = 50,000)
- **Interpretation:** Not statistically distinguishable from zero — no measurable price sensitivity in this dataset (consistent with a signal-free DGP).

## 2. Regression models

### Hedonic price model

`log_price ~ C(Model) + C(Region) + C(Fuel_Type) + C(Transmission) + Engine_Size_L + vehicle_age + log_mileage`

- **N** = 50,000 · **R²** = 0.0004 · **Adj. R²** = -0.0000 · **F p-value** = 0.534 · SE = HC3
- **Verdict:** NO EXPLANATORY POWER (significant terms at 5%: C(Model)[T.M5])

<details><summary>Coefficient table (first 15 terms)</summary>

|                            |     coef |   std_err |   p_value |   ci_low |   ci_high |
|:---------------------------|---------:|----------:|----------:|---------:|----------:|
| Intercept                  | 11.1647  |   0.02165 |   0       | 11.1222  |  11.2071  |
| C(Model)[T.5 Series]       | -0.00497 |   0.00799 |   0.53393 | -0.02063 |   0.01069 |
| C(Model)[T.7 Series]       |  0.00102 |   0.00791 |   0.89723 | -0.01447 |   0.01652 |
| C(Model)[T.M3]             | -0.00976 |   0.00803 |   0.22467 | -0.0255  |   0.00599 |
| C(Model)[T.M5]             | -0.01643 |   0.00806 |   0.04165 | -0.03223 |  -0.00062 |
| C(Model)[T.X1]             | -0.00459 |   0.00798 |   0.56574 | -0.02023 |   0.01106 |
| C(Model)[T.X3]             | -0.00864 |   0.00804 |   0.28256 | -0.02441 |   0.00712 |
| C(Model)[T.X5]             | -0.01233 |   0.00803 |   0.12469 | -0.02806 |   0.00341 |
| C(Model)[T.X6]             | -0.01566 |   0.00802 |   0.05074 | -0.03138 |   5e-05   |
| C(Model)[T.i3]             | -0.01001 |   0.00794 |   0.20773 | -0.02558 |   0.00556 |
| C(Model)[T.i8]             | -0.00086 |   0.00791 |   0.91321 | -0.01636 |   0.01463 |
| C(Region)[T.Asia]          |  0.00971 |   0.0059  |   0.10002 | -0.00186 |   0.02128 |
| C(Region)[T.Europe]        |  0.00147 |   0.00593 |   0.80387 | -0.01015 |   0.0131  |
| C(Region)[T.Middle East]   | -0.00271 |   0.00594 |   0.64857 | -0.01435 |   0.00893 |
| C(Region)[T.North America] |  0.00218 |   0.00594 |   0.71376 | -0.00946 |   0.01382 |

</details>

### Demand model

`Sales_Volume ~ log_price + vehicle_age + C(Region) + C(Fuel_Type) + inflation_pct + gdp_per_capita_usd + price_usd_per_litre`

- **N** = 50,000 · **R²** = 0.0001 · **Adj. R²** = -0.0001 · **F p-value** = 0.88 · SE = HC3
- **Verdict:** NO EXPLANATORY POWER (significant terms at 5%: none)

<details><summary>Coefficient table (first 15 terms)</summary>

|                            |       coef |    std_err |   p_value |      ci_low |     ci_high |
|:---------------------------|-----------:|-----------:|----------:|------------:|------------:|
| Intercept                  | 4956.37    |  441.277   |   0       |  4091.49    |  5821.26    |
| C(Region)[T.Asia]          |  214.936   |  782.8     |   0.78364 | -1319.32    |  1749.2     |
| C(Region)[T.Europe]        |  512.225   | 3037.57    |   0.86609 | -5441.3     |  6465.75    |
| C(Region)[T.Middle East]   |  342.612   | 1705.17    |   0.84076 | -2999.46    |  3684.69    |
| C(Region)[T.North America] |  807.802   | 5003.05    |   0.87173 | -8997.99    | 10613.6     |
| C(Region)[T.South America] |  102.263   |  613.155   |   0.86754 | -1099.5     |  1304.02    |
| C(Fuel_Type)[T.Electric]   |   41.2814  |   79.1622  |   0.60203 |  -113.874   |   196.436   |
| C(Fuel_Type)[T.Hybrid]     |   26.1485  |   55.3608  |   0.63669 |   -82.3566  |   134.654   |
| C(Fuel_Type)[T.Petrol]     |  -47.4925  |   37.5197  |   0.20558 |  -121.03    |    26.0447  |
| log_price                  |   -7.29901 |   33.4787  |   0.82741 |   -72.916   |    58.318   |
| vehicle_age                |   -0.7484  |    5.08416 |   0.88297 |   -10.7132  |     9.21637 |
| inflation_pct              |   15.4141  |   15.956   |   0.33402 |   -15.859   |    46.6873  |
| gdp_per_capita_usd         |   -0.01091 |    0.08195 |   0.89408 |    -0.17153 |     0.14971 |
| price_usd_per_litre        |  103.676   |  116.821   |   0.37482 |  -125.289   |   332.64    |

</details>

## 3. Multicollinearity diagnostic (VIF)

VIF > 10 would indicate problematic collinearity among regressors.

| feature                |   VIF |
|:-----------------------|------:|
| Engine_Size_L          | 4.785 |
| Mileage_KM             | 4.026 |
| vehicle_age            | 1     |
| log_price              | 4.824 |
| log_mileage            | 4.026 |
| price_per_litre_engine | 8.61  |

## 4. Target-leakage proof

CONFIRMED leakage: Sales_Classification == 'High' ⟺ Sales_Volume ≥ 7000 (perfectly separable, threshold rule accuracy = 1.0000).

- High class min volume = 7,000 · Low class max volume = 6,999 → classes are perfectly separable.
- A trivial rule `Sales_Volume ≥ 7000` reproduces the label with accuracy 1.0000. The column **must be excluded** as a classification feature.

## Business insights

1. **Pricing power is unobservable in this data.** A real hedonic model would attribute price to model tier and engine size; here those effects are null, so list-price optimisation must rely on external benchmarks, not this dataset.
2. **Demand is macro-insensitive in-sample**, so regional go-to-market decisions should be driven by the (real) external macro/regulatory signals surfaced in the Simulator rather than historical volumes.
3. **The classification target is unusable as-is** (leakage); any reported >0.99 accuracy elsewhere would be a red flag.

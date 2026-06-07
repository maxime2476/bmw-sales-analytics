# Causal Analysis — does price *cause* demand?

*Generated: 2026-06-07 · Author: Maxime GOURGUECHON*

> Backdoor adjustment under an explicit DAG (see `econometrics/causal.py`). Reproduce with `make causal`.

## Assumed DAG

`Region, Model tier, Year, Engine` are confounders of **Price → Demand**. Conditioning on the adjustment set blocks the backdoor paths.

## Estimates (HC3 robust SE)

| Estimate | log-price coefficient | p-value |
|---|---|---|
| Naïve (unadjusted) | -0.0009 | 0.927 |
| **Backdoor-adjusted** | **-0.0011** | 0.915 |

Adjustment set: `C(Region), C(Model), Year, Engine_Size_L, vehicle_age`.

## Conclusion

**no causal price→demand effect (consistent with a signal-free DGP).** Under the stated assumptions, the adjusted effect is statistically indistinguishable from zero — there is no evidence that price causally moves demand in this dataset. The value here is the **method**: an explicit DAG, a justified adjustment set, and an honest null. For forward-looking price effects grounded in the literature, see the Scenario Simulator.

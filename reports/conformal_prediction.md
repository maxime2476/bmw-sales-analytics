# Conformal Prediction - calibrated, honest uncertainty

*Generated: 2026-06-08 · Author: Maxime GOURGUECHON*

> Split-conformal intervals targeting **90% coverage**. Reproduce with `make conformal`.

| Target | Empirical coverage | Interval (half-width) | Relative width | Verdict |
|---|---|---|---|---|
| Real `Sales_Volume` | 89% | ±4,706 (9,412 wide) | 95% of range | honest *'I don't know'* |
| Synthetic signal-bearing | 91% | ±710 (1,421 wide) | 22% of range | informative |

## Reading this

- Both targets achieve **≈ 90% coverage** - the conformal guarantee holds regardless of model quality.
- On the **real** data the interval spans **95%** of the target range: the model honestly says *'I don't know'* (there is no signal to narrow it).
- On **signal-bearing** data the interval collapses to **22%** of the range - calibrated, *useful* uncertainty.

The interval **width** is thus a principled measure of how much the data actually permits you to say - the perfect complement to the project's honest-analytics thesis (see the Signal Audit and Predictive Capability).

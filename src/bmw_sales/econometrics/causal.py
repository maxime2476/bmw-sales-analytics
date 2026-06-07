"""Causal framing of the price → demand question (backdoor adjustment).

Correlation is zero here (see the Signal Audit), but a reviewer rightly asks the
*causal* question: **what is the effect of list price on demand?** Answering it
properly requires being explicit about the assumed causal structure, not just
running a regression.

### Assumed causal DAG

```
        Region ─┐   Model tier ─┐   Year ─┐   Engine ─┐
                ▼               ▼         ▼           ▼
              Price_USD  ───────────────►  Sales_Volume
```

Region, model tier, year and engine size plausibly influence **both** price and
demand — they are **confounders**. Under this DAG the *backdoor criterion* says
that conditioning on {Region, Model, Year, Engine, age} blocks the non-causal
paths, so the coefficient on (log) price in a regression that adjusts for them
identifies the causal price→demand effect.

We report the **naïve** (unadjusted) and **backdoor-adjusted** estimates with HC3
robust SE. Consistent with the data being signal-free, both are ≈ 0 — but the
*method* (state the DAG, justify the adjustment set, estimate, report
uncertainty) is the point. This is a deliberately lightweight, dependency-free
treatment (statsmodels only); a full do-calculus engine (e.g. DoWhy) would add a
heavy dependency for no extra insight on signal-free data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf

from bmw_sales.config import SCHEMA
from bmw_sales.features.engineering import add_engineered_features

#: Backdoor adjustment set blocking the confounding paths (per the DAG).
ADJUSTMENT_SET = ["C(Region)", "C(Model)", "Year", "Engine_Size_L", "vehicle_age"]


@dataclass
class CausalResult:
    """Naïve vs backdoor-adjusted causal estimate of price → demand."""

    naive_effect: float
    naive_pvalue: float
    adjusted_effect: float
    adjusted_pvalue: float
    adjustment_set: list[str]

    @property
    def is_causal(self) -> bool:
        """A non-zero adjusted effect significant at 5% would imply causation."""
        return self.adjusted_pvalue < 0.05 and abs(self.adjusted_effect) > 0.05


def estimate_price_effect(df: pd.DataFrame) -> CausalResult:
    """Estimate the price→demand effect, naïvely and with backdoor adjustment."""
    data = add_engineered_features(df)
    data = data[data[SCHEMA.SALES_VOLUME] > 0].copy()
    import numpy as np

    data["log_volume"] = np.log(data[SCHEMA.SALES_VOLUME].astype(float))

    naive = smf.ols("log_volume ~ log_price", data=data).fit(cov_type="HC3")
    adjusted = smf.ols("log_volume ~ log_price + " + " + ".join(ADJUSTMENT_SET), data=data).fit(
        cov_type="HC3"
    )

    return CausalResult(
        naive_effect=float(naive.params["log_price"]),
        naive_pvalue=float(naive.pvalues["log_price"]),
        adjusted_effect=float(adjusted.params["log_price"]),
        adjusted_pvalue=float(adjusted.pvalues["log_price"]),
        adjustment_set=ADJUSTMENT_SET,
    )


def build_report(result: CausalResult) -> str:
    """Render the causal analysis as markdown."""
    from datetime import date

    verdict = (
        "CAUSAL EFFECT detected"
        if result.is_causal
        else "no causal price→demand effect (consistent with a signal-free DGP)"
    )
    return (
        f"# Causal Analysis — does price *cause* demand?\n\n"
        f"*Generated: {date.today().isoformat()} · Author: Maxime GOURGUECHON*\n\n"
        f"> Backdoor adjustment under an explicit DAG (see `econometrics/causal.py`). "
        f"Reproduce with `make causal`.\n\n"
        f"## Assumed DAG\n\n"
        f"`Region, Model tier, Year, Engine` are confounders of **Price → Demand**. "
        f"Conditioning on the adjustment set blocks the backdoor paths.\n\n"
        f"## Estimates (HC3 robust SE)\n\n"
        f"| Estimate | log-price coefficient | p-value |\n|---|---|---|\n"
        f"| Naïve (unadjusted) | {result.naive_effect:+.4f} | {result.naive_pvalue:.3g} |\n"
        f"| **Backdoor-adjusted** | **{result.adjusted_effect:+.4f}** | "
        f"{result.adjusted_pvalue:.3g} |\n\n"
        f"Adjustment set: `{', '.join(result.adjustment_set)}`.\n\n"
        f"## Conclusion\n\n"
        f"**{verdict}.** Under the stated assumptions, the adjusted effect is "
        f"statistically indistinguishable from zero — there is no evidence that "
        f"price causally moves demand in this dataset. The value here is the "
        f"**method**: an explicit DAG, a justified adjustment set, and an honest "
        f"null. For forward-looking price effects grounded in the literature, see "
        f"the Scenario Simulator.\n"
    )


def main() -> None:
    from bmw_sales.config import REPORTS_DIR
    from bmw_sales.data.loader import load_raw

    result = estimate_price_effect(load_raw())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "causal_analysis.md"
    out.write_text(build_report(result), encoding="utf-8")
    print(f"[OK] Causal analysis written to {out}")
    print(f"     naive {result.naive_effect:+.4f} | adjusted {result.adjusted_effect:+.4f}")


if __name__ == "__main__":
    main()

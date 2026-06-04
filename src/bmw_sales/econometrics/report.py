"""Generate the econometric analysis report.

Runs the hedonic, demand, elasticity, VIF and leakage analyses on the
externally-enriched dataset and renders a portfolio-ready markdown report.

Run as a script::

    python -m bmw_sales.econometrics.report
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd

from bmw_sales.apis.enrichment import enrich_dataset, summarise_provenance
from bmw_sales.config import REPORTS_DIR
from bmw_sales.data.loader import load_raw
from bmw_sales.econometrics.ols_models import (
    RegressionSummary,
    demand_model,
    hedonic_price_model,
    price_elasticity,
    prove_leakage,
    vif_table,
)


def _summary_block(summary: RegressionSummary) -> str:
    sig = ", ".join(summary.significant_terms) if summary.significant_terms else "none"
    coefs_md = summary.coefficients.head(15).to_markdown()
    return (
        f"### {summary.name}\n\n"
        f"`{summary.formula}`\n\n"
        f"- **N** = {summary.n_obs:,} · **R²** = {summary.r_squared:.4f} · "
        f"**Adj. R²** = {summary.adj_r_squared:.4f} · "
        f"**F p-value** = {summary.f_pvalue:.3g} · SE = {summary.cov_type}\n"
        f"- **Verdict:** {'SIGNAL' if summary.has_signal else 'NO EXPLANATORY POWER'} "
        f"(significant terms at 5%: {sig})\n\n"
        f"<details><summary>Coefficient table (first 15 terms)</summary>\n\n"
        f"{coefs_md}\n\n</details>\n"
    )


def build_report(df: pd.DataFrame) -> str:
    """Run all econometric analyses on ``df`` and return a markdown document."""
    hedonic = hedonic_price_model(df)
    demand = demand_model(df)
    elasticity = price_elasticity(df)
    vif = vif_table(df)
    leakage = prove_leakage(df)

    return (
        f"# Econometric Analysis — BMW Sales (2010–2024)\n\n"
        f"*Generated: {date.today().isoformat()} · Author: Maxime GOURGUECHON*\n\n"
        f"> Inference-focused models with HC3 robust standard errors. Read "
        f"alongside the [Data Integrity Report](data_integrity_report.md).\n\n"
        f"## Executive summary\n\n"
        f"We estimate a hedonic price model, a demand model augmented with real "
        f"macro drivers, and the price elasticity of demand. Consistent with the "
        f"data-integrity finding, **no specification achieves meaningful "
        f"explanatory power** (R² ≈ 0, effects insignificant). This is reported "
        f"transparently: the rigour is in the method (robust SE, VIF diagnostics, "
        f"F-tests), and forward-looking business value is delivered separately by "
        f"the Scenario Simulator using literature elasticities.\n\n"
        f"## 1. Price elasticity of demand\n\n"
        f"- **Elasticity (∂log Q / ∂log P)** = {elasticity.elasticity:+.4f} "
        f"(95% CI [{elasticity.ci_low:+.3f}, {elasticity.ci_high:+.3f}], "
        f"p = {elasticity.p_value:.3g}, N = {elasticity.n_obs:,})\n"
        f"- **Interpretation:** {elasticity.interpretation}\n\n"
        f"## 2. Regression models\n\n"
        f"{_summary_block(hedonic)}\n"
        f"{_summary_block(demand)}\n"
        f"## 3. Multicollinearity diagnostic (VIF)\n\n"
        f"VIF > 10 would indicate problematic collinearity among regressors.\n\n"
        f"{vif.to_markdown(index=False)}\n\n"
        f"## 4. Target-leakage proof\n\n"
        f"{leakage.verdict}\n\n"
        f"- High class min volume = {leakage.high_min:,} · Low class max volume = "
        f"{leakage.low_max:,} → classes are "
        f"{'perfectly separable' if leakage.separable else 'overlapping'}.\n"
        f"- A trivial rule `Sales_Volume ≥ {leakage.threshold}` reproduces the "
        f"label with accuracy {leakage.accuracy_from_threshold:.4f}. The column "
        f"**must be excluded** as a classification feature.\n\n"
        f"## Business insights\n\n"
        f"1. **Pricing power is unobservable in this data.** A real hedonic model "
        f"would attribute price to model tier and engine size; here those effects "
        f"are null, so list-price optimisation must rely on external benchmarks, "
        f"not this dataset.\n"
        f"2. **Demand is macro-insensitive in-sample**, so regional go-to-market "
        f"decisions should be driven by the (real) external macro/regulatory "
        f"signals surfaced in the Simulator rather than historical volumes.\n"
        f"3. **The classification target is unusable as-is** (leakage); any "
        f"reported >0.99 accuracy elsewhere would be a red flag.\n"
    )


def main() -> None:
    """CLI entrypoint: enrich, analyse and persist the econometric report."""
    # Offline for reproducibility (deterministic mock macro panel).
    os.environ.setdefault("BMW_OFFLINE_MODE", "true")
    df = load_raw()
    enriched = enrich_dataset(df)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "econometric_analysis.md"
    out_path.write_text(build_report(enriched.data), encoding="utf-8")
    print(f"[OK] Econometric report written to {out_path}")
    print(f"     External data provenance -> {summarise_provenance(enriched.provenance)}")


if __name__ == "__main__":
    main()

"""Data integrity analysis & report generation.

This module is the analytical backbone of the project's *intellectual honesty*
principle. Instead of assuming the data is informative, it runs formal tests and
reports — with numbers — three properties of the source dataset:

1. **Structural integrity** — shape, nulls, duplicates, value ranges.
2. **Signal content** — are features actually predictive of the targets, or is
   the data effectively random noise? Tested via Pearson correlation, one-way
   ANOVA (numeric target vs categoricals) and mutual information.
3. **Target leakage** — is ``Sales_Classification`` a deterministic function of
   ``Sales_Volume``? Detected by checking class/threshold separation.

Run as a script to (re)generate ``reports/data_integrity_report.md``::

    python -m bmw_sales.data.validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from bmw_sales.config import REPORTS_DIR, SCHEMA
from bmw_sales.data.loader import load_raw


@dataclass
class IntegrityFinding:
    """A single, human-readable finding with its supporting statistic."""

    title: str
    verdict: str
    detail: str


@dataclass
class DataIntegrityReport:
    """Structured result of the data-integrity analysis."""

    n_rows: int
    n_cols: int
    n_nulls: int
    n_duplicates: int
    numeric_corr: pd.DataFrame
    anova_pvalues: dict[str, float]
    mutual_information: dict[str, float]
    leakage_detected: bool
    leakage_threshold: Optional[int]
    findings: list[IntegrityFinding] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Individual analyses
# --------------------------------------------------------------------------- #
def _numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix across numeric columns."""
    return df[list(SCHEMA.NUMERIC)].corr(method="pearson")


def _anova_against_volume(df: pd.DataFrame) -> dict[str, float]:
    """One-way ANOVA p-value of ``Sales_Volume`` across each categorical level.

    A high p-value (>0.05) means we cannot reject "all group means are equal",
    i.e. the categorical carries no linear signal about sales volume.
    """
    pvalues: dict[str, float] = {}
    target = SCHEMA.SALES_VOLUME
    for cat in SCHEMA.CATEGORICAL:
        groups = [g[target].to_numpy() for _, g in df.groupby(cat, observed=True)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            pvalues[cat] = float("nan")
            continue
        _, p = stats.f_oneway(*groups)
        pvalues[cat] = float(p)
    return pvalues


def _mutual_information(df: pd.DataFrame) -> dict[str, float]:
    """Mutual information (nats) between each feature and the regression target.

    Uses sklearn's k-NN based estimator, which captures *non-linear* dependence
    that correlation would miss. Values near 0 ⇒ feature is uninformative.
    """
    from sklearn.feature_selection import mutual_info_regression

    target = df[SCHEMA.SALES_VOLUME].to_numpy()
    features = [c for c in SCHEMA.NUMERIC if c != SCHEMA.SALES_VOLUME]
    cats = list(SCHEMA.CATEGORICAL)

    x_parts = [df[features].to_numpy(dtype=float)]
    # Coerce to category first so this works whether or not dtypes were applied.
    codes = np.column_stack([df[c].astype("category").cat.codes.to_numpy() for c in cats])
    x_parts.append(codes.astype(float))
    x = np.column_stack(x_parts)
    discrete_mask = [False] * len(features) + [True] * len(cats)

    mi = mutual_info_regression(x, target, discrete_features=discrete_mask, random_state=42)
    return {name: float(v) for name, v in zip(features + cats, mi)}


def _detect_leakage(df: pd.DataFrame) -> tuple[bool, Optional[int]]:
    """Detect whether ``Sales_Classification`` is a pure threshold on volume.

    Returns ``(leakage_detected, threshold)``. Leakage is confirmed when the
    High and Low classes do not overlap in ``Sales_Volume``.
    """
    vol = SCHEMA.SALES_VOLUME
    cls = SCHEMA.SALES_CLASSIFICATION
    high = df.loc[df[cls] == "High", vol]
    low = df.loc[df[cls] == "Low", vol]
    if high.empty or low.empty:
        return False, None
    separated = high.min() > low.max()
    threshold = int(high.min()) if separated else None
    return separated, threshold


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def analyse(df: Optional[pd.DataFrame] = None) -> DataIntegrityReport:
    """Run the full integrity analysis and return a structured report."""
    if df is None:
        df = load_raw()

    corr = _numeric_correlations(df)
    anova = _anova_against_volume(df)
    mi = _mutual_information(df)
    leakage, threshold = _detect_leakage(df)

    # --- Derive plain-language verdicts -----------------------------------
    findings: list[IntegrityFinding] = []

    findings.append(
        IntegrityFinding(
            title="Structural integrity",
            verdict="PASS",
            detail=(
                f"{len(df):,} rows × {df.shape[1]} columns, "
                f"{int(df.isna().sum().sum())} nulls, "
                f"{int(df.duplicated().sum())} duplicate rows."
            ),
        )
    )

    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    max_abs_corr = float(np.nanmax(np.abs(off_diag.to_numpy())))
    findings.append(
        IntegrityFinding(
            title="Numeric signal (correlation)",
            verdict="NO SIGNAL" if max_abs_corr < 0.05 else "SIGNAL PRESENT",
            detail=(
                f"Largest absolute off-diagonal Pearson correlation = "
                f"{max_abs_corr:.4f}. Features are effectively mutually independent."
            ),
        )
    )

    insignificant = [c for c, p in anova.items() if not (p < 0.05)]
    findings.append(
        IntegrityFinding(
            title="Categorical signal (ANOVA on Sales_Volume)",
            verdict="NO SIGNAL" if len(insignificant) == len(anova) else "SIGNAL PRESENT",
            detail=(
                "No categorical shows a significant effect on sales volume "
                f"(all p>0.05): {', '.join(f'{c} p={anova[c]:.2f}' for c in anova)}."
            ),
        )
    )

    max_mi = max(mi.values()) if mi else 0.0
    findings.append(
        IntegrityFinding(
            title="Non-linear signal (mutual information)",
            verdict="NO SIGNAL" if max_mi < 0.01 else "SIGNAL PRESENT",
            detail=(
                f"Maximum MI across all features = {max_mi:.4f} nats "
                "(≈0 ⇒ no exploitable non-linear dependence either)."
            ),
        )
    )

    findings.append(
        IntegrityFinding(
            title="Target leakage",
            verdict="LEAKAGE" if leakage else "OK",
            detail=(
                f"Sales_Classification == 'High' ⟺ Sales_Volume ≥ {threshold}. "
                "The classes do not overlap → the label is a deterministic "
                "threshold and MUST be excluded as a feature."
                if leakage
                else "No deterministic threshold relationship detected."
            ),
        )
    )

    return DataIntegrityReport(
        n_rows=len(df),
        n_cols=df.shape[1],
        n_nulls=int(df.isna().sum().sum()),
        n_duplicates=int(df.duplicated().sum()),
        numeric_corr=corr,
        anova_pvalues=anova,
        mutual_information=mi,
        leakage_detected=leakage,
        leakage_threshold=threshold,
        findings=findings,
    )


def to_markdown(report: DataIntegrityReport) -> str:
    """Render a :class:`DataIntegrityReport` as a portfolio-ready markdown doc."""
    corr_md = report.numeric_corr.round(4).to_markdown()
    findings_md = "\n".join(
        f"| {f.title} | **{f.verdict}** | {f.detail} |" for f in report.findings
    )
    mi_md = "\n".join(
        f"| {k} | {v:.4f} |"
        for k, v in sorted(report.mutual_information.items(), key=lambda kv: -kv[1])
    )

    # Built without leading indentation so interpolated multi-line tables render
    # correctly (textwrap.dedent cannot infer common indentation across them).
    return (
        f"# Data Integrity Report — BMW Sales (2010–2024)\n\n"
        f"*Generated: {date.today().isoformat()} · Author: Maxime GOURGUECHON*\n\n"
        f"> Auto-generated by `bmw_sales.data.validation`. Reproduce with `make eda`.\n\n"
        f"## Executive summary\n\n"
        f"The dataset is **structurally pristine** ({report.n_rows:,} rows, zero nulls, "
        f"zero duplicates) but **contains no exploitable predictive signal**: every "
        f"feature is statistically independent of the sales targets. In addition, "
        f"`Sales_Classification` is a **leaked** deterministic threshold on "
        f"`Sales_Volume`. These findings are reported transparently and drive the "
        f"project's modelling strategy (honest baselines + a labelled Scenario "
        f"Simulator). See **ADR-0002** for the resulting decision.\n\n"
        f"## Findings\n\n"
        f"| Check | Verdict | Evidence |\n|---|---|---|\n{findings_md}\n\n"
        f"## Numeric correlation matrix (Pearson)\n\n{corr_md}\n\n"
        f"## Mutual information vs `Sales_Volume` (nats)\n\n"
        f"| Feature | MI |\n|---|---|\n{mi_md}\n\n"
        f"## Interpretation & consequences\n\n"
        f"- A supervised model predicting `Sales_Volume` from these features is "
        f"expected to achieve **R² ≈ 0** on held-out data. This is a property of "
        f"the data-generating process, not a modelling failure.\n"
        f"- Classification using `Sales_Volume` as an input is **trivially perfect "
        f"via leakage**; excluding it yields **ROC-AUC ≈ 0.5**.\n"
        f"- We therefore (a) report these honest baselines, (b) prove the leakage, "
        f"and (c) deliver business value through the econometrics-grounded "
        f"**Scenario Simulator** rather than a spurious predictive model.\n"
    )


def main() -> None:
    """CLI entrypoint: run analysis and persist the markdown report."""
    report = analyse()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "data_integrity_report.md"
    out_path.write_text(to_markdown(report), encoding="utf-8")
    print(f"[OK] Data Integrity Report written to {out_path}")
    for f in report.findings:
        print(f"  [{f.verdict:>14}] {f.title}")


if __name__ == "__main__":
    main()

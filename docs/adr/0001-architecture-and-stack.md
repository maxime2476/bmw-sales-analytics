# ADR-0001 — Project architecture & technology stack

- **Status:** Accepted
- **Date:** 2026-06-03
- **Author:** Maxime GOURGUECHON

## Context

We are building a production-grade analytics & forecasting platform on the
`BMW_sales_data_(2010-2024)` dataset (50,000 rows, 11 columns, no missing values).
The deliverable targets a senior data-science portfolio: it must demonstrate
econometrics, ML, DL, external-data augmentation, a premium UI, containerisation
and CI/CD — with clean, modular, typed, tested code.

## Decision

### Layout — `src/` layout package (`bmw_sales`)
A `src/`-layout installable package (`pip install -e .`) rather than loose
notebook scripts. This forces explicit imports, prevents accidental reliance on
the CWD, and makes the same code importable from tests, the Streamlit app and
Docker identically.

```
src/bmw_sales/
  config.py          # typed settings + canonical dataset schema
  data/              # loading, validation, integrity report
  audit/             # No-Signal Auditor: permutation, positive control (ADR-0006)
  apis/              # hybrid real+mock external clients
  features/          # feature engineering pipelines
  econometrics/      # OLS / hedonic / elasticity models
  models/            # ML (XGB/LGBM/CatBoost) + DL (tabular NN) + MLflow tracking
  simulation/        # Scenario Simulator + Monte-Carlo uncertainty (ADR-0008)
  explainability/    # SHAP analysis
  sql/               # DuckDB analytics over sql/queries/*.sql (ADR-0007)
app/                 # Streamlit premium UI (7 tabs)
tests/               # pytest (unit + integration)
docs/                # MkDocs Material site + ADRs (ADR-0009)
```

> This layout has grown with the project; the modules added later are recorded in
> their own ADRs (signal audit · SQL · uncertainty · observability), cross-linked
> above and listed in the [ADR index](https://github.com/maxime2476/bmw-sales-analytics#9-architecture-decision-records).

### Configuration — `pydantic-settings`
A single typed `Settings` object reads from env / `.env`. No magic strings; the
dataset schema lives in one `DatasetSchema` class so a rename is one edit and is
type-checked.

### Stack rationale
| Concern | Choice | Why |
|---|---|---|
| Econometrics | `statsmodels` | p-values, confidence intervals, robust SE — explanatory, not just predictive. |
| ML | XGBoost / LightGBM / CatBoost | SOTA gradient boosting on tabular; CatBoost handles native categoricals. |
| DL | PyTorch tabular MLP | Benchmarked **against** ML to justify (or refute) its use — see ADR-0004. |
| Explainability | SHAP | Model-agnostic, board-ready feature attributions. |
| External data | `requests` + `tenacity` | Hybrid real+mock with retry/circuit-breaker; offline-safe. |
| UI | Streamlit + Plotly | Fast, interactive, fully themeable to the BMW luxury identity. |
| SQL analytics | DuckDB-over-CSV | Portable, reviewable SQL with no ETL or server — ADR-0007. |
| Experiment tracking | MLflow (file store) | Zero-infrastructure run history — ADR-0009. |
| Docs | MkDocs Material + mkdocstrings | Docs-as-code: ADRs + auto API reference — ADR-0009. |
| Packaging | Docker multi-stage | Small, reproducible runtime image. |
| CI/CD | GitHub Actions | Lint · mypy · pytest (coverage gate) · pip-audit · Docker + Trivy — ADR-0005/0007. |

## Consequences

- **+** Clear separation of concerns; every layer is independently testable.
- **+** Offline-by-default reproducibility (CI never flakes on a third-party API).
- **−** More upfront structure than a notebook; justified by the production bar.

> A critical data-quality finding shapes the *analytical* framing of this project;
> it is recorded separately in **ADR-0002 (Data Integrity)**.

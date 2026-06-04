# Changelog

All notable changes to this project. Format inspired by
[Keep a Changelog](https://keepachangelog.com/); the authoritative history is the
git log.

## [Unreleased]

### Added — senior / top-tier enhancements (2026-06-04)
- **Signal audit** (`bmw_sales.audit`): a reusable, dataset-agnostic No-Signal
  Auditor — permutation (label-shuffle) test, **positive control**, KS-uniformity
  and χ² independence. Proves the null result statistically. (ADR-0006)
- **Decision under uncertainty** (`bmw_sales.simulation.uncertainty`): Monte-Carlo
  propagation of elasticity priors → predictive distribution with **80%/95%
  credible intervals** in the simulator. (ADR-0008)
- **SQL analytics** (`bmw_sales.sql` + `sql/queries/`): DuckDB-over-CSV business
  queries and a **SQL Insights** dashboard tab. (ADR-0007)
- **Experiment tracking**: MLflow logging of every benchmarked model. (ADR-0009)
- **Documentation site**: MkDocs Material (ADRs + auto API reference) deployed to
  **GitHub Pages**. (ADR-0009)
- **Hardened CI**: blocking `mypy`, a coverage gate, `pip-audit`, **Trivy** image
  scan, **Dependabot**, and a `pre-commit` config. (ADR-0007)
- **Docs**: detailed **Mermaid** pipeline diagrams in the README; executive
  storytelling notebook; dashboard screenshots and animated demo.

### Fixed
- **docker**: ship `sql/` (and `.streamlit/`) in the image — fixed a `KeyError`
  on the deployed SQL Insights tab.
- **app**: keep the SHAP feature space consistent with the explained model.
- **ci**: native-`bool` portability for leakage detection across numpy versions;
  declared the `tabulate` runtime dependency; resolvable Trivy action ref.

## [0.1.0] — Initial production deliverable (2026-06-03)

### Added
- **Data integrity report** and the honest-modelling strategy. (ADR-0002)
- **Hybrid external-data augmentation** — World Bank / FX (real) + fuel / CO₂
  (mock-first), with cache, retry and a circuit breaker. (ADR-0003)
- **Econometrics**: hedonic price model, price elasticity (HC3), VIF, and a
  formal target-leakage proof.
- **ML benchmark**: tuned XGBoost / LightGBM / CatBoost, plus a tabular DL model
  benchmarked to justify (and refute) its use. (ADR-0004)
- **Scenario Simulator** (elasticity-based, labelled decision-support).
- **pytest** suite, **premium Streamlit** dashboard, **multi-stage Docker**,
  **GitHub Actions** CI/CD. (ADR-0001, ADR-0005)
- Deployed to **Hugging Face Spaces**.

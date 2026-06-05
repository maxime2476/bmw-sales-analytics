<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

# ◆ BMW LUXURY SALES ANALYTICS ◆

### Production-grade analytics, econometrics & decision intelligence for the BMW luxury-car market

<em>Econometrics · Gradient Boosting · Tabular Deep Learning · External-Data Augmentation · SHAP · Streamlit · Docker · CI/CD</em>

<br/>

![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13-1b1b1d?logo=python&logoColor=D4AF37)
![CI](https://github.com/maxime2476/bmw-sales-analytics/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/maxime2476/bmw-sales-analytics/branch/main/graph/badge.svg)](https://codecov.io/gh/maxime2476/bmw-sales-analytics)
![Lint](https://img.shields.io/badge/black%20·%20isort%20·%20flake8%20·%20mypy-clean-D4AF37)
![Docker](https://img.shields.io/badge/Docker-multi--stage-1b1b1d?logo=docker&logoColor=2496ED)
![GHCR](https://img.shields.io/badge/GHCR-ghcr.io%2Fmaxime2476%2Fbmw--sales--analytics-1b1b1d?logo=github)
![License](https://img.shields.io/badge/license-MIT-D4AF37)

<br/>

### ▶ [**Open the live dashboard**](https://maxime2476-bmw-sales-analytics.hf.space)

[![Open in Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Open%20in%20Spaces-D4AF37?labelColor=1b1b1d)](https://maxime2476-bmw-sales-analytics.hf.space)
[![Live Demo](https://img.shields.io/badge/●%20Live-online-3a7d44?labelColor=1b1b1d)](https://maxime2476-bmw-sales-analytics.hf.space)
[![Docs](https://img.shields.io/badge/📖%20Docs-GitHub%20Pages-1b1b1d)](https://maxime2476.github.io/bmw-sales-analytics/)

</div>

---

## Dashboard preview

<div align="center">

![Animated demo](docs/assets/demo.gif)

<em>Live tour: executive overview → data integrity → econometrics → ML benchmark → scenario simulator.</em>

<br/>

| Executive Overview | Data Integrity |
|:---:|:---:|
| ![Overview](docs/assets/screenshots/overview.png) | ![Data Integrity](docs/assets/screenshots/data_integrity.png) |
| **SQL Insights (DuckDB)** | **Econometrics** |
| ![SQL Insights](docs/assets/screenshots/sql_insights.png) | ![Econometrics](docs/assets/screenshots/econometrics.png) |
| **ML Benchmark** | **Explainability (SHAP)** |
| ![ML Benchmark](docs/assets/screenshots/ml_benchmark.png) | ![SHAP](docs/assets/screenshots/explainability_shap.png) |
| **Scenario Simulator** | **Decision under uncertainty** |
| ![Scenario Simulator](docs/assets/screenshots/scenario_simulator.png) | ![Uncertainty](docs/assets/screenshots/scenario_uncertainty.png) |

<em>DuckDB SQL · interactive Plotly · SHAP explainability · Bayesian-flavoured what-if simulator with credible intervals</em>

</div>

---

## 1. Overview

An end-to-end **decision-support platform** built on 15 years (2010–2024) of BMW
sales records (50,000 transactions, 11 features). It pairs **rigorous
econometrics** with **modern machine learning**, enriches the data with **real
external APIs** (macro-economics, fuel prices, CO₂ regulation, FX), and ships a
**premium Streamlit dashboard** behind a fully containerised, CI/CD-tested
codebase.

> **Data source:** the base dataset is the public
> [BMW Sales Dataset on Kaggle](https://www.kaggle.com/datasets/eshummalik/bmw-sales-dataset)
> by *eshummalik*. All external macro/fuel/CO₂/FX context is added by this
> project (see [ADR-0003](docs/adr/0003-api-augmentation.md)).

> ### This project proves two things
>
> **1 — I can build a model that works.** The pipeline reaches a **cross-validated
> R² ≈ 0.85** on signal-bearing data, with SHAP recovering the true drivers — a
> *validated* model, not a lucky split.
> ([predictive capability](reports/predictive_capability.md))
>
> **2 — I won't fake it when the data is empty.** This particular dataset is
> **structurally pristine but signal-free** (every feature is statistically
> independent of the targets), and `Sales_Classification` is a **leaked**
> threshold on `Sales_Volume`. On it the *same* pipeline honestly scores **R² ≈ 0
> / AUC ≈ 0.5** — proven with a permutation test and a positive control, not
> hidden. Business value is then delivered through a clearly-labelled **Scenario
> Simulator**.
>
> *Predictive competence **and** intellectual honesty — that is the senior
> deliverable.* Evidence: [Predictive Capability](reports/predictive_capability.md) ·
> [Data Integrity](reports/data_integrity_report.md) ·
> [Signal Audit](reports/signal_audit.md) · [ADR-0002](docs/adr/0002-data-integrity.md).

---

## 2. Headline results (honest, reproducible)

| Analysis | Result | What it means |
|---|---|---|
| Max \|correlation\| among numeric features | **0.009** | Features are mutually independent noise |
| Price elasticity of demand (log-log, HC3) | **−0.001** (p = 0.92) | No measurable price sensitivity in-sample |
| Hedonic price model R² | **0.0004** | Price is unexplained by attributes here |
| Regression R² (best of XGB/LGBM/CatBoost) | **≈ 0.00** | Boosting cannot beat the mean — no signal |
| Classification ROC-AUC (leakage-free) | **≈ 0.51** | No discriminative signal once leakage removed |
| Classification ROC-AUC (leak left in) | **1.00** | The signature of target leakage |
| **Permutation test** (label-shuffle) | **p ≈ 0.90** | Real score indistinguishable from chance |
| **Predictive capability** (same pipeline, signal-bearing target) | **CV R² ≈ 0.85 ± 0.003** | The pipeline *does* predict — when there is signal |
| Tabular MLP vs gradient boosting | both no-skill | Deep learning **not justified** (ADR-0004) |

Reports: [econometrics](reports/econometric_analysis.md) ·
[model benchmark](reports/model_benchmark.md) ·
[DL vs ML](reports/dl_vs_ml.md).

---

## 3. Architecture

```
bmw-sales/
├── src/bmw_sales/
│   ├── config.py            # typed pydantic-settings + canonical DatasetSchema
│   ├── data/                # loader (schema validation) · validation (integrity report)
│   ├── audit/               # No-Signal Auditor: permutation · positive control · KS · χ²
│   ├── apis/                # hybrid real+mock clients · enrichment join
│   │   ├── base.py          #   cache + retry + circuit breaker + provenance
│   │   ├── worldbank.py · fx_rates.py · fuel_prices.py · co2_regulations.py
│   ├── features/            # domain feature engineering
│   ├── econometrics/        # OLS hedonic · demand · elasticity · VIF · leakage proof
│   ├── models/              # preprocessing · XGB/LGBM/CatBoost · tabular MLP · MLflow
│   ├── simulation/          # Scenario Simulator + Monte-Carlo uncertainty
│   ├── explainability/      # SHAP attributions
│   └── sql/                 # DuckDB analytics over sql/queries/*.sql
├── app/                     # Streamlit premium UI (theme · data_access · 7 tabs)
├── sql/queries/             # versioned analytical SQL
├── tests/                   # pytest suite (unit + integration)
├── docs/                    # MkDocs Material site + 9 ADRs
├── reports/                 # generated analyses (committed)
├── Dockerfile · docker-compose.yml · .github/workflows/{main,docs}.yml
└── Makefile · mkdocs.yml · pyproject.toml · requirements*.txt
```

Design rationale: [ADR-0001](docs/adr/0001-architecture-and-stack.md).

## Pipeline (end-to-end)

The full flow from raw data to a deployed decision-support app. The
**honest-analytics spine** (gold) is what makes this a senior deliverable: the
data is audited and proven signal-free *before* any model is trusted.

```mermaid
flowchart TB
    RAW["Raw dataset<br/>BMW_sales_data 2010–2024<br/>50,000 rows × 11 cols"]

    subgraph L1["① Data foundation · bmw_sales.data"]
        LOAD["loader.py<br/>schema validation · dtype coercion"]
        VAL["validation.py<br/>correlation · ANOVA · mutual-info · leakage"]
    end

    subgraph L2["② Signal audit · bmw_sales.audit"]
        PERM["permutation / label-shuffle test<br/>p ≈ 0.90 → no signal"]
        CTRL["positive control<br/>synthetic R² ≈ 0.86 vs real ≈ 0"]
        KS["KS-uniformity · χ² independence"]
    end

    subgraph L3["③ External augmentation · bmw_sales.apis"]
        WB["WorldBank · FX<br/>real endpoints"]
        FC["Fuel · CO₂<br/>mock-first"]
        BASE["base.py<br/>cache → retry → circuit-breaker → mock"]
        ENR["enrichment.py<br/>region × year × fuel panel join"]
    end

    subgraph L4["④ Features · bmw_sales.features"]
        FE["engineering.py<br/>age · usage · electrified · log transforms"]
    end

    subgraph L5["⑤ Modelling"]
        ECON["econometrics<br/>hedonic OLS · elasticity (HC3) · leakage proof"]
        ML["ml_models<br/>XGBoost · LightGBM · CatBoost + RandomizedSearchCV"]
        DL["dl_models<br/>PyTorch tabular MLP (early stopping)"]
    end

    subgraph L6["⑥ Decision intelligence"]
        SIM["simulation<br/>elasticity scenario + Monte-Carlo CIs"]
        SHAP["explainability<br/>SHAP attributions"]
        SQL["sql · DuckDB<br/>region · price · YoY · electrification"]
    end

    REPORTS[("reports/<br/>integrity · signal_audit · econometric<br/>model_benchmark · dl_vs_ml · sql_insights")]
    MLF[("MLflow<br/>./mlruns")]
    ART[("models/*.joblib")]

    APP["Streamlit app · 7 tabs<br/>Overview · Integrity · SQL · Econometrics<br/>ML · SHAP · Scenario Simulator"]

    RAW --> LOAD --> VAL
    RAW --> L2
    RAW --> SQL
    LOAD --> ENR
    WB & FC --> BASE --> ENR
    VAL --> FE
    ENR --> FE
    FE --> ECON & ML & DL
    ML --> ART
    ML --> SHAP
    ENR -. macro baselines .-> SIM
    L2 --> REPORTS
    ECON & ML & DL & SQL --> REPORTS
    ML --> MLF
    REPORTS --> APP
    SIM & SHAP & SQL --> APP

    classDef honest fill:#241f08,stroke:#D4AF37,stroke-width:2px,color:#fff;
    classDef store fill:#15151a,stroke:#8FA9C7,color:#cfe;
    class RAW,L1,L2,FE honest;
    class REPORTS,MLF,ART store;
```

### Hybrid-API resilience (offline-safe by design)

Every external client degrades gracefully, so CI/Docker run with no network or
keys yet the real path is proven live.

```mermaid
flowchart LR
    REQ["client.fetch(region, years)"] --> C{"disk cache hit?"}
    C -- yes --> HIT["return cached<br/>provenance = cache"]
    C -- no --> OFF{"offline mode<br/>or breaker open?"}
    OFF -- yes --> MOCK["deterministic mock<br/>provenance = mock"]
    OFF -- no --> LIVE["HTTP GET + retry/backoff"]
    LIVE -- success --> SAVE["cache + return<br/>provenance = live"]
    LIVE -- failure --> TRIP["trip circuit-breaker"] --> MOCK
```

### Delivery — tests, CI/CD & deployment

```mermaid
flowchart LR
    DEV["commit on feature/* branch"] --> PC["pre-commit<br/>black · isort · flake8 · mypy"]
    PC --> PUSH["push → main"]
    PUSH --> CI{"GitHub Actions"}
    CI --> Q["quality (3.11 / 3.12)<br/>black · isort · flake8 · mypy<br/>pytest + 62% coverage gate"]
    CI --> SEC["pip-audit"]
    Q --> DK["Docker build + Trivy scan"]
    PUSH --> DOCS["MkDocs build"]
    DK -. image .-> HF["HF Spaces (Docker)<br/>live app"]
    DOCS --> GP["GitHub Pages<br/>docs site"]
```

---

## 4. External-data augmentation (hybrid: real + mock)

Four sources mapped to the six regions via **official World Bank aggregate codes**
(EAS, NAC, MEA, LCN, EMU, SSF) and representative currencies/countries. Every
client **caches** responses, **retries** with backoff, and trips a **circuit
breaker** to a deterministic **mock** on failure — so the project runs fully
offline yet **three of the four sources are validated live** against real APIs.

| Source | Status | Real endpoint | Signal it adds |
|---|---|---|---|
| World Bank macro | 🟢 **real** | inflation `FP.CPI.TOTL.ZG`, GDP/cap `NY.GDP.PCAP.CD` | regional purchasing power |
| FX rates | 🟢 **real** | exchangerate.host | local-currency price normalisation |
| CO₂ emissions | 🟢 **real** | World Bank CO₂/capita `EN.GHG.CO2.PC.CE.AR5` | the electrification transition |
| Fuel prices | 🟡 mock-first | WB pump-price `EP.PMP.SGAS.CD` **archived by WB (2024)** | Petrol/Diesel vs electrified economics |

> *Honesty applies to the data layer too:* fuel stays mock-first because the World
> Bank archived its pump-price series — the real hook is kept and the provenance is
> reported as `mock` rather than faking it.

Details: [ADR-0003](docs/adr/0003-api-augmentation.md).

---

## 5. The Scenario Simulator (where business value lives)

Because the data cannot forecast, decision value comes from an **explicit
what-if simulation** — a constant-elasticity demand model with
literature-grounded priors (own-price ε ≈ −0.6, income ε ≈ +1.5, fuel
cross-elasticity, CO₂-regulation shift) and **baselines seeded from the real
macro APIs**. Every driver's contribution is decomposed in a waterfall chart, and
all assumptions are adjustable in the UI. It is never presented as a fit to the
historical data.

---

## 6. Quickstart

```bash
# Install (dev includes linting, tests, torch for the DL benchmark)
make install-dev                 # or: pip install -r requirements-dev.txt

make eda                         # regenerate the Data Integrity Report
make pipeline                    # train & benchmark all models (writes reports/)
make test                        # full suite, offline & deterministic
make app                         # launch the dashboard → http://localhost:8501
```

### Docker

```bash
docker compose up --build        # → http://localhost:8501
```

Or pull the **published image** from the GitHub Container Registry (built, scanned
and pushed by CI on every `main` update):

```bash
docker run -p 8501:8501 ghcr.io/maxime2476/bmw-sales-analytics:latest
```

**Managed deployment** (Streamlit Community Cloud or Hugging Face Spaces):
see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

> On Windows + Anaconda, `KMP_DUPLICATE_LIB_OK=TRUE` is set in-code to avoid the
> known OpenMP (`libiomp5md.dll`) clash when importing PyTorch.

---

## 7. Quality & engineering

- **Typed** (PEP 484) and **`mypy`-clean** across the `src/` package.
- **Formatted & linted:** `black`, `isort`, `flake8` — all clean; `pre-commit`
  hooks run the same gates locally.
- **Tested:** a `pytest` suite behind a **coverage gate of ≥ 62%** (live status in
  the CI and Codecov badges above) — schema, leakage, mock determinism &
  circuit-breaker fallback, leakage-aware splits, signal audit, predictive
  capability, Monte-Carlo simulator, SQL layer, report builders; real-data checks
  marked `integration`. A guard test keeps this gate in sync between the README
  and CI.
- **Security:** `pip-audit` dependency scan, **Trivy** image scan, and
  **Dependabot** updates (pip · actions · docker).
- **SQL analytics:** decision queries in `sql/queries/` executed by **DuckDB**
  directly over the CSV (window functions, quantiles, YoY) — `make sql`.
- **Experiment tracking:** every benchmarked model is logged to **MLflow**
  (`mlflow ui --backend-store-uri ./mlruns`).
- **Docs site:** **MkDocs Material** (ADRs + auto API reference) auto-deployed to
  **[GitHub Pages](https://maxime2476.github.io/bmw-sales-analytics/)**.
- **CI/CD:** GitHub Actions — lint + type + test matrix (3.11/3.12) with a
  coverage gate → cached Docker build + Trivy scan. See
  [ADR-0005](docs/adr/0005-devops-and-cicd.md), [ADR-0007](docs/adr/0007-sql-and-quality-gates.md).

---

## 8. Business insights for decision-makers

1. **This dataset cannot price or forecast.** Any model claiming high accuracy on
   it is either leaking the target or overfitting noise — a useful red-flag
   heuristic for reviewing vendor models.
2. **Pricing & go-to-market must lean on external signals** (regional income,
   fuel economics, CO₂ regulation) — exactly what the Simulator operationalises.
3. **The electrification transition is the real story:** regulation stringency,
   not historical volume, should drive the Petrol→Electric portfolio mix.

---

## 9. Architecture Decision Records

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-architecture-and-stack.md) | Architecture & stack |
| [0002](docs/adr/0002-data-integrity.md) | Data-integrity finding & honest-modelling strategy |
| [0003](docs/adr/0003-api-augmentation.md) | Hybrid external-data augmentation |
| [0004](docs/adr/0004-deep-learning-justification.md) | DL tested, not assumed |
| [0005](docs/adr/0005-devops-and-cicd.md) | Containerisation & CI/CD |
| [0006](docs/adr/0006-signal-audit.md) | Statistical signal audit & positive control |
| [0007](docs/adr/0007-sql-and-quality-gates.md) | SQL analytics & hardened quality gates |
| [0008](docs/adr/0008-decision-under-uncertainty.md) | Decision-making under uncertainty (Monte-Carlo) |
| [0009](docs/adr/0009-observability-and-docs.md) | Experiment tracking & published docs site |

Project evolution is summarised in the **[CHANGELOG](CHANGELOG.md)**.

---

## 10. Author

**Maxime GOURGUECHON** — maxime.gourguechon76@gmail.com

## License

[MIT](LICENSE)

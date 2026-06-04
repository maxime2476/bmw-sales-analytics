<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

# ◆ BMW LUXURY SALES ANALYTICS ◆
### Production-grade analytics, econometrics & forecasting for the BMW luxury-car market

<em>Econometrics · Gradient Boosting · Tabular Deep Learning · External-Data Augmentation · SHAP · Streamlit · Docker · CI/CD</em>

</div>

---

> **Status:** 🚧 Under active development — see [project board](#-roadmap) below.

## Overview

An end-to-end decision-support platform built on 15 years (2010–2024) of BMW
sales records (50,000 transactions). The project pairs **rigorous econometrics**
with **modern machine learning**, enriches the data with **real external APIs**
(macro-economics, fuel prices, CO₂ regulation, FX), and ships a **premium
Streamlit dashboard** behind a fully containerised, CI/CD-tested codebase.

### A note on intellectual honesty
A core principle of this project is methodological transparency. The exploratory
analysis surfaced an important property of the source data, documented in the
**[Data Integrity Report](docs/adr/0002-data-integrity.md)**. Rather than inflate
metrics, we report honest results and deliver business value through a clearly
labelled **Scenario Simulator** grounded in econometric elasticities and live
macro data. This is what separates a senior deliverable from a demo.

## Architecture

See **[ADR-0001](docs/adr/0001-architecture-and-stack.md)** for the full rationale.

```
src/bmw_sales/   data · apis · features · econometrics · models · simulation · explainability
app/             Streamlit premium UI (dark + champagne-gold)
tests/           pytest unit + integration
docs/adr/        architecture decision records
```

## Quickstart

```bash
# 1. Install (dev includes linting, tests, DL)
make install-dev          # or: pip install -r requirements-dev.txt

# 2. Generate the Data Integrity Report (EDA)
make eda

# 3. Run the test suite
make test

# 4. Launch the dashboard
make app                  # → http://localhost:8501
```

Or with Docker:

```bash
make docker-up            # docker compose up --build
```

## Roadmap

- [x] Project foundation, typed config, ADR-0001
- [ ] Phase 1 — Data validation & external API augmentation
- [ ] Phase 2 — Econometrics, ML, DL, Scenario Simulator
- [ ] Phase 3 — Test suite
- [ ] Phase 4 — Streamlit premium UI + SHAP explainability
- [ ] Phase 5 — Docker & GitHub Actions CI/CD

## Author

**Maxime GOURGUECHON** — maxime.gourguechon76@gmail.com

## License

MIT

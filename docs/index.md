# ◆ BMW Luxury Sales Analytics

Production-grade analytics, econometrics & decision intelligence for the BMW
luxury-car market — honest ML/DL, external-API augmentation, SHAP, a
Bayesian-flavoured Scenario Simulator, a DuckDB SQL layer, and a fully
containerised, CI/CD-tested codebase.

[:material-rocket-launch: **Open the live dashboard**](https://maxime2476-bmw-sales-analytics.hf.space){ .md-button .md-button--primary }
[:material-github: **Source on GitHub**](https://github.com/maxime2476/bmw-sales-analytics){ .md-button }

!!! info "Data source"
    Base dataset: [BMW Sales Dataset on Kaggle](https://www.kaggle.com/datasets/eshummalik/bmw-sales-dataset)
    by *eshummalik*. All macro/fuel/CO₂/FX context is added by this project's
    augmentation layer.

## The defining principle — intellectual honesty

The dataset is **structurally pristine but signal-free**, and
`Sales_Classification` is a **leaked** deterministic threshold on `Sales_Volume`.
Rather than inflate metrics, this project **detects, proves and communicates** the
issue — then delivers business value through a clearly labelled Scenario Simulator.

!!! quote "Headline, reproducible results"
    | Check | Result |
    |---|---|
    | Permutation (label-shuffle) test | p ≈ 0.90 — indistinguishable from chance |
    | Positive control (same pipeline, synthetic target) | R² ≈ 0.86 — pipeline is sound |
    | Regression R² / leakage-free ROC-AUC | ≈ 0.00 / ≈ 0.51 |
    | Leaked classification ROC-AUC | 1.00 🚩 |

## How the docs are organised

- **Architecture Decisions** — the nine ADRs that record every significant choice
  (architecture, the data-integrity finding, augmentation, DL justification,
  DevOps, the signal audit, SQL & quality gates, decision-under-uncertainty,
  and observability & docs).
- **API Reference** — auto-generated from the `bmw_sales` package docstrings.

## Quickstart

```bash
make install-dev     # dependencies
make audit           # statistical signal audit (permutation + positive control)
make sql             # DuckDB business-insights report
make test            # full suite + coverage gate
make app             # launch the dashboard
```

---

*Built by **Maxime GOURGUECHON**.*

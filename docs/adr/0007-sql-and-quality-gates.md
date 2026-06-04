# ADR-0007 — SQL analytics layer & hardened quality gates

- **Status:** Accepted
- **Date:** 2026-06-04
- **Author:** Maxime GOURGUECHON
- **Related:** ADR-0001, ADR-0005

## Context

Two gaps remained on the path from "very good" to "production-grade": the brief
asked for **SQL** (under-used so far), and the CI enforced style/tests but not
**types, coverage or security**.

## Decision

### SQL analytics with DuckDB
Add `bmw_sales.sql`: versioned, reviewable `.sql` files in `sql/queries/`
executed by **DuckDB** straight over the raw CSV (no server, no ETL). Queries use
window functions, `quantile_cont`, and `LAG` for YoY. Surfaced via `make sql`, a
published `reports/sql_insights.md`, and a **SQL Insights** dashboard tab.
DuckDB is a lightweight runtime dependency; the trusted dataset path is inlined
with quote-escaping (DuckDB cannot bind parameters in `CREATE VIEW`).

### Hardened CI/CD
- **`mypy`** type-checking is now a blocking gate (the codebase is mypy-clean).
- **Coverage gate** `--cov-fail-under=62` (current ~67%); report builders gained
  smoke tests to raise meaningful coverage.
- **Security:** a `pip-audit` job and a **Trivy** image scan (report-only, so a
  newly-disclosed upstream CVE never red-bars a feature PR), plus **Dependabot**
  for pip, GitHub Actions and Docker.
- **`pre-commit`** mirrors the CI gates locally (black, isort, flake8, mypy,
  hygiene hooks).

## Rationale

- DuckDB-over-CSV keeps business logic in portable SQL while avoiding database
  infrastructure — ideal for an analytical, reproducible project.
- Security scans are **report-only** by design: supply-chain noise should inform,
  not block, unrelated changes; criticality is reviewed, not auto-failed.

## Consequences

- **+** Types, coverage and supply-chain are now first-class, enforced signals.
- **+** SQL competence is demonstrated end-to-end (files → report → live tab).
- **−** Slightly longer CI (mypy + scans); acceptable for the added assurance.

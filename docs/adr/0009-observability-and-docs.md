# ADR-0009 — Experiment tracking & a published documentation site

- **Status:** Accepted
- **Date:** 2026-06-04
- **Author:** Maxime GOURGUECHON
- **Related:** ADR-0005, ADR-0007

## Context

Two production concerns were unaddressed: model runs were not **tracked** (no
record of params/metrics across experiments), and the project's rich
documentation (ADRs, module docstrings) lived only in the repo, not as a
browsable site.

## Decision

### Experiment tracking — MLflow (best-effort)
`bmw_sales.models.tracking.log_models` logs every benchmarked model (params,
metrics, tags) to a local file-based MLflow store (`./mlruns`) from the training
pipeline. Tracking is **best-effort**: MLflow is a dev-only dependency, and the
helper no-ops cleanly when it is absent, so `make pipeline` never fails because
of an optional tool. Browse with `mlflow ui --backend-store-uri ./mlruns`.

### Documentation — MkDocs Material on GitHub Pages
A **MkDocs Material** site (`mkdocs.yml`) publishes the seven+ ADRs and an
**auto-generated API reference** (mkdocstrings reads the package docstrings). A
dedicated GitHub Actions workflow (`docs.yml`) builds and deploys it to **GitHub
Pages** on every push to `main`, so the docs are always current.

## Rationale

- **MLflow file store, not a server.** Matches the project's zero-infrastructure
  ethos (cf. DuckDB-over-CSV in ADR-0007); a tracking server would be overkill.
- **Best-effort logging** keeps the runtime image slim (MLflow stays in
  `requirements-dev.txt`) while still demonstrating MLOps maturity.
- **Docs-as-code.** Auto-generating the API reference from docstrings means the
  site cannot drift from the code, and the ADRs become first-class, linkable
  artefacts.

## Consequences

- **+** Reproducible experiment history; a live, always-current docs site.
- **+** No new runtime dependencies (both tools are dev/CI-only).
- **−** The docs deploy needs GitHub Pages enabled with "GitHub Actions" as the
  source (one-time repo setting).

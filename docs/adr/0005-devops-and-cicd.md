# ADR-0005 — Containerisation & CI/CD

- **Status:** Accepted
- **Date:** 2026-06-03
- **Author:** Maxime GOURGUECHON
- **Related:** ADR-0001, ADR-0003

## Context

The deliverable must be reproducible and shippable: a one-command local run, a
slim production image, and automated quality gates.

## Decision

### Docker — multi-stage, slim, non-root
- **Builder stage** creates an isolated `/opt/venv` and installs **runtime**
  dependencies only (`requirements.txt`) — the heavy DL/dev toolchain (`torch`,
  linters) is excluded, keeping the image small.
- **Runtime stage** (`python:3.12-slim`) copies the venv, installs `libgomp1`
  (required by XGBoost/LightGBM), runs as an unprivileged `appuser`, sets
  `BMW_OFFLINE_MODE=true` for deterministic startup, and exposes a Streamlit
  health-checked service on `8501`.
- `.dockerignore` keeps the build context lean (no `.git`, tests, notebooks,
  caches, model artefacts).

### Compose
`docker-compose.yml` provides a single `dashboard` service with port mapping,
healthcheck and `restart: unless-stopped` for friction-free local runs.

### GitHub Actions — `quality` then `docker`
- **`quality`** matrix over Python 3.11/3.12: `black --check`, `isort --check`,
  `flake8`, then `pytest` with coverage. Runs with `BMW_OFFLINE_MODE=true` so it
  never depends on a third-party API (the hybrid clients fall back to mocks).
- **`docker`** builds the image with Buildx + GHA layer cache (no push) to prove
  the container builds on every change.
- `concurrency` cancels superseded runs to save minutes.

## Rationale

- **3.12 base, not 3.13:** broadest wheel availability for the scientific stack
  in containers; the code itself supports 3.11–3.13.
- **Runtime-only image:** the dashboard needs neither `torch` nor linters, so
  they stay out of production.
- **Offline CI:** deterministic and fast; live API behaviour is covered by the
  client design and exercised manually, not in CI.

## Consequences

- **+** Reproducible, small, secure-by-default image; green-by-design CI.
- **−** The DL benchmark is not re-run in CI (heavy); its results are committed
  as reports and reproducible locally via `make`.

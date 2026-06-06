# Contributing

Thanks for your interest in **BMW Luxury Sales Analytics**. This is primarily a
portfolio project, but contributions, issues and suggestions are welcome.

## Development setup

```bash
git clone https://github.com/maxime2476/bmw-sales-analytics
cd bmw-sales-analytics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install-dev
pre-commit install        # run the same quality gates locally on every commit
```

## Quality gates (must pass)

The CI mirrors these; run them before opening a PR:

```bash
make format     # black + isort
make lint       # flake8
make typecheck  # mypy (the src/ package must stay mypy-clean)
make test       # pytest behind a coverage gate (>= 62%)
```

`pre-commit` runs black, isort, flake8, mypy and hygiene hooks automatically.

## Conventions

- **Commits:** semantic and atomic — `feat(scope): …`, `fix(scope): …`,
  `docs: …`, `test: …`, `ci: …`, `chore: …`.
- **Branches:** `feature/*`, `fix/*`, `docs/*`, `chore/*`; open a PR into `main`.
- **Style:** typed (PEP 484), `black` line length 100, docstrings on public APIs.
- **Tests:** add/extend tests for any behaviour change; keep them offline and
  deterministic (mark real-data/integration tests with `@pytest.mark.integration`).
- **Decisions:** non-trivial design choices get an ADR under `docs/adr/` — see
  the [wiki](https://github.com/maxime2476/bmw-sales-analytics/wiki/16-Architecture-Decision-Records).

## Project principles

This project values **intellectual honesty** above impressive-looking metrics.
Please keep that spirit: report honest results, never leak targets, and label
synthetic/simulated artefacts clearly.

## Reporting issues

Use the issue templates (bug report / feature request). For security matters, see
[`SECURITY.md`](SECURITY.md).

# ============================================================================
# BMW Sales Analytics — developer entrypoints
# Usage: `make help`
# ============================================================================
.DEFAULT_GOAL := help
PY := python

.PHONY: help install install-dev format lint typecheck test test-fast \
        eda econ pipeline dl-report reports app docker-build docker-up clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev: ## Install dev + runtime dependencies
	$(PY) -m pip install -r requirements-dev.txt

format: ## Auto-format (black + isort)
	black src app tests
	isort src app tests

lint: ## Static lint (flake8)
	flake8 src app tests

typecheck: ## Static type check (mypy)
	mypy src

test: ## Run full test suite with coverage
	pytest --cov=bmw_sales --cov-report=term-missing

test-fast: ## Run tests, skip integration (no network)
	pytest -m "not integration"

eda: ## Generate the Data Integrity Report
	$(PY) -m bmw_sales.data.validation

econ: ## Generate the econometric analysis report
	$(PY) -m bmw_sales.econometrics.report

pipeline: ## Train & benchmark all ML models (writes reports + artefacts)
	$(PY) -m bmw_sales.models.train

dl-report: ## Benchmark the tabular DL model vs gradient boosting
	$(PY) -m bmw_sales.models.dl_report

reports: eda econ pipeline dl-report ## Regenerate every analysis report

app: ## Launch the Streamlit dashboard
	streamlit run app/streamlit_app.py

docker-build: ## Build the production image (multi-stage)
	docker build -t bmw-sales-analytics:latest .

docker-up: ## Start via docker-compose
	docker compose up --build

clean: ## Remove caches & build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

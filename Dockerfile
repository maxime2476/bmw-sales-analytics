# syntax=docker/dockerfile:1
# =============================================================================
# BMW Luxury Sales Analytics — multi-stage production image
# Stage 1 builds an isolated virtualenv; Stage 2 ships only the runtime.
# =============================================================================

# ---- Stage 1: builder -------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Create a self-contained virtualenv we can copy wholesale into the runtime.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only (no dev/DL toolchain) for a slim image.
# Post-install: drop XGBoost's optional CUDA/NCCL libs (~400 MB; we run CPU-only)
# and strip bundled bytecode caches & test suites to keep the layer lean.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip uninstall -y nvidia-nccl-cu12 || true \
    && find /opt/venv -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/venv -depth -type d -name 'tests' -exec rm -rf {} + \
    && find /opt/venv -type f -name '*.pyc' -delete

# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim AS runtime

# OCI metadata — links the GHCR package to the repository.
LABEL org.opencontainers.image.source="https://github.com/maxime2476/bmw-sales-analytics" \
      org.opencontainers.image.description="BMW Luxury Sales Analytics — Streamlit dashboard" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Maxime GOURGUECHON"

# libgomp1 is required at runtime by xgboost/lightgbm.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    BMW_OFFLINE_MODE=true \
    KMP_DUPLICATE_LIB_OK=TRUE \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ ./src/
COPY app/ ./app/
COPY sql/ ./sql/
COPY .streamlit/ ./.streamlit/
COPY data/raw/ ./data/raw/
COPY reports/ ./reports/
COPY pyproject.toml README.md ./

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

"""Cached data & model access for the Streamlit app.

Centralises expensive operations behind ``st.cache_*`` so the UI stays snappy.
Training falls back to a quick, untuned fit on a sample when no persisted model
artefact is present (e.g. a fresh clone), so the app always works out of the box.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import streamlit as st

# Make the src/ package importable when running without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Reproducible, offline-by-default external data for the app.
os.environ.setdefault("BMW_OFFLINE_MODE", "true")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from bmw_sales.apis.enrichment import enrich_dataset  # noqa: E402
from bmw_sales.config import MODELS_DIR  # noqa: E402
from bmw_sales.data.loader import load_raw  # noqa: E402
from bmw_sales.data.validation import DataIntegrityReport, analyse  # noqa: E402
from bmw_sales.models.ml_models import TrainedModel, train_one  # noqa: E402
from bmw_sales.models.preprocessing import Dataset, make_dataset  # noqa: E402


@st.cache_data(show_spinner=False)
def get_raw() -> pd.DataFrame:
    return load_raw()


@st.cache_data(show_spinner="Enriching with external macro/fuel/CO₂/FX data…")
def get_enriched() -> tuple[pd.DataFrame, dict[str, str]]:
    res = enrich_dataset(load_raw())
    return res.data, res.provenance


@st.cache_data(show_spinner="Running data-integrity analysis…")
def get_integrity() -> DataIntegrityReport:
    return analyse(load_raw())


@st.cache_data(show_spinner="Running statistical signal audit (permutation + control)…")
def get_signal_audit():
    """Positive-control R² and a regression permutation test (cached)."""
    from bmw_sales.audit.control import run_control
    from bmw_sales.audit.signal_tests import permutation_test

    df = load_raw()
    control = run_control(df, sample=10000)
    perm = permutation_test(df, "regression", n_permutations=20, sample=5000)
    return control, perm


@st.cache_resource(show_spinner="Preparing model & SHAP explainer…")
def get_regression_model(_sample: int = 12000) -> tuple[TrainedModel, Dataset]:
    """Load the persisted regression pipeline, or train a quick one on a sample."""
    df = load_raw()
    ds = make_dataset(df.sample(_sample, random_state=42), "regression")
    artefact: Optional[Path] = MODELS_DIR / "regression_best.joblib"
    model = train_one("XGBoost", ds, tune=False)
    if artefact.exists():
        try:
            model.pipeline = joblib.load(artefact)
        except Exception:  # noqa: BLE001 — fall back to the freshly trained pipeline
            pass
    return model, ds

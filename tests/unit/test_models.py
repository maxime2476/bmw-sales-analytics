"""Tests for preprocessing, leakage-aware splits and a fast model fit."""

from __future__ import annotations

import pandas as pd

from bmw_sales.config import SCHEMA
from bmw_sales.models.evaluate import classification_metrics, regression_metrics
from bmw_sales.models.ml_models import train_one
from bmw_sales.models.preprocessing import make_dataset


def test_classification_excludes_leaked_volume(sample_df: pd.DataFrame) -> None:
    ds = make_dataset(sample_df, "classification")
    assert SCHEMA.SALES_VOLUME not in ds.numeric
    assert SCHEMA.SALES_CLASSIFICATION not in ds.X_train.columns


def test_leakage_flag_includes_volume(sample_df: pd.DataFrame) -> None:
    ds = make_dataset(sample_df, "classification", include_leakage=True)
    assert SCHEMA.SALES_VOLUME in ds.numeric


def test_regression_target_not_in_features(sample_df: pd.DataFrame) -> None:
    ds = make_dataset(sample_df, "regression")
    assert SCHEMA.SALES_VOLUME not in ds.X_train.columns
    assert SCHEMA.SALES_CLASSIFICATION not in ds.X_train.columns


def test_split_sizes_sum_to_total(sample_df: pd.DataFrame) -> None:
    ds = make_dataset(sample_df, "regression")
    total = len(ds.X_train) + len(ds.X_val) + len(ds.X_test)
    assert total == len(sample_df)


def test_leaked_classifier_is_near_perfect(sample_df: pd.DataFrame) -> None:
    """Including Sales_Volume must recover the label almost perfectly (leakage)."""
    ds = make_dataset(sample_df, "classification", include_leakage=True)
    model = train_one("LightGBM", ds, tune=False)
    assert model.metrics["roc_auc"] > 0.95


def test_metric_helpers_basic() -> None:
    rm = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert rm.r2 == 1.0 and rm.rmse == 0.0
    cm = classification_metrics([0, 1, 1], [0, 1, 1], [0.1, 0.9, 0.8])
    assert cm.roc_auc == 1.0 and cm.accuracy == 1.0

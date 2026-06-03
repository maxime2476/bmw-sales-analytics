"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bmw_sales.config import SCHEMA
from bmw_sales.features.engineering import (
    ENGINEERED_NUMERIC,
    add_engineered_features,
    feature_columns,
)


def test_adds_expected_columns(sample_df: pd.DataFrame) -> None:
    out = add_engineered_features(sample_df)
    for col in ENGINEERED_NUMERIC:
        assert col in out.columns
    assert "is_electrified" in out.columns
    assert "is_premium_tier" in out.columns


def test_vehicle_age_non_negative(sample_df: pd.DataFrame) -> None:
    out = add_engineered_features(sample_df, reference_year=2024)
    assert (out["vehicle_age"] >= 0).all()


def test_log_price_matches(sample_df: pd.DataFrame) -> None:
    out = add_engineered_features(sample_df)
    assert np.allclose(out["log_price"], np.log(sample_df[SCHEMA.PRICE_USD].astype(float)))


def test_electrified_flag(sample_df: pd.DataFrame) -> None:
    out = add_engineered_features(sample_df)
    mask = sample_df[SCHEMA.FUEL_TYPE].isin(["Hybrid", "Electric"])
    assert (out.loc[mask, "is_electrified"] == 1).all()
    assert (out.loc[~mask, "is_electrified"] == 0).all()


def test_feature_columns_leakage_toggle() -> None:
    without = feature_columns(include_leakage=False)
    with_leak = feature_columns(include_leakage=True)
    assert SCHEMA.SALES_VOLUME not in without["numeric"]
    assert SCHEMA.SALES_VOLUME in with_leak["numeric"]

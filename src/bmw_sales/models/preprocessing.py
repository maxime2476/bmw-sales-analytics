"""Shared preprocessing for the supervised ML/DL pipelines.

Provides a single, leakage-aware way to turn the (optionally enriched) dataset
into model-ready ``X``/``y`` plus a fitted-on-train ``ColumnTransformer``. Using
the same preprocessing for every model keeps the benchmark fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bmw_sales.config import SCHEMA, get_settings
from bmw_sales.features.engineering import add_engineered_features, feature_columns

Task = Literal["regression", "classification"]

#: External (API-enriched) numeric columns to use as features when present.
ENRICHED_NUMERIC: tuple[str, ...] = (
    "inflation_pct",
    "gdp_per_capita_usd",
    "regulation_stringency_index",
    "fleet_co2_target_g_km",
    "price_usd_per_litre",
    "usd_per_unit",
)


@dataclass
class Dataset:
    """A train/validation/test split plus column metadata."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    numeric: list[str]
    categorical: list[str]
    task: Task

    @property
    def n_features(self) -> int:
        return len(self.numeric) + len(self.categorical)


def select_features(
    df: pd.DataFrame, *, include_leakage: bool = False
) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names present in ``df``.

    Adds API-enriched numeric columns when available. Never includes a target.
    """
    cols = feature_columns(include_leakage=include_leakage)
    numeric = [c for c in cols["numeric"] if c in df.columns]
    numeric += [c for c in ENRICHED_NUMERIC if c in df.columns and df[c].notna().any()]
    categorical = [c for c in cols["categorical"] if c in df.columns]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Standardise numerics and one-hot encode categoricals (dense, unknown-safe)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_dataset(
    df: pd.DataFrame,
    task: Task,
    *,
    include_leakage: bool = False,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Dataset:
    """Build a leakage-aware train/val/test split for the requested task.

    Parameters
    ----------
    df:
        Raw or enriched dataset.
    task:
        ``"regression"`` (target ``Sales_Volume``) or ``"classification"``
        (target ``Sales_Classification``).
    include_leakage:
        Classification only — include ``Sales_Volume`` to demonstrate leakage.
    """
    seed = get_settings().random_seed
    data = add_engineered_features(df)

    if task == "regression":
        target = SCHEMA.SALES_VOLUME
        # Drop the leaked label so it cannot inform the regression target.
        data = data.drop(columns=[SCHEMA.SALES_CLASSIFICATION], errors="ignore")
        y = data[target].astype(float)
        stratify = None
    else:
        target = SCHEMA.SALES_CLASSIFICATION
        y = (data[target].astype(str) == "High").astype(int)
        stratify = y

    numeric, categorical = select_features(data, include_leakage=include_leakage)
    x = data[numeric + categorical].copy()

    # First carve out the test set, then split the remainder into train/val.
    x_tr, x_test, y_tr, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    rel_val = val_size / (1.0 - test_size)
    strat2 = y_tr if task == "classification" else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_tr, y_tr, test_size=rel_val, random_state=seed, stratify=strat2
    )

    return Dataset(
        X_train=x_train,
        X_val=x_val,
        X_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        numeric=numeric,
        categorical=categorical,
        task=task,
    )

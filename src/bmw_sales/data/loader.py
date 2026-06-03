"""Dataset loading with schema enforcement.

The loader is the single supported entrypoint for reading the raw BMW dataset.
It validates structure on the way in so that downstream code can assume a clean,
well-typed frame and fail fast (with a clear message) otherwise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from bmw_sales.config import RAW_DATASET_PATH, SCHEMA


class SchemaValidationError(ValueError):
    """Raised when the loaded dataset does not match the expected schema."""


#: Expected pandas dtypes after load. CSV ints can read as int64; we coerce
#: categoricals to ``category`` for memory efficiency and correct modelling.
_EXPECTED_COLUMNS: tuple[str, ...] = (
    SCHEMA.MODEL, SCHEMA.YEAR, SCHEMA.REGION, SCHEMA.COLOR, SCHEMA.FUEL_TYPE,
    SCHEMA.TRANSMISSION, SCHEMA.ENGINE_SIZE_L, SCHEMA.MILEAGE_KM,
    SCHEMA.PRICE_USD, SCHEMA.SALES_VOLUME, SCHEMA.SALES_CLASSIFICATION,
)


def _validate_columns(df: pd.DataFrame) -> None:
    """Ensure every expected column is present (order-independent)."""
    missing = [c for c in _EXPECTED_COLUMNS if c not in df.columns]
    unexpected = [c for c in df.columns if c not in _EXPECTED_COLUMNS]
    if missing:
        raise SchemaValidationError(f"Missing expected columns: {missing}")
    if unexpected:
        raise SchemaValidationError(f"Unexpected columns present: {unexpected}")


def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to their canonical dtypes (categoricals as ``category``)."""
    df = df.copy()
    for col in SCHEMA.CATEGORICAL + (SCHEMA.SALES_CLASSIFICATION,):
        df[col] = df[col].astype("category")
    df[SCHEMA.YEAR] = df[SCHEMA.YEAR].astype("int16")
    df[SCHEMA.MILEAGE_KM] = df[SCHEMA.MILEAGE_KM].astype("int32")
    df[SCHEMA.PRICE_USD] = df[SCHEMA.PRICE_USD].astype("int32")
    df[SCHEMA.SALES_VOLUME] = df[SCHEMA.SALES_VOLUME].astype("int32")
    df[SCHEMA.ENGINE_SIZE_L] = df[SCHEMA.ENGINE_SIZE_L].astype("float32")
    return df


def load_raw(
    path: Optional[Path] = None, *, validate: bool = True, apply_dtypes: bool = True
) -> pd.DataFrame:
    """Load the raw BMW sales dataset.

    Parameters
    ----------
    path:
        Override the default raw dataset location (useful for tests/fixtures).
    validate:
        If ``True`` (default), enforce the expected column schema.
    apply_dtypes:
        If ``True`` (default), coerce columns to canonical, memory-efficient dtypes.

    Returns
    -------
    pandas.DataFrame
        The validated dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    SchemaValidationError
        If validation is enabled and the schema does not match.
    """
    csv_path = Path(path) if path is not None else RAW_DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{csv_path}'. "
            "Place 'BMW_sales_data_(2010-2024).csv' under data/raw/."
        )

    df = pd.read_csv(csv_path)

    if validate:
        _validate_columns(df)
    if apply_dtypes:
        df = _apply_dtypes(df)
    return df

"""Tests for the dataset loader and schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bmw_sales.config import SCHEMA
from bmw_sales.data.loader import SchemaValidationError, load_raw


def test_load_valid_csv(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    csv = tmp_path / "valid.csv"
    sample_df.to_csv(csv, index=False)
    df = load_raw(csv)
    assert df.shape[0] == sample_df.shape[0]
    assert set(df.columns) == set(sample_df.columns)


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(Path("does/not/exist.csv"))


def test_missing_column_raises(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    bad = sample_df.drop(columns=[SCHEMA.PRICE_USD])
    csv = tmp_path / "bad.csv"
    bad.to_csv(csv, index=False)
    with pytest.raises(SchemaValidationError, match="Missing"):
        load_raw(csv)


def test_unexpected_column_raises(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    extra = sample_df.assign(Spurious=1)
    csv = tmp_path / "extra.csv"
    extra.to_csv(csv, index=False)
    with pytest.raises(SchemaValidationError, match="Unexpected"):
        load_raw(csv)


def test_dtype_coercion(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    csv = tmp_path / "valid.csv"
    sample_df.to_csv(csv, index=False)
    df = load_raw(csv)
    assert str(df[SCHEMA.MODEL].dtype) == "category"
    assert str(df[SCHEMA.YEAR].dtype) == "int16"
    assert str(df[SCHEMA.ENGINE_SIZE_L].dtype) == "float32"

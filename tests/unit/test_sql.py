"""Tests for the DuckDB SQL analytics layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bmw_sales.sql.analytics import list_queries, run_all, run_query


@pytest.fixture
def csv_path(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    p = tmp_path / "bmw.csv"
    sample_df.to_csv(p, index=False)
    return p


def test_lists_all_queries() -> None:
    names = list_queries()
    assert "top_regions_by_volume" in names
    assert "yoy_volume" in names
    assert len(names) >= 5


def test_unknown_query_raises(csv_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_query("does_not_exist", dataset_path=csv_path)


def test_region_shares_sum_to_100(csv_path: Path) -> None:
    df = run_query("top_regions_by_volume", dataset_path=csv_path)
    assert {"region", "total_volume", "pct_of_total"} <= set(df.columns)
    assert abs(df["pct_of_total"].sum() - 100.0) < 0.5


def test_electrification_pct_in_range(csv_path: Path) -> None:
    df = run_query("electrification_by_region", dataset_path=csv_path)
    assert df["electrified_pct"].between(0, 100).all()


def test_yoy_first_year_is_null(csv_path: Path) -> None:
    df = run_query("yoy_volume", dataset_path=csv_path).sort_values("year")
    assert pd.isna(df.iloc[0]["yoy_pct"])


def test_run_all_returns_every_query(csv_path: Path) -> None:
    results = run_all(dataset_path=csv_path)
    assert set(results) == set(list_queries())
    assert all(isinstance(v, pd.DataFrame) for v in results.values())

"""Tests for the hybrid API clients and enrichment layer (offline/mock path)."""

from __future__ import annotations

import pandas as pd

from bmw_sales.apis.base import DataSource
from bmw_sales.apis.co2_regulations import CO2RegulationClient
from bmw_sales.apis.enrichment import enrich_dataset
from bmw_sales.apis.fx_rates import FXRateClient
from bmw_sales.apis.worldbank import WorldBankClient


def test_offline_mode_returns_mock() -> None:
    res = WorldBankClient().fetch(region="Europe", start_year=2015, end_year=2020)
    assert res.source is DataSource.MOCK
    assert not res.data.empty


def test_mock_is_deterministic() -> None:
    a = CO2RegulationClient().fetch(region="Asia", start_year=2010, end_year=2024).data
    b = CO2RegulationClient().fetch(region="Asia", start_year=2010, end_year=2024).data
    pd.testing.assert_frame_equal(a, b)


def test_regulation_stringency_increases_over_time() -> None:
    df = CO2RegulationClient().fetch(region="Europe", start_year=2010, end_year=2024).data
    # Europe tightens regulation; last year should exceed first by a clear margin.
    assert df["regulation_stringency_index"].iloc[-1] > df["regulation_stringency_index"].iloc[0]


def test_co2_client_exposes_emissions_per_capita() -> None:
    df = CO2RegulationClient().fetch(region="Europe", start_year=2015, end_year=2020).data
    assert "co2_emissions_pc" in df.columns
    assert (df["co2_emissions_pc"] > 0).all()


def test_fx_usd_region_is_unity() -> None:
    df = FXRateClient().fetch(region="North America", start_year=2018, end_year=2020).data
    assert (df["usd_per_unit"] == 1.0).all()


def test_enrichment_adds_columns_without_nulls(sample_df: pd.DataFrame) -> None:
    result = enrich_dataset(sample_df)
    added = [c for c in result.data.columns if c not in sample_df.columns]
    assert "inflation_pct" in added
    assert "price_usd_per_litre" in added
    assert int(result.data[added].isna().sum().sum()) == 0
    assert len(result.data) == len(sample_df)  # left join keeps all rows


def test_provenance_reported(sample_df: pd.DataFrame) -> None:
    result = enrich_dataset(sample_df)
    assert set(result.provenance) == {"worldbank", "fuel_prices", "co2_regulations", "fx_rates"}
    assert all(v == "mock" for v in result.provenance.values())

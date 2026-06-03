"""Augmentation layer: join external API data onto the BMW sales dataset.

Builds a region×year (×fuel) **external panel** from the four hybrid clients and
left-joins it onto the transactional sales data. All joins are left joins so the
sales data is never dropped, and provenance per source is reported so the UI can
show whether each block came from a live API or a mock fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bmw_sales.apis._regions import REGIONS
from bmw_sales.apis.base import BaseAPIClient, DataSource
from bmw_sales.apis.co2_regulations import CO2RegulationClient
from bmw_sales.apis.fuel_prices import FuelPriceClient
from bmw_sales.apis.fx_rates import FXRateClient
from bmw_sales.apis.worldbank import WorldBankClient
from bmw_sales.config import SCHEMA


@dataclass
class EnrichmentResult:
    """Augmented dataset plus per-source provenance."""

    data: pd.DataFrame
    provenance: dict[str, str]


def _collect(client: BaseAPIClient, start: int, end: int) -> tuple[pd.DataFrame, str]:
    """Fetch every region from ``client`` and report the dominant provenance."""
    parts: list[pd.DataFrame] = []
    sources: set[str] = set()
    for region in REGIONS:
        res = client.fetch(region=region, start_year=start, end_year=end)
        parts.append(res.data)
        sources.add(res.source.value)
    # Report the "weakest" provenance actually used (mock > cache > live).
    if DataSource.MOCK.value in sources:
        dominant = DataSource.MOCK.value
    elif DataSource.CACHE.value in sources:
        dominant = DataSource.CACHE.value
    else:
        dominant = DataSource.LIVE.value
    return pd.concat(parts, ignore_index=True), dominant


def build_external_panel(
    start_year: int = 2010, end_year: int = 2024
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Assemble the region×year(×fuel) external panel from all four clients."""
    macro, macro_src = _collect(WorldBankClient(), start_year, end_year)
    fuel_panel, fuel_src = _collect(FuelPriceClient(), start_year, end_year)
    co2_panel, co2_src = _collect(CO2RegulationClient(), start_year, end_year)
    fx_panel, fx_src = _collect(FXRateClient(), start_year, end_year)

    # region×year block (macro + co2 + fx); fuel stays region×year×fuel_type.
    panel = macro.merge(co2_panel, on=["region", "year"], how="outer")
    panel = panel.merge(fx_panel, on=["region", "year"], how="outer")

    provenance = {
        "worldbank": macro_src,
        "fuel_prices": fuel_src,
        "co2_regulations": co2_src,
        "fx_rates": fx_src,
    }
    return panel, fuel_panel, provenance


def enrich_dataset(
    df: pd.DataFrame, *, start_year: int = 2010, end_year: int = 2024
) -> EnrichmentResult:
    """Left-join the external panel onto the sales dataset.

    Parameters
    ----------
    df:
        The raw/clean sales dataset (must contain Region, Year, Fuel_Type).

    Returns
    -------
    EnrichmentResult
        The augmented frame and per-source provenance (``live`` / ``mock``).
    """
    region_panel, fuel_panel, provenance = build_external_panel(start_year, end_year)

    out = df.copy()
    # Normalise join keys (sales data uses categorical dtypes).
    out["_region"] = out[SCHEMA.REGION].astype(str)
    out["_year"] = out[SCHEMA.YEAR].astype(int)
    out["_fuel"] = out[SCHEMA.FUEL_TYPE].astype(str)

    out = out.merge(
        region_panel.rename(columns={"region": "_region", "year": "_year"}),
        on=["_region", "_year"],
        how="left",
    )
    out = out.merge(
        fuel_panel.rename(columns={"region": "_region", "year": "_year", "fuel_type": "_fuel"}),
        on=["_region", "_year", "_fuel"],
        how="left",
    )
    out = out.drop(columns=["_region", "_year", "_fuel"])

    return EnrichmentResult(data=out, provenance=provenance)


def summarise_provenance(provenance: dict[str, str]) -> str:
    """One-line human summary of where the external data came from."""
    live = [k for k, v in provenance.items() if v == DataSource.LIVE.value]
    mock = [k for k, v in provenance.items() if v == DataSource.MOCK.value]
    cache = [k for k, v in provenance.items() if v == DataSource.CACHE.value]
    bits = []
    if live:
        bits.append(f"live: {', '.join(live)}")
    if cache:
        bits.append(f"cache: {', '.join(cache)}")
    if mock:
        bits.append(f"mock: {', '.join(mock)}")
    return " | ".join(bits) if bits else "no sources"

"""Fuel prices per region/year/fuel type.

The live path tries the World Bank pump-price indicator (EP.PMP.SGAS.CD), but
that series was archived in the 2024 refresh and no keyless replacement exists,
so in practice this falls back to the mock (regional USD/litre baselines with
fuel-type multipliers) and reports its provenance as 'mock'. The _fetch_live hook
is kept for when a provider is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bmw_sales.apis._regions import REGIONS, ref_for
from bmw_sales.apis.base import BaseAPIClient

#: World Bank: pump price for gasoline (US$ per litre).
_WB_PUMP_PRICE = "EP.PMP.SGAS.CD"

#: Realistic ~2017 regional baselines, USD/litre (petrol equivalent).
_BASELINE_USD_PER_L: dict[str, float] = {
    "Europe": 1.55,
    "Asia": 1.05,
    "North America": 0.78,
    "Middle East": 0.45,
    "South America": 1.00,
    "Africa": 0.95,
}
#: Fuel-type multipliers relative to the petrol baseline.
_FUEL_MULTIPLIER: dict[str, float] = {
    "Petrol": 1.00,
    "Diesel": 0.92,
    "Hybrid": 0.55,
    "Electric": 0.30,
}


class FuelPriceClient(BaseAPIClient):
    """Regional historical fuel prices by year and fuel type (USD/litre eq.)."""

    name = "fuel_prices"

    @staticmethod
    def _build_frame(region: str, years: np.ndarray, petrol: np.ndarray) -> pd.DataFrame:
        """Expand a region's petrol price series into a region×year×fuel frame."""
        records: list[dict[str, Any]] = []
        for fuel, mult in _FUEL_MULTIPLIER.items():
            for yr, p in zip(years, petrol):
                records.append(
                    {
                        "region": region,
                        "year": int(yr),
                        "fuel_type": fuel,
                        "price_usd_per_litre": round(float(max(p * mult, 0.05)), 3),
                    }
                )
        return pd.DataFrame(records)

    def _fetch_live(self, **params: Any) -> pd.DataFrame:
        region = params["region"]
        start = int(params.get("start_year", 2010))
        end = int(params.get("end_year", 2024))
        iso3 = ref_for(region).country_iso3

        # Pump-price surveys are biennial; pull a wider window then fill the gaps.
        raw = self._fetch_wb_indicator(iso3, _WB_PUMP_PRICE, start - 6, end)
        if not raw:
            raise ValueError(f"No World Bank pump-price data for {region}")

        years = np.arange(start, end + 1)
        series = pd.Series(raw).sort_index()
        # Reindex onto the full horizon and forward/back-fill the biennial gaps.
        filled = series.reindex(range(start - 6, end + 1)).ffill().bfill()
        petrol = np.array([float(filled.loc[y]) for y in years])
        return self._build_frame(region, years, petrol)

    def _mock(self, **params: Any) -> pd.DataFrame:
        region = params["region"]
        start = int(params.get("start_year", 2010))
        end = int(params.get("end_year", 2024))
        rng = np.random.default_rng(self._seed_from(self.name, region, start, end))

        years = np.arange(start, end + 1)
        base = _BASELINE_USD_PER_L.get(region, 1.0)
        # Long-run drift (~1.5%/yr) anchored at 2017, plus shocks.
        drift = 1 + 0.015 * (years - 2017)
        shocks = rng.normal(0, 0.06, size=len(years))
        petrol = base * drift * (1 + shocks)
        return self._build_frame(region, years, petrol)

    def fetch_all_regions(self, start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """Fetch & concatenate fuel prices for every region."""
        parts = [
            self.fetch(region=r, start_year=start_year, end_year=end_year).data for r in REGIONS
        ]
        return pd.concat(parts, ignore_index=True)

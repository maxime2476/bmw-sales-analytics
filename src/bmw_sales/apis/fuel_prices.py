"""Historical fuel-price client (mock-first, real-hook ready).

Pump prices vary strongly by region and are a key driver of demand for fuel
types (Petrol/Diesel vs Hybrid/Electric). Reliable historical pump-price series
are typically behind keyed providers, so this client is *mock-first* but exposes
the same hybrid interface: if a real endpoint is configured later, only
``_fetch_live`` needs filling in.

Mock values are anchored to realistic regional baselines (USD per litre) with a
mild upward trend and year-to-year volatility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bmw_sales.apis._regions import REGIONS
from bmw_sales.apis.base import BaseAPIClient

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

    def _fetch_live(self, **params: Any) -> pd.DataFrame:
        # No keyless public historical pump-price API is reliably available;
        # raising here makes the client transparently fall back to the mock.
        raise NotImplementedError(
            "No real fuel-price provider configured; using deterministic mock."
        )

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

    def fetch_all_regions(self, start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """Fetch & concatenate fuel prices for every region."""
        parts = [
            self.fetch(region=r, start_year=start_year, end_year=end_year).data for r in REGIONS
        ]
        return pd.concat(parts, ignore_index=True)

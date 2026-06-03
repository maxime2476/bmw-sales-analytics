"""Foreign-exchange client (real exchangerate.host hook + mock fallback).

Returns the annual average USD value of one unit of each region's representative
currency. This lets the pricing analysis normalise nominal ``Price_USD`` into
local purchasing terms, and feeds the Scenario Simulator's FX sensitivity.

The live path targets exchangerate.host; if it is unreachable or requires a key
that is not configured, the client transparently falls back to a deterministic
mock anchored to realistic recent rates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bmw_sales.apis._regions import REGIONS, ref_for
from bmw_sales.apis.base import BaseAPIClient

#: Approx. recent USD value of 1 unit of the currency (anchor for the mock).
_USD_PER_UNIT_ANCHOR: dict[str, float] = {
    "USD": 1.00,
    "EUR": 1.08,
    "CNY": 0.14,
    "AED": 0.272,
    "BRL": 0.18,
    "ZAR": 0.054,
}
#: Mild annual drift applied to the anchor (depreciation = negative).
_ANNUAL_DRIFT: dict[str, float] = {
    "USD": 0.0,
    "EUR": -0.005,
    "CNY": -0.010,
    "AED": 0.0,
    "BRL": -0.030,
    "ZAR": -0.035,
}


class FXRateClient(BaseAPIClient):
    """Annual average USD-per-unit exchange rates for regional currencies."""

    name = "fx_rates"

    def _fetch_live(self, **params: Any) -> pd.DataFrame:
        region = params["region"]
        start = int(params.get("start_year", 2010))
        end = int(params.get("end_year", 2024))
        currency = ref_for(region).currency
        if currency == "USD":
            years = np.arange(start, end + 1)
            return pd.DataFrame(
                {"region": region, "year": years, "currency": "USD", "usd_per_unit": 1.0}
            )

        records: list[dict[str, Any]] = []
        for year in range(start, end + 1):
            url = f"{self.settings.fx_base_url}/{year}-12-31"
            query: dict[str, Any] = {"base": currency, "symbols": "USD"}
            if self.settings.fx_api_key:
                query["access_key"] = self.settings.fx_api_key
            payload = self._http_get_json(url, params=query)
            rate = (payload.get("rates") or {}).get("USD")
            if rate is None:
                raise ValueError(f"FX rate missing for {currency} {year}")
            records.append(
                {"region": region, "year": year, "currency": currency, "usd_per_unit": float(rate)}
            )
        return pd.DataFrame(records)

    def _mock(self, **params: Any) -> pd.DataFrame:
        region = params["region"]
        start = int(params.get("start_year", 2010))
        end = int(params.get("end_year", 2024))
        currency = ref_for(region).currency
        rng = np.random.default_rng(self._seed_from(self.name, region, start, end))

        years = np.arange(start, end + 1)
        if currency == "USD":
            # The USD-denominated region is the numeraire: exactly 1.0, no noise.
            usd_per_unit = np.ones(len(years))
        else:
            anchor = _USD_PER_UNIT_ANCHOR.get(currency, 0.5)
            drift = _ANNUAL_DRIFT.get(currency, -0.01)
            # Anchor is "recent"; walk backwards/forwards from the 2024 reference.
            factor = (1 + drift) ** (years - 2024)
            noise = 1 + rng.normal(0, 0.02, size=len(years))
            usd_per_unit = anchor * factor * noise

        return pd.DataFrame(
            {
                "region": region,
                "year": years.astype(int),
                "currency": currency,
                "usd_per_unit": np.round(usd_per_unit, 5),
            }
        )

    def fetch_all_regions(self, start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """Fetch & concatenate FX rates for every region."""
        parts = [
            self.fetch(region=r, start_year=start_year, end_year=end_year).data for r in REGIONS
        ]
        return pd.concat(parts, ignore_index=True)

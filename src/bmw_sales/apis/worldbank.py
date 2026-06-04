"""World Bank macro-economic client (real, keyless API + mock fallback).

The World Bank Indicators API is public and requires no key. We pull two
indicators per region aggregate over the dataset horizon (2010–2024):

- ``FP.CPI.TOTL.ZG`` — inflation, consumer prices (annual %).
- ``NY.GDP.PCAP.CD`` — GDP per capita (current US$), a purchasing-power proxy.

These let us confront sales against regional macro conditions in the econometric
and simulation layers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bmw_sales.apis._regions import REGIONS, ref_for
from bmw_sales.apis.base import BaseAPIClient

INDICATORS: dict[str, str] = {
    "inflation_pct": "FP.CPI.TOTL.ZG",
    "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
}

#: Plausible long-run baselines per region for the mock (current US$ / annual %).
_MOCK_GDP_BASE: dict[str, float] = {
    "Asia": 11000,
    "North America": 62000,
    "Middle East": 22000,
    "South America": 9000,
    "Europe": 38000,
    "Africa": 1700,
}
_MOCK_INFLATION_BASE: dict[str, float] = {
    "Asia": 2.6,
    "North America": 2.3,
    "Middle East": 3.5,
    "South America": 6.0,
    "Europe": 1.9,
    "Africa": 7.5,
}


class WorldBankClient(BaseAPIClient):
    """Fetch regional inflation & GDP-per-capita from the World Bank API."""

    name = "worldbank"

    def _fetch_live(self, **params: Any) -> pd.DataFrame:
        region = params["region"]
        start = int(params.get("start_year", 2010))
        end = int(params.get("end_year", 2024))
        code = ref_for(region).worldbank_code

        frames: list[pd.DataFrame] = []
        for friendly, indicator in INDICATORS.items():
            url = f"{self.settings.worldbank_base_url}/country/{code}" f"/indicator/{indicator}"
            payload = self._http_get_json(
                url,
                params={"date": f"{start}:{end}", "format": "json", "per_page": "500"},
            )
            rows = self._parse_wb_payload(payload, friendly)
            # Guarantee columns exist even when the aggregate has no observations,
            # so the downstream merge on "year" never raises.
            frames.append(pd.DataFrame(rows, columns=["year", friendly]))

        df = frames[0]
        for extra in frames[1:]:
            df = df.merge(extra, on="year", how="outer")
        if df.empty:
            raise ValueError(f"World Bank returned no observations for {region}")
        df.insert(0, "region", region)
        return df.sort_values("year").reset_index(drop=True)

    @staticmethod
    def _parse_wb_payload(payload: Any, value_name: str) -> list[dict[str, Any]]:
        """Parse the WB ``[metadata, observations]`` JSON envelope."""
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            raise ValueError("Unexpected World Bank payload shape")
        out: list[dict[str, Any]] = []
        for obs in payload[1]:
            if obs.get("value") is None:
                continue
            out.append({"year": int(obs["date"]), value_name: float(obs["value"])})
        return out

    def _mock(self, **params: Any) -> pd.DataFrame:
        region = params["region"]
        start = int(params.get("start_year", 2010))
        end = int(params.get("end_year", 2024))
        rng = np.random.default_rng(self._seed_from(self.name, region, start, end))

        years = np.arange(start, end + 1)
        gdp0 = _MOCK_GDP_BASE.get(region, 15000)
        infl0 = _MOCK_INFLATION_BASE.get(region, 3.0)

        # GDP per capita: gentle real growth + noise; inflation: mean-reverting.
        growth = 1 + rng.normal(0.022, 0.01, size=len(years)).cumsum() / 10
        gdp = gdp0 * growth
        inflation = infl0 + rng.normal(0, 0.8, size=len(years))

        return pd.DataFrame(
            {
                "region": region,
                "year": years,
                "inflation_pct": np.round(inflation, 2),
                "gdp_per_capita_usd": np.round(gdp, 0),
            }
        )

    def fetch_all_regions(self, start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """Convenience: fetch & concatenate macro data for every region."""
        parts = [
            self.fetch(region=r, start_year=start_year, end_year=end_year).data for r in REGIONS
        ]
        return pd.concat(parts, ignore_index=True)

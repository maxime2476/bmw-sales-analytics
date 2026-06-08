"""CO2 context per region and year.

Returns real World Bank CO2 emissions per capita (live, with a mock fallback)
plus a curated regulation-stringency index and fleet CO2 target. The regulation
figures are a labelled proxy since no free API publishes that series.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bmw_sales.apis._regions import REGIONS, ref_for
from bmw_sales.apis.base import BaseAPIClient

#: World Bank: CO2 emissions per capita (AR5). The legacy ``EN.ATM.CO2E.PC`` was
#: archived in the 2024 WB data refresh; this AR5 series is the live replacement.
_WB_CO2_PC = "EN.GHG.CO2.PC.CE.AR5"

#: Plausible ~2015 CO2 emissions per capita (t) per region, for the mock.
_EMISSIONS_PC_BASE: dict[str, float] = {
    "North America": 15.0,
    "Middle East": 12.0,
    "Europe": 6.5,
    "Asia": 5.0,
    "South America": 2.5,
    "Africa": 0.9,
}

#: 2010 stringency index (0–100) and annual tightening per region.
_STRINGENCY_2010: dict[str, float] = {
    "Europe": 55,
    "North America": 42,
    "Asia": 35,
    "South America": 22,
    "Africa": 15,
    "Middle East": 12,
}
_ANNUAL_TIGHTENING: dict[str, float] = {
    "Europe": 2.6,
    "North America": 1.9,
    "Asia": 2.3,
    "South America": 1.2,
    "Africa": 0.9,
    "Middle East": 1.0,
}
#: Fleet CO2 target (g/km) shrinks as stringency rises.
_CO2_TARGET_2010 = 160.0


class CO2RegulationClient(BaseAPIClient):
    """Regional emissions-regulation stringency and fleet CO2 targets by year."""

    name = "co2_regulations"

    @staticmethod
    def _curated_schedule(region: str, years: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic curated stringency index and fleet CO2 target (no noise)."""
        base = _STRINGENCY_2010.get(region, 25)
        slope = _ANNUAL_TIGHTENING.get(region, 1.5)
        stringency = np.clip(base + slope * (years - 2010), 0, 100)
        co2_target = np.maximum(_CO2_TARGET_2010 - 0.6 * (stringency - base), 45)
        return stringency, co2_target

    def _frame(
        self,
        region: str,
        years: np.ndarray,
        stringency: np.ndarray,
        co2_target: np.ndarray,
        emissions_pc: np.ndarray,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "region": region,
                "year": years.astype(int),
                "regulation_stringency_index": np.round(stringency, 1),
                "fleet_co2_target_g_km": np.round(co2_target, 1),
                "co2_emissions_pc": np.round(emissions_pc, 3),
            }
        )

    def _fetch_live(self, **params: Any) -> pd.DataFrame:
        region = params["region"]
        start = int(params.get("start_year", 2010))
        end = int(params.get("end_year", 2024))

        raw = self._fetch_wb_indicator(ref_for(region).worldbank_code, _WB_CO2_PC, start - 4, end)
        if not raw:
            raise ValueError(f"No World Bank CO2 emissions data for {region}")

        years = np.arange(start, end + 1)
        series = pd.Series(raw).sort_index().reindex(range(start - 4, end + 1)).ffill().bfill()
        emissions = np.array([float(series.loc[y]) for y in years])
        stringency, co2_target = self._curated_schedule(region, years)
        return self._frame(region, years, stringency, co2_target, emissions)

    def _mock(self, **params: Any) -> pd.DataFrame:
        region = params["region"]
        start = int(params.get("start_year", 2010))
        end = int(params.get("end_year", 2024))
        rng = np.random.default_rng(self._seed_from(self.name, region, start, end))

        years = np.arange(start, end + 1)
        stringency, co2_target = self._curated_schedule(region, years)
        stringency = np.clip(stringency + rng.normal(0, 0.5, len(years)), 0, 100)
        # Emissions per capita: mild decline from a regional baseline + noise.
        base_e = _EMISSIONS_PC_BASE.get(region, 4.0)
        emissions = base_e * (1 - 0.008 * (years - 2015)) * (1 + rng.normal(0, 0.03, len(years)))
        return self._frame(region, years, stringency, co2_target, np.maximum(emissions, 0.05))

    def fetch_all_regions(self, start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """Fetch & concatenate CO2 regulation data for every region."""
        parts = [
            self.fetch(region=r, start_year=start_year, end_year=end_year).data for r in REGIONS
        ]
        return pd.concat(parts, ignore_index=True)

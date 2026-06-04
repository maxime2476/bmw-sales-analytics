"""CO2 / emissions-regulation stringency client (mock-first).

Provides a regional, time-varying **regulatory stringency index** (0–100) plus a
fleet CO2 target (g/km), capturing how aggressively each region pushes the shift
to low-emission vehicles. Europe leads, the Middle East trails; all tighten over
time. Used to narrate and simulate the Petrol→Electric transition for BMW.

Mock-first by design: there is no single canonical free API for historical
emission-regulation stringency, so values encode a curated, defensible schedule.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bmw_sales.apis._regions import REGIONS
from bmw_sales.apis.base import BaseAPIClient

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

    def _fetch_live(self, **params: Any) -> pd.DataFrame:
        raise NotImplementedError(
            "No canonical free CO2-regulation API configured; using curated mock."
        )

    def _mock(self, **params: Any) -> pd.DataFrame:
        region = params["region"]
        start = int(params.get("start_year", 2010))
        end = int(params.get("end_year", 2024))
        rng = np.random.default_rng(self._seed_from(self.name, region, start, end))

        years = np.arange(start, end + 1)
        base = _STRINGENCY_2010.get(region, 25)
        slope = _ANNUAL_TIGHTENING.get(region, 1.5)
        stringency = np.clip(base + slope * (years - 2010) + rng.normal(0, 0.5, len(years)), 0, 100)
        # Targets fall ~0.6 g/km per stringency point above the 2010 anchor.
        co2_target = np.maximum(_CO2_TARGET_2010 - 0.6 * (stringency - base), 45)

        return pd.DataFrame(
            {
                "region": region,
                "year": years.astype(int),
                "regulation_stringency_index": np.round(stringency, 1),
                "fleet_co2_target_g_km": np.round(co2_target, 1),
            }
        )

    def fetch_all_regions(self, start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        """Fetch & concatenate CO2 regulation data for every region."""
        parts = [
            self.fetch(region=r, start_year=start_year, end_year=end_year).data for r in REGIONS
        ]
        return pd.concat(parts, ignore_index=True)

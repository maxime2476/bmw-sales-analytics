"""Mapping of the dataset's 6 regions to external-data identifiers.

The BMW dataset uses coarse continental regions. To join them against real
external sources we map each region to:

- a **World Bank aggregate code** (real WB region aggregates),
- a **representative currency** (for FX), and
- a **representative country label** (documentation only).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionRef:
    """External-data references for one dataset region."""

    region: str
    worldbank_code: str  # WB aggregate code (real) - for indicators with aggregates
    currency: str  # ISO-4217, representative
    representative: str  # human label
    country_iso3: str  # representative country (for country-only indicators)


#: Region → external references. WB aggregate codes for macro indicators; a
#: representative ISO-3 country for indicators only published at country level
#: (e.g. pump prices).
REGION_REFS: dict[str, RegionRef] = {
    "Asia": RegionRef("Asia", "EAS", "CNY", "East Asia & Pacific", "CHN"),
    "North America": RegionRef("North America", "NAC", "USD", "North America", "USA"),
    "Middle East": RegionRef("Middle East", "MEA", "AED", "Middle East & N. Africa", "SAU"),
    "South America": RegionRef("South America", "LCN", "BRL", "Latin America & Carib.", "BRA"),
    "Europe": RegionRef("Europe", "EMU", "EUR", "Euro area", "DEU"),
    "Africa": RegionRef("Africa", "SSF", "ZAR", "Sub-Saharan Africa", "ZAF"),
}

REGIONS: tuple[str, ...] = tuple(REGION_REFS.keys())


def ref_for(region: str) -> RegionRef:
    """Return the :class:`RegionRef` for ``region`` (raises ``KeyError`` if unknown)."""
    return REGION_REFS[region]

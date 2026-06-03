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
    worldbank_code: str  # WB aggregate code (real)
    currency: str  # ISO-4217, representative
    representative: str  # human label


#: Region → external references. WB codes are official aggregate identifiers.
REGION_REFS: dict[str, RegionRef] = {
    "Asia": RegionRef("Asia", "EAS", "CNY", "East Asia & Pacific"),
    "North America": RegionRef("North America", "NAC", "USD", "North America"),
    "Middle East": RegionRef("Middle East", "MEA", "AED", "Middle East & N. Africa"),
    "South America": RegionRef("South America", "LCN", "BRL", "Latin America & Carib."),
    "Europe": RegionRef("Europe", "EMU", "EUR", "Euro area"),
    "Africa": RegionRef("Africa", "SSF", "ZAR", "Sub-Saharan Africa"),
}

REGIONS: tuple[str, ...] = tuple(REGION_REFS.keys())


def ref_for(region: str) -> RegionRef:
    """Return the :class:`RegionRef` for ``region`` (raises ``KeyError`` if unknown)."""
    return REGION_REFS[region]

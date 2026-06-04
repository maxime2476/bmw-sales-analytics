"""Scenario Simulator — forward-looking demand decision-support.

⚠️ **This is an explicit what-if simulation, not a fit to the historical data.**
The source dataset carries no predictive signal (ADR-0002), so we cannot forecast
demand from it. Instead, this module projects demand under user-chosen scenarios
using a transparent constant-elasticity model whose parameters come from the
**automotive-economics literature** (cited below) and whose baselines come from
the **real external macro APIs**. Every assumption is visible and adjustable.

Model
-----
Projected demand is the baseline volume scaled by independent multiplicative
factors (a standard log-linear / constant-elasticity demand specification):

    Q' = Q0 · (1+Δp)^εp · (1+Δy)^εy · (1+Δf)^εf · R(Δs) · (1+Δfx)^εp

where Δp = list-price change, Δy = income (GDP/cap) growth, Δf = fuel-price
change, Δs = regulation-stringency change, Δfx = local-currency depreciation
(passed through to effective price via the own-price elasticity).

Elasticity priors (orders of magnitude from the literature)
-----------------------------------------------------------
- **Own-price elasticity εp ≈ -0.6.** Luxury vehicles are relatively price-
  inelastic (segment estimates commonly fall in the -0.2…-1.0 range).
- **Income elasticity εy ≈ +1.5.** Luxury cars are superior goods (εy > 1).
- **Fuel-price cross-elasticity εf ≈ -0.15** for combustion variants; small and
  positive for electrified variants (substitution).
- **Regulation response.** Tighter CO₂ rules shift demand toward electrified
  models and away from combustion ones; modelled as ±r per +10 stringency pts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bmw_sales.apis.co2_regulations import CO2RegulationClient
from bmw_sales.apis.fuel_prices import FuelPriceClient
from bmw_sales.apis.worldbank import WorldBankClient

ELECTRIFIED_FUELS: frozenset[str] = frozenset({"Hybrid", "Electric"})


@dataclass(frozen=True)
class ElasticityAssumptions:
    """Literature-grounded elasticity priors (all user-overridable in the UI)."""

    own_price: float = -0.6
    income: float = 1.5
    fuel_price_combustion: float = -0.15
    fuel_price_electrified: float = 0.10
    #: Demand share shifted per +10 stringency points (toward electrified).
    regulation_per_10pts: float = 0.08


@dataclass
class ScenarioInput:
    """A single what-if scenario."""

    region: str
    fuel_type: str
    base_volume: float
    price_change_pct: float = 0.0  # Δ list price (%)
    gdp_growth_pct: float = 0.0  # Δ income / GDP per capita (%)
    fuel_price_change_pct: float = 0.0  # Δ pump price (%)
    regulation_change_pts: float = 0.0  # Δ stringency index (absolute pts)
    fx_depreciation_pct: float = 0.0  # local-currency depreciation vs USD (%)


@dataclass
class FactorContribution:
    """Multiplicative contribution of one driver to projected demand."""

    driver: str
    multiplier: float

    @property
    def pct_effect(self) -> float:
        return (self.multiplier - 1.0) * 100.0


@dataclass
class ScenarioResult:
    """Output of a scenario projection."""

    base_volume: float
    projected_volume: float
    contributions: list[FactorContribution] = field(default_factory=list)

    @property
    def total_change_pct(self) -> float:
        if self.base_volume == 0:
            return 0.0
        return (self.projected_volume / self.base_volume - 1.0) * 100.0


def _is_electrified(fuel_type: str) -> bool:
    return fuel_type in ELECTRIFIED_FUELS


def simulate(
    scenario: ScenarioInput, assumptions: ElasticityAssumptions | None = None
) -> ScenarioResult:
    """Project demand for a scenario using the constant-elasticity model.

    All effects are independent and multiplicative; each is reported separately
    so the user can see *why* demand moves, not just by how much.
    """
    a = assumptions or ElasticityAssumptions()
    electrified = _is_electrified(scenario.fuel_type)

    # Convert percentage inputs to fractional changes.
    dp = scenario.price_change_pct / 100.0
    dy = scenario.gdp_growth_pct / 100.0
    df = scenario.fuel_price_change_pct / 100.0
    dfx = scenario.fx_depreciation_pct / 100.0

    price_mult = (1.0 + dp) ** a.own_price
    income_mult = (1.0 + dy) ** a.income
    fuel_elasticity = a.fuel_price_electrified if electrified else a.fuel_price_combustion
    fuel_mult = (1.0 + df) ** fuel_elasticity
    # FX depreciation raises the effective local price -> apply own-price elasticity.
    fx_mult = (1.0 + dfx) ** a.own_price
    # Regulation: tighter rules help electrified, hurt combustion.
    reg_direction = 1.0 if electrified else -1.0
    reg_mult = 1.0 + reg_direction * a.regulation_per_10pts * (
        scenario.regulation_change_pts / 10.0
    )
    reg_mult = max(reg_mult, 0.0)

    contributions = [
        FactorContribution("List price", price_mult),
        FactorContribution("Income (GDP/cap)", income_mult),
        FactorContribution("Fuel price", fuel_mult),
        FactorContribution("CO₂ regulation", reg_mult),
        FactorContribution("FX (currency)", fx_mult),
    ]

    projected = scenario.base_volume
    for c in contributions:
        projected *= c.multiplier

    return ScenarioResult(
        base_volume=scenario.base_volume,
        projected_volume=projected,
        contributions=contributions,
    )


def macro_defaults(region: str, *, year: int = 2024) -> dict[str, float]:
    """Suggest scenario defaults from the (real/mock) external APIs for a region.

    Returns plausible starting values: recent inflation, a GDP-growth proxy, and
    the latest regulation-stringency level — so the UI opens on realistic numbers.
    """
    wb = WorldBankClient().fetch(region=region, start_year=year - 2, end_year=year).data
    co2 = CO2RegulationClient().fetch(region=region, start_year=year, end_year=year).data
    fuel = FuelPriceClient().fetch(region=region, start_year=year, end_year=year).data

    inflation = float(wb["inflation_pct"].dropna().tail(1).iloc[0]) if not wb.empty else 3.0
    gdp = wb["gdp_per_capita_usd"].dropna()
    gdp_growth = (
        float((gdp.iloc[-1] / gdp.iloc[0]) ** (1 / max(len(gdp) - 1, 1)) - 1.0) * 100.0
        if len(gdp) >= 2
        else 2.0
    )
    stringency = float(co2["regulation_stringency_index"].iloc[0]) if not co2.empty else 50.0
    fuel_price = float(fuel["price_usd_per_litre"].mean()) if not fuel.empty else 1.0
    return {
        "inflation_pct": round(inflation, 2),
        "gdp_growth_pct": round(gdp_growth, 2),
        "regulation_stringency": round(stringency, 1),
        "fuel_price_usd_per_litre": round(fuel_price, 3),
    }

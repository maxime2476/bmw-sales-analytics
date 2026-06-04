"""Tests for the Scenario Simulator."""

from __future__ import annotations

from bmw_sales.simulation.scenario import (
    ElasticityAssumptions,
    ScenarioInput,
    simulate,
)


def test_no_change_keeps_baseline() -> None:
    res = simulate(ScenarioInput(region="Europe", fuel_type="Petrol", base_volume=5000))
    assert res.projected_volume == 5000
    assert res.total_change_pct == 0.0


def test_price_increase_reduces_demand() -> None:
    res = simulate(
        ScenarioInput(region="Europe", fuel_type="Petrol", base_volume=5000, price_change_pct=10)
    )
    # Own-price elasticity is negative -> higher price lowers demand.
    assert res.projected_volume < 5000


def test_income_growth_increases_demand() -> None:
    res = simulate(
        ScenarioInput(region="Asia", fuel_type="Petrol", base_volume=5000, gdp_growth_pct=10)
    )
    assert res.projected_volume > 5000


def test_regulation_helps_electrified_hurts_combustion() -> None:
    common = dict(region="Europe", base_volume=5000, regulation_change_pts=20)
    ev = simulate(ScenarioInput(fuel_type="Electric", **common))
    ice = simulate(ScenarioInput(fuel_type="Petrol", **common))
    assert ev.projected_volume > 5000
    assert ice.projected_volume < 5000


def test_contributions_multiply_to_total() -> None:
    res = simulate(
        ScenarioInput(
            region="Europe",
            fuel_type="Hybrid",
            base_volume=4000,
            price_change_pct=5,
            gdp_growth_pct=3,
            fuel_price_change_pct=10,
            regulation_change_pts=10,
        )
    )
    product = res.base_volume
    for c in res.contributions:
        product *= c.multiplier
    assert abs(product - res.projected_volume) < 1e-6


def test_custom_assumptions_respected() -> None:
    inelastic = ElasticityAssumptions(own_price=0.0)
    res = simulate(
        ScenarioInput(region="Europe", fuel_type="Petrol", base_volume=5000, price_change_pct=50),
        inelastic,
    )
    # With zero price elasticity, a price change must not move demand.
    assert res.projected_volume == 5000

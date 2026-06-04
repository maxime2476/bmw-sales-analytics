"""Tests for Monte-Carlo uncertainty propagation in the simulator."""

from __future__ import annotations

from bmw_sales.simulation.scenario import ScenarioInput, simulate
from bmw_sales.simulation.uncertainty import (
    ElasticityPriors,
    ScenarioDistribution,
    simulate_mc,
)


def _scenario(**kw) -> ScenarioInput:
    base = dict(region="Europe", fuel_type="Petrol", base_volume=5000)
    base.update(kw)
    return ScenarioInput(**base)


def test_returns_distribution_of_requested_size() -> None:
    dist = simulate_mc(_scenario(price_change_pct=10), n_draws=2000)
    assert isinstance(dist, ScenarioDistribution)
    assert dist.samples.shape == (2000,)


def test_median_tracks_deterministic() -> None:
    sc = _scenario(price_change_pct=8, gdp_growth_pct=3)
    point = simulate(sc).projected_volume
    median = simulate_mc(sc, n_draws=8000).median
    # Median of the MC draws should sit close to the deterministic point estimate.
    assert abs(median - point) / point < 0.05


def test_credible_intervals_nested_and_ordered() -> None:
    dist = simulate_mc(_scenario(price_change_pct=10), n_draws=6000)
    lo80, hi80 = dist.ci80
    lo95, hi95 = dist.ci95
    assert lo95 <= lo80 <= dist.median <= hi80 <= hi95


def test_more_prior_uncertainty_widens_interval() -> None:
    sc = _scenario(price_change_pct=15)
    tight = simulate_mc(sc, ElasticityPriors(own_price_sd=0.05), n_draws=6000)
    wide = simulate_mc(sc, ElasticityPriors(own_price_sd=0.5), n_draws=6000)
    tl, th = tight.ci80
    wl, wh = wide.ci80
    assert (wh - wl) > (th - tl)


def test_price_increase_lowers_demand() -> None:
    dist = simulate_mc(_scenario(price_change_pct=20), n_draws=4000)
    assert dist.median < 5000

"""Tests for the scenario narrator (deterministic template fallback)."""

from __future__ import annotations

from bmw_sales.simulation.narrator import narrate, narration_source
from bmw_sales.simulation.scenario import ScenarioInput, simulate
from bmw_sales.simulation.uncertainty import simulate_mc


def test_source_is_template_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert narration_source() == "template"


def test_narrative_uses_real_numbers_and_disclaimer(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sc = ScenarioInput(region="Europe", fuel_type="Electric", base_volume=5000, gdp_growth_pct=5)
    text = narrate(sc, simulate(sc))
    assert "Europe" in text and "Electric" in text
    assert "what-if" in text.lower()


def test_narrative_directionality(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    up = ScenarioInput(region="Asia", fuel_type="Petrol", base_volume=5000, gdp_growth_pct=10)
    down = ScenarioInput(region="Asia", fuel_type="Petrol", base_volume=5000, price_change_pct=25)
    assert "rise" in narrate(up, simulate(up)).lower()
    assert "fall" in narrate(down, simulate(down)).lower()


def test_narrative_includes_credible_interval(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sc = ScenarioInput(region="Europe", fuel_type="Electric", base_volume=5000, price_change_pct=10)
    text = narrate(sc, simulate(sc), simulate_mc(sc, n_draws=2000))
    assert "credible interval" in text.lower()

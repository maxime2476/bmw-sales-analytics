"""Plain-English narrator for a simulated scenario (optional LLM + offline fallback).

Turns a :class:`ScenarioResult` into a concise, executive-ready narrative. If the
**Anthropic Claude** SDK and an ``ANTHROPIC_API_KEY`` are available, it asks Claude
for polished prose; otherwise it falls back to a **deterministic, template-based**
narrative so the feature works everywhere — offline, in CI and on the deployed app
— with no key required.

The narrative is grounded in the *computed numbers* (it never invents figures),
keeping it consistent with the project's honesty principle.
"""

from __future__ import annotations

import os
from typing import Optional

from bmw_sales.simulation.scenario import ScenarioInput, ScenarioResult
from bmw_sales.simulation.uncertainty import ScenarioDistribution

#: Claude model used when the SDK + key are available.
_CLAUDE_MODEL = "claude-opus-4-8"


def narration_source() -> str:
    """Return which backend `narrate` would use: ``'claude'`` or ``'template'``."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "template"
    try:
        import anthropic  # noqa: F401
    except Exception:  # noqa: BLE001
        return "template"
    return "claude"


def _facts(
    scenario: ScenarioInput, result: ScenarioResult, dist: Optional[ScenarioDistribution]
) -> str:
    drivers = sorted(result.contributions, key=lambda c: abs(c.pct_effect), reverse=True)
    top = "; ".join(
        f"{c.driver} {c.pct_effect:+.1f}%" for c in drivers[:3] if abs(c.pct_effect) > 0.05
    )
    ci = ""
    if dist is not None:
        lo, hi = dist.pct_change_ci()
        ci = f" 80% credible interval [{lo:+.0f}%, {hi:+.0f}%]."
    return (
        f"Region={scenario.region}; fuel={scenario.fuel_type}; "
        f"baseline volume={result.base_volume:,.0f}; "
        f"projected volume={result.projected_volume:,.0f}; "
        f"net change={result.total_change_pct:+.1f}%.{ci} "
        f"Drivers: {top or 'no material drivers (all changes ~0)'}."
    )


def _template_narrative(
    scenario: ScenarioInput, result: ScenarioResult, dist: Optional[ScenarioDistribution]
) -> str:
    direction = (
        "rise"
        if result.total_change_pct > 0.5
        else ("fall" if result.total_change_pct < -0.5 else "stay broadly flat")
    )
    drivers = sorted(result.contributions, key=lambda c: abs(c.pct_effect), reverse=True)
    material = [c for c in drivers if abs(c.pct_effect) > 0.05][:2]
    driver_txt = (
        " The move is driven mainly by "
        + " and ".join(f"**{c.driver.lower()}** ({c.pct_effect:+.1f}%)" for c in material)
        + "."
        if material
        else " No single driver moves materially in this scenario."
    )
    ci_txt = ""
    if dist is not None:
        lo, hi = dist.pct_change_ci()
        ci_txt = (
            f" Accounting for elasticity uncertainty, the 80% credible interval is "
            f"**[{lo:+.0f}%, {hi:+.0f}%]** — the planning range to budget against."
        )
    return (
        f"In **{scenario.region}**, demand for **{scenario.fuel_type}** BMWs is "
        f"projected to **{direction}** by **{result.total_change_pct:+.1f}%** under "
        f"this scenario ({result.base_volume:,.0f} → {result.projected_volume:,.0f} "
        f"units).{driver_txt}{ci_txt} *This is a labelled what-if simulation grounded "
        f"in literature elasticities and real macro data — not a forecast from the "
        f"historical dataset.*"
    )


def narrate(
    scenario: ScenarioInput,
    result: ScenarioResult,
    dist: Optional[ScenarioDistribution] = None,
) -> str:
    """Return an executive narrative for the scenario (Claude if available, else template)."""
    if narration_source() == "claude":
        try:
            import anthropic

            client = anthropic.Anthropic()
            facts = _facts(scenario, result, dist)
            msg = client.messages.create(
                model=_CLAUDE_MODEL,
                max_tokens=280,
                system=(
                    "You are a concise BMW strategy analyst. Write a 2-3 sentence "
                    "executive summary of the scenario using ONLY the numbers given. "
                    "Do not invent figures. End by noting it is a what-if simulation, "
                    "not a forecast."
                ),
                messages=[{"role": "user", "content": facts}],
            )
            return msg.content[0].text.strip()
        except Exception:  # noqa: BLE001 — any failure falls back to the template
            pass
    return _template_narrative(scenario, result, dist)

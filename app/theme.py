"""Theme helpers for the Streamlit dashboard (luxury dark + champagne gold)."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- Brand palette -------------------------------------------------------- #
OBSIDIAN = "#121212"
PANEL = "#1b1b1d"
GOLD = "#D4AF37"
GOLD_SOFT = "#E8D69A"
PLATINUM = "#EDEDED"
MUTED = "#9A9A9A"

#: Ordered categorical palette for charts (gold-led, BMW-luxe).
BRAND_SEQUENCE = [
    "#D4AF37",
    "#8FA9C7",
    "#C9A27E",
    "#7FB3A6",
    "#B98CC0",
    "#D98C7A",
    "#9DB87F",
    "#C7C7C7",
]

_ASSETS = Path(__file__).parent / "assets"


def _register_template() -> None:
    """Register a Plotly template matching the dark-gold identity."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Lato, Helvetica, Arial", color=PLATINUM, size=13),
        title=dict(font=dict(family="Playfair Display, serif", color=GOLD_SOFT, size=18)),
        colorway=BRAND_SEQUENCE,
        xaxis=dict(gridcolor="#2e2e33", zerolinecolor="#2e2e33", linecolor="#3a3a40"),
        yaxis=dict(gridcolor="#2e2e33", zerolinecolor="#2e2e33", linecolor="#3a3a40"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    pio.templates["bmw_luxe"] = tpl


def apply_theme() -> None:
    """Inject the CSS and register the Plotly template (idempotent)."""
    css = (_ASSETS / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    _register_template()
    pio.templates.default = "plotly_dark+bmw_luxe"


def hero(title: str, subtitle: str) -> None:
    """Render the branded hero header."""
    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-title">{title}</p>
            <p class="hero-sub">{subtitle}</p>
            <hr class="gold-rule"/>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, delta: str = "") -> None:
    """Render a single luxury KPI card."""
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def provenance_badge(source: str) -> str:
    """Return an HTML badge string for an API provenance value."""
    cls = "badge-live" if source == "live" else "badge-mock"
    return f'<span class="badge {cls}">{source.upper()}</span>'

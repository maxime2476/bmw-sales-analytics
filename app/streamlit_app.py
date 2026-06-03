"""BMW Luxury Sales Analytics — premium Streamlit dashboard.

A decision-support cockpit over 15 years of BMW sales, pairing honest analytics
(econometrics, ML/DL benchmarks, SHAP) with a forward-looking Scenario Simulator.

Launch with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the src/ package importable when running without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import data_access as da  # noqa: E402
from theme import GOLD, GOLD_SOFT, apply_theme, hero, kpi, provenance_badge  # noqa: E402

from bmw_sales.config import REPORTS_DIR, SCHEMA  # noqa: E402
from bmw_sales.econometrics.ols_models import (  # noqa: E402
    hedonic_price_model,
    price_elasticity,
    prove_leakage,
)
from bmw_sales.explainability.shap_analysis import compute_shap  # noqa: E402
from bmw_sales.simulation.scenario import (  # noqa: E402
    ElasticityAssumptions,
    ScenarioInput,
    macro_defaults,
    simulate,
)

st.set_page_config(
    page_title="BMW Luxury Sales Analytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# --------------------------------------------------------------------------- #
# Cached analytics
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Estimating econometric models…")
def _econometrics() -> dict:
    df = da.get_raw()
    elasticity = price_elasticity(df)
    hedonic = hedonic_price_model(df)
    leak = prove_leakage(df)
    return {
        "elasticity": elasticity.elasticity,
        "elasticity_p": elasticity.p_value,
        "elasticity_text": elasticity.interpretation,
        "hedonic_r2": hedonic.r_squared,
        "leak_threshold": leak.threshold,
        "leak_accuracy": leak.accuracy_from_threshold,
    }


@st.cache_data(show_spinner=False)
def _model_metrics() -> dict:
    path = REPORTS_DIR / "model_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.markdown("## ◆ Filters")
    regions = st.sidebar.multiselect(
        "Region", sorted(df[SCHEMA.REGION].unique()), default=list(df[SCHEMA.REGION].unique())
    )
    fuels = st.sidebar.multiselect(
        "Fuel type",
        sorted(df[SCHEMA.FUEL_TYPE].unique()),
        default=list(df[SCHEMA.FUEL_TYPE].unique()),
    )
    yr = df[SCHEMA.YEAR]
    year_range = st.sidebar.slider(
        "Year", int(yr.min()), int(yr.max()), (int(yr.min()), int(yr.max()))
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Honest-analytics build. External data runs offline by default "
        "(deterministic mocks); see the Data Integrity tab."
    )
    return {"regions": regions, "fuels": fuels, "year_range": year_range}


def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    lo, hi = f["year_range"]
    return df[
        df[SCHEMA.REGION].isin(f["regions"])
        & df[SCHEMA.FUEL_TYPE].isin(f["fuels"])
        & df[SCHEMA.YEAR].between(lo, hi)
    ]


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_overview(df: pd.DataFrame, provenance: dict[str, str]) -> None:
    c = st.columns(5)
    electrified = df[SCHEMA.FUEL_TYPE].isin(["Hybrid", "Electric"]).mean() * 100
    with c[0]:
        kpi("Records", f"{len(df):,}")
    with c[1]:
        kpi("Avg list price", f"${df[SCHEMA.PRICE_USD].mean():,.0f}")
    with c[2]:
        kpi("Total volume", f"{df[SCHEMA.SALES_VOLUME].sum():,.0f}")
    with c[3]:
        kpi("Electrified mix", f"{electrified:.1f}%")
    with c[4]:
        kpi("Models", f"{df[SCHEMA.MODEL].nunique()}")

    st.markdown("### Market structure")
    left, right = st.columns(2)
    with left:
        by_region = (
            df.groupby(SCHEMA.REGION, observed=True)[SCHEMA.SALES_VOLUME]
            .sum()
            .sort_values(ascending=True)
            .reset_index()
        )
        fig = px.bar(
            by_region,
            x=SCHEMA.SALES_VOLUME,
            y=SCHEMA.REGION,
            orientation="h",
            title="Total sales volume by region",
        )
        fig.update_traces(marker_color=GOLD)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        mix = (
            df.groupby([SCHEMA.YEAR, SCHEMA.FUEL_TYPE], observed=True)[SCHEMA.SALES_VOLUME]
            .sum()
            .reset_index()
        )
        fig = px.area(
            mix,
            x=SCHEMA.YEAR,
            y=SCHEMA.SALES_VOLUME,
            color=SCHEMA.FUEL_TYPE,
            title="Fuel-type volume mix over time",
        )
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x=SCHEMA.PRICE_USD, nbins=40, title="List-price distribution")
        fig.update_traces(marker_color=GOLD_SOFT)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        by_model = (
            df.groupby(SCHEMA.MODEL, observed=True)[SCHEMA.PRICE_USD]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(by_model, x=SCHEMA.MODEL, y=SCHEMA.PRICE_USD, title="Average price by model")
        fig.update_traces(marker_color=GOLD)
        st.plotly_chart(fig, use_container_width=True)

    badges = " ".join(f"{k} {provenance_badge(v)}" for k, v in provenance.items())
    st.markdown(f"**External data sources:** {badges}", unsafe_allow_html=True)


def tab_integrity() -> None:
    report = da.get_integrity()
    st.markdown("### Data Integrity Report")
    st.caption(
        "Senior principle: test the data before trusting it. Every verdict below "
        "is computed from the raw 50k rows."
    )
    findings = pd.DataFrame(
        [{"Check": f.title, "Verdict": f.verdict, "Evidence": f.detail} for f in report.findings]
    )
    st.dataframe(findings, use_container_width=True, hide_index=True)

    left, right = st.columns([1.1, 1])
    with left:
        fig = px.imshow(
            report.numeric_corr,
            text_auto=".3f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Numeric correlation matrix (≈0 everywhere ⇒ no signal)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Target leakage")
        st.metric("Implied threshold", f"Sales_Volume ≥ {report.leakage_threshold:,}")
        st.markdown(
            "<span class='small-note'>`Sales_Classification` is a deterministic "
            "threshold on `Sales_Volume` — perfectly separable, so it must be "
            "excluded as a feature.</span>",
            unsafe_allow_html=True,
        )


def tab_econometrics() -> None:
    econ = _econometrics()
    st.markdown("### Explanatory econometrics")
    c = st.columns(3)
    with c[0]:
        kpi("Price elasticity", f"{econ['elasticity']:+.3f}", f"p = {econ['elasticity_p']:.2f}")
    with c[1]:
        kpi("Hedonic R²", f"{econ['hedonic_r2']:.4f}", "OLS · HC3 robust SE")
    with c[2]:
        kpi("Leakage accuracy", f"{econ['leak_accuracy']:.3f}", f"≥ {econ['leak_threshold']:,}")
    st.info(f"**Price elasticity of demand:** {econ['elasticity_text']}", icon="📉")
    st.markdown(
        "- A real luxury market would show a negative, significant price elasticity "
        "and a high hedonic R². Here both are ~0 — pricing power is **unobservable "
        "in this data**, so list-price decisions must rely on external benchmarks.\n"
        "- Detailed coefficient tables: `reports/econometric_analysis.md`."
    )


def tab_models() -> None:
    metrics = _model_metrics()
    st.markdown("### Model benchmark — honest results")
    if not metrics:
        st.warning("Run `make pipeline` to generate the benchmark metrics.")
        return
    reg = pd.DataFrame(metrics["regression"]).T
    clf = pd.DataFrame(metrics["classification_leakage_free"]).T
    leak = pd.DataFrame(metrics["classification_with_leakage"]).T

    left, right = st.columns(2)
    with left:
        fig = px.bar(reg.reset_index(), x="index", y="r2", title="Regression R² (≈0 = no signal)")
        fig.update_traces(marker_color=GOLD)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        comp = (
            pd.DataFrame(
                {
                    "Leakage-free": clf["roc_auc"],
                    "With leakage": leak["roc_auc"],
                }
            )
            .reset_index()
            .melt(id_vars="index", var_name="Setup", value_name="ROC-AUC")
        )
        fig = px.bar(
            comp,
            x="index",
            y="ROC-AUC",
            color="Setup",
            barmode="group",
            title="Classification ROC-AUC: leakage-free vs leaked",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.error(
        "ROC-AUC ≈ 1.0 only appears when the leaked `Sales_Volume` is left in. "
        "That is a **red flag**, not a win — see ADR-0002.",
        icon="🚩",
    )


def tab_explainability() -> None:
    st.markdown("### Explainability — SHAP feature attributions")
    model, ds = da.get_regression_model()
    with st.spinner("Computing SHAP values…"):
        shap_res = compute_shap(model.pipeline, ds.X_test, max_rows=300)
    imp = shap_res.importance().head(15)
    fig = px.bar(
        imp.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Top features by mean |SHAP| (regression on Sales_Volume)",
    )
    fig.update_traces(marker_color=GOLD)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "SHAP magnitudes are tiny relative to the ~5,000-unit mean volume and have "
        "no stable ranking across runs — the explainer honestly confirms no feature "
        "systematically drives the prediction."
    )


def tab_simulator(df: pd.DataFrame) -> None:
    st.markdown("### Scenario Simulator")
    st.markdown(
        "<span class='small-note'>Forward-looking what-if tool — a decision-support "
        "simulation grounded in literature elasticities and real macro data, NOT a "
        "fit to the historical dataset.</span>",
        unsafe_allow_html=True,
    )

    c = st.columns(3)
    region = c[0].selectbox("Region", sorted(df[SCHEMA.REGION].unique()))
    fuel = c[1].selectbox("Fuel type", sorted(df[SCHEMA.FUEL_TYPE].unique()))
    base_volume = c[2].number_input("Baseline annual volume", 100, 20000, 5000, step=100)

    defaults = macro_defaults(region)
    st.caption(
        f"Macro defaults for {region}: inflation {defaults['inflation_pct']}% · "
        f"GDP-growth proxy {defaults['gdp_growth_pct']}% · regulation stringency "
        f"{defaults['regulation_stringency']} · fuel "
        f"${defaults['fuel_price_usd_per_litre']}/L"
    )

    s = st.columns(5)
    price_chg = s[0].slider("Δ List price %", -30, 30, 0)
    gdp_chg = s[1].slider("Δ Income %", -10, 15, int(min(max(defaults["gdp_growth_pct"], -10), 15)))
    fuel_chg = s[2].slider("Δ Fuel price %", -50, 50, 0)
    reg_chg = s[3].slider("Δ CO₂ stringency (pts)", -20, 30, 0)
    fx_chg = s[4].slider("FX depreciation %", -20, 20, 0)

    with st.expander("Elasticity assumptions (literature priors — adjustable)"):
        a = st.columns(3)
        own_price = a[0].number_input("Own-price ε", -2.0, 0.0, -0.6, step=0.1)
        income = a[1].number_input("Income ε", 0.0, 3.0, 1.5, step=0.1)
        reg_sens = a[2].number_input("Regulation per +10 pts", 0.0, 0.3, 0.08, step=0.01)

    scenario = ScenarioInput(
        region=region,
        fuel_type=fuel,
        base_volume=float(base_volume),
        price_change_pct=price_chg,
        gdp_growth_pct=gdp_chg,
        fuel_price_change_pct=fuel_chg,
        regulation_change_pts=reg_chg,
        fx_depreciation_pct=fx_chg,
    )
    result = simulate(
        scenario,
        ElasticityAssumptions(own_price=own_price, income=income, regulation_per_10pts=reg_sens),
    )

    k = st.columns(3)
    with k[0]:
        kpi("Baseline volume", f"{result.base_volume:,.0f}")
    with k[1]:
        kpi("Projected volume", f"{result.projected_volume:,.0f}")
    with k[2]:
        kpi("Net change", f"{result.total_change_pct:+.1f}%")

    # Waterfall of multiplicative contributions.
    drivers = [c.driver for c in result.contributions]
    effects = [c.pct_effect for c in result.contributions]
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative"] * len(drivers),
            x=drivers,
            y=effects,
            connector={"line": {"color": "#3a3a40"}},
            increasing={"marker": {"color": "#8fd49b"}},
            decreasing={"marker": {"color": "#d98c7a"}},
            totals={"marker": {"color": GOLD}},
        )
    )
    fig.update_layout(title="Demand drivers — % contribution to projected volume")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    hero("BMW Luxury Sales Analytics", "Econometrics · Machine Learning · Decision Intelligence")
    raw = da.get_raw()
    _, provenance = da.get_enriched()
    filters = _sidebar(raw)
    fdf = _apply_filters(raw, filters)
    if fdf.empty:
        st.warning("No rows match the current filters.")
        return

    tabs = st.tabs(
        [
            "Executive Overview",
            "Data Integrity",
            "Econometrics",
            "ML Benchmark",
            "Explainability (SHAP)",
            "Scenario Simulator",
        ]
    )
    with tabs[0]:
        tab_overview(fdf, provenance)
    with tabs[1]:
        tab_integrity()
    with tabs[2]:
        tab_econometrics()
    with tabs[3]:
        tab_models()
    with tabs[4]:
        tab_explainability()
    with tabs[5]:
        tab_simulator(raw)

    st.markdown(
        "<hr/><span class='small-note'>© Maxime GOURGUECHON — built with an "
        "honest-analytics methodology. See docs/adr for decisions.</span>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

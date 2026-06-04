"""Explanatory econometrics for the BMW market.

The goal here is *inference*, not prediction: quantify and test economic
relationships with proper standard errors and diagnostics. Consistent with the
Data Integrity Report (ADR-0002), we expect most effects to be statistically
indistinguishable from zero — and we report that honestly, with the rigour
(robust SE, VIF, F-tests) a reviewer expects, rather than hiding it.

Models
------
- **Hedonic price model** — decompose ``Price_USD`` into attribute contributions.
- **Demand model** — regress ``Sales_Volume`` on price and external macro drivers.
- **Price elasticity** — log-log demand specification (coefficient = elasticity).
- **Leakage proof** — formally demonstrate ``Sales_Classification`` is a
  deterministic threshold on ``Sales_Volume``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from bmw_sales.config import SCHEMA
from bmw_sales.features.engineering import add_engineered_features


@dataclass
class RegressionSummary:
    """Compact, presentation-ready summary of an OLS fit."""

    name: str
    formula: str
    n_obs: int
    r_squared: float
    adj_r_squared: float
    f_pvalue: float
    cov_type: str
    coefficients: pd.DataFrame  # index: term; cols: coef, std_err, p_value, ci_low, ci_high
    significant_terms: list[str] = field(default_factory=list)

    @property
    def has_signal(self) -> bool:
        """Whether the model explains a non-trivial share of variance."""
        return self.r_squared >= 0.05 and self.f_pvalue < 0.05


def _summarise(name: str, formula: str, result) -> RegressionSummary:
    """Convert a fitted statsmodels result into a :class:`RegressionSummary`."""
    conf = result.conf_int()
    coefs = pd.DataFrame(
        {
            "coef": result.params,
            "std_err": result.bse,
            "p_value": result.pvalues,
            "ci_low": conf[0],
            "ci_high": conf[1],
        }
    )
    significant = [term for term, p in result.pvalues.items() if term != "Intercept" and p < 0.05]
    return RegressionSummary(
        name=name,
        formula=formula,
        n_obs=int(result.nobs),
        r_squared=float(result.rsquared),
        adj_r_squared=float(result.rsquared_adj),
        f_pvalue=float(result.f_pvalue),
        cov_type=str(result.cov_type),
        coefficients=coefs.round(5),
        significant_terms=significant,
    )


# --------------------------------------------------------------------------- #
# 1. Hedonic price model
# --------------------------------------------------------------------------- #
def hedonic_price_model(df: pd.DataFrame) -> RegressionSummary:
    """Hedonic OLS decomposing (log) price into attribute contributions.

    Uses HC3 heteroskedasticity-robust standard errors. In a real luxury market
    the model dummies and engine size would dominate; here we test whether that
    holds for this dataset.
    """
    data = add_engineered_features(df)
    formula = (
        "log_price ~ C(Model) + C(Region) + C(Fuel_Type) + C(Transmission) "
        "+ Engine_Size_L + vehicle_age + log_mileage"
    )
    result = smf.ols(formula, data=data).fit(cov_type="HC3")
    return _summarise("Hedonic price model", formula, result)


# --------------------------------------------------------------------------- #
# 2. Demand model (with external macro drivers if present)
# --------------------------------------------------------------------------- #
def demand_model(df: pd.DataFrame) -> RegressionSummary:
    """OLS of sales volume on price and (when available) macro drivers.

    If the frame has been enriched (``inflation_pct``, ``gdp_per_capita_usd``,
    ``price_usd_per_litre``), those regressors are included to test whether
    macro context explains demand.
    """
    data = add_engineered_features(df)
    terms = ["log_price", "vehicle_age", "C(Region)", "C(Fuel_Type)"]
    for macro in ("inflation_pct", "gdp_per_capita_usd", "price_usd_per_litre"):
        if macro in data.columns and data[macro].notna().any():
            terms.append(macro)
    formula = f"{SCHEMA.SALES_VOLUME} ~ " + " + ".join(terms)
    result = smf.ols(formula, data=data).fit(cov_type="HC3")
    return _summarise("Demand model", formula, result)


# --------------------------------------------------------------------------- #
# 3. Price elasticity of demand (log-log)
# --------------------------------------------------------------------------- #
@dataclass
class ElasticityResult:
    """Point estimate and CI for the price elasticity of demand."""

    elasticity: float
    std_err: float
    p_value: float
    ci_low: float
    ci_high: float
    n_obs: int

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def interpretation(self) -> str:
        if not self.is_significant:
            return (
                "Not statistically distinguishable from zero — no measurable price "
                "sensitivity in this dataset (consistent with a signal-free DGP)."
            )
        sign = "elastic" if abs(self.elasticity) > 1 else "inelastic"
        return f"Demand is {sign} (elasticity {self.elasticity:+.3f})."


def price_elasticity(df: pd.DataFrame) -> ElasticityResult:
    """Estimate price elasticity via a log-log demand regression.

    ``log(Sales_Volume) = a + e * log(Price) + controls``; the coefficient ``e``
    is the elasticity. Uses HC3 robust SE.
    """
    data = add_engineered_features(df)
    data = data[data[SCHEMA.SALES_VOLUME] > 0].copy()
    data["log_volume"] = np.log(data[SCHEMA.SALES_VOLUME].astype(float))
    formula = "log_volume ~ log_price + vehicle_age + C(Region)"
    result = smf.ols(formula, data=data).fit(cov_type="HC3")
    conf = result.conf_int()
    return ElasticityResult(
        elasticity=float(result.params["log_price"]),
        std_err=float(result.bse["log_price"]),
        p_value=float(result.pvalues["log_price"]),
        ci_low=float(conf.loc["log_price", 0]),
        ci_high=float(conf.loc["log_price", 1]),
        n_obs=int(result.nobs),
    )


# --------------------------------------------------------------------------- #
# 4. Multicollinearity diagnostic (VIF)
# --------------------------------------------------------------------------- #
def vif_table(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Variance Inflation Factors for the numeric regressors.

    VIF > 10 signals problematic multicollinearity. Demonstrates diagnostic
    rigour even when (as here) the regressors are mutually independent.
    """
    data = add_engineered_features(df)
    cols = columns or [
        SCHEMA.ENGINE_SIZE_L,
        SCHEMA.MILEAGE_KM,
        "vehicle_age",
        "log_price",
        "log_mileage",
        "price_per_litre_engine",
    ]
    x = data[cols].astype(float).dropna()
    x = x.assign(_const=1.0)
    rows = []
    for i, col in enumerate(x.columns):
        if col == "_const":
            continue
        rows.append({"feature": col, "VIF": round(variance_inflation_factor(x.values, i), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Target-leakage proof
# --------------------------------------------------------------------------- #
@dataclass
class LeakageProof:
    """Evidence that the classification label is a deterministic threshold."""

    threshold: int
    high_min: int
    low_max: int
    separable: bool
    accuracy_from_threshold: float

    @property
    def verdict(self) -> str:
        return (
            f"CONFIRMED leakage: Sales_Classification == 'High' ⟺ "
            f"Sales_Volume ≥ {self.threshold} (perfectly separable, "
            f"threshold rule accuracy = {self.accuracy_from_threshold:.4f})."
            if self.separable
            else "No deterministic threshold relationship detected."
        )


def prove_leakage(df: pd.DataFrame) -> LeakageProof:
    """Formally demonstrate the classification target's leakage.

    Finds the implied threshold from the High/Low volume ranges and verifies a
    simple ``Volume ≥ threshold`` rule reproduces the label exactly.
    """
    vol = df[SCHEMA.SALES_VOLUME].astype(int)
    cls = df[SCHEMA.SALES_CLASSIFICATION].astype(str)
    high = vol[cls == "High"]
    low = vol[cls == "Low"]
    separable = bool(high.min() > low.max())
    threshold = int(high.min())
    predicted = np.where(vol >= threshold, "High", "Low")
    accuracy = float((predicted == cls.to_numpy()).mean())
    return LeakageProof(
        threshold=threshold,
        high_min=int(high.min()),
        low_max=int(low.max()),
        separable=separable,
        accuracy_from_threshold=accuracy,
    )

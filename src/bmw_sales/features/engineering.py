"""Feature engineering shared by the econometric and ML pipelines.

Adds vehicle age, usage intensity, electrification and premium-tier flags, and a
couple of log transforms. The same frame feeds statsmodels and the boosters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bmw_sales.config import SCHEMA

#: Reference year used to derive vehicle age (latest year in the dataset).
REFERENCE_YEAR: int = 2024

#: Models considered top-tier / performance luxury (domain knowledge).
_PREMIUM_MODELS: frozenset[str] = frozenset({"7 Series", "M5", "M3", "i8", "X6", "X5"})

#: Fuel types representing the electrified transition.
_ELECTRIFIED_FUELS: frozenset[str] = frozenset({"Hybrid", "Electric"})

#: Engineered numeric columns added by :func:`add_engineered_features`.
ENGINEERED_NUMERIC: tuple[str, ...] = (
    "vehicle_age",
    "mileage_per_year",
    "price_per_litre_engine",
    "log_price",
    "log_mileage",
)
ENGINEERED_CATEGORICAL: tuple[str, ...] = (
    "is_electrified",
    "is_premium_tier",
)


def add_engineered_features(
    df: pd.DataFrame, *, reference_year: int = REFERENCE_YEAR
) -> pd.DataFrame:
    """Return a copy of ``df`` with domain-informed engineered features.

    Parameters
    ----------
    df:
        A frame containing the canonical raw columns.
    reference_year:
        Year used as "today" when computing vehicle age.

    Returns
    -------
    pandas.DataFrame
        ``df`` plus :data:`ENGINEERED_NUMERIC` and :data:`ENGINEERED_CATEGORICAL`.
    """
    out = df.copy()

    age = (reference_year - out[SCHEMA.YEAR].astype(int)).clip(lower=0)
    out["vehicle_age"] = age
    # Avoid divide-by-zero for current-year vehicles (age 0 -> treat as 1).
    out["mileage_per_year"] = out[SCHEMA.MILEAGE_KM].astype(float) / age.replace(0, 1)
    out["price_per_litre_engine"] = out[SCHEMA.PRICE_USD].astype(float) / out[
        SCHEMA.ENGINE_SIZE_L
    ].astype(float)
    # Log transforms (stabilise scale; enable elasticity interpretation).
    out["log_price"] = np.log(out[SCHEMA.PRICE_USD].astype(float))
    out["log_mileage"] = np.log1p(out[SCHEMA.MILEAGE_KM].astype(float))

    out["is_electrified"] = out[SCHEMA.FUEL_TYPE].astype(str).isin(_ELECTRIFIED_FUELS).astype(int)
    out["is_premium_tier"] = out[SCHEMA.MODEL].astype(str).isin(_PREMIUM_MODELS).astype(int)
    return out


def feature_columns(*, include_leakage: bool = False) -> dict[str, list[str]]:
    """Return the modelling feature sets (categorical vs numeric).

    Parameters
    ----------
    include_leakage:
        If ``True``, includes ``Sales_Volume`` in the numeric set - used *only*
        to demonstrate target leakage for the classification task (see ADR-0002).
    """
    categorical = list(SCHEMA.CATEGORICAL) + list(ENGINEERED_CATEGORICAL)
    numeric = [
        SCHEMA.YEAR,
        SCHEMA.ENGINE_SIZE_L,
        SCHEMA.MILEAGE_KM,
        SCHEMA.PRICE_USD,
        *ENGINEERED_NUMERIC,
    ]
    if include_leakage:
        numeric.append(SCHEMA.SALES_VOLUME)
    return {"categorical": categorical, "numeric": numeric}

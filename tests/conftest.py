"""Shared pytest fixtures.

Tests run fully offline (mock APIs) and on a small synthetic frame for speed and
determinism, with a couple of integration tests that touch the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmw_sales.config import SCHEMA


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force offline mode so no test ever hits the network."""
    monkeypatch.setenv("BMW_OFFLINE_MODE", "true")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small, schema-valid synthetic BMW frame with a built-in leaked label."""
    rng = np.random.default_rng(7)
    n = 1500
    df = pd.DataFrame(
        {
            SCHEMA.MODEL: rng.choice(["X5", "M5", "3 Series", "i8"], n),
            SCHEMA.YEAR: rng.integers(2010, 2025, n),
            SCHEMA.REGION: rng.choice(["Europe", "Asia", "Africa"], n),
            SCHEMA.COLOR: rng.choice(["Black", "White", "Red"], n),
            SCHEMA.FUEL_TYPE: rng.choice(["Petrol", "Diesel", "Hybrid", "Electric"], n),
            SCHEMA.TRANSMISSION: rng.choice(["Manual", "Automatic"], n),
            SCHEMA.ENGINE_SIZE_L: rng.uniform(1.5, 5.0, n).round(1),
            SCHEMA.MILEAGE_KM: rng.integers(3, 200_000, n),
            SCHEMA.PRICE_USD: rng.integers(30_000, 120_000, n),
            SCHEMA.SALES_VOLUME: rng.integers(100, 9999, n),
        }
    )
    # Pin two sentinel rows so the leakage threshold is deterministically 7000
    # (smallest possible High volume) regardless of the random draw.
    df.loc[0, SCHEMA.SALES_VOLUME] = 7000
    df.loc[1, SCHEMA.SALES_VOLUME] = 6999
    # Reproduce the dataset's leakage rule exactly: High iff volume >= 7000.
    df[SCHEMA.SALES_CLASSIFICATION] = np.where(df[SCHEMA.SALES_VOLUME] >= 7000, "High", "Low")
    return df

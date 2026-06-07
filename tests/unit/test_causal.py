"""Tests for the causal (backdoor-adjustment) price→demand analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bmw_sales.config import SCHEMA
from bmw_sales.econometrics.causal import build_report, estimate_price_effect


def test_no_causal_effect_on_real_data(sample_df: pd.DataFrame) -> None:
    res = estimate_price_effect(sample_df)
    assert not res.is_causal
    assert res.adjusted_pvalue > 0.05


def test_detects_injected_price_effect(sample_df: pd.DataFrame) -> None:
    """When demand is built to depend strongly on price, adjustment must find it."""
    df = sample_df.copy()
    rng = np.random.default_rng(0)
    # Strong negative price dependence + noise -> a real causal effect.
    vol = 9000 - 0.05 * df[SCHEMA.PRICE_USD].astype(float) + rng.normal(0, 50, len(df))
    df[SCHEMA.SALES_VOLUME] = np.clip(vol, 100, 9999).astype(int)
    res = estimate_price_effect(df)
    assert res.adjusted_effect < 0
    assert res.is_causal


def test_report_builds(sample_df: pd.DataFrame) -> None:
    md = build_report(estimate_price_effect(sample_df))
    assert "Causal Analysis" in md
    assert "Backdoor-adjusted" in md

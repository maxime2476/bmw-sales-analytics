"""Tests for the statistical signal-audit module."""

from __future__ import annotations

import pandas as pd

from bmw_sales.audit.control import make_signal_bearing_target, run_control
from bmw_sales.audit.signal_tests import (
    chi2_independence,
    permutation_test,
    uniformity_tests,
)
from bmw_sales.config import SCHEMA


def test_control_validates_pipeline(sample_df: pd.DataFrame) -> None:
    """The pipeline must recover a known synthetic signal (sanity of the method)."""
    res = run_control(sample_df, model_name="LightGBM", sample=1200)
    assert res.r2_synthetic > 0.4
    assert res.pipeline_validated


def test_signal_bearing_target_depends_on_features(sample_df: pd.DataFrame) -> None:
    y = make_signal_bearing_target(sample_df)
    # Premium-tier vehicles should, on average, get a higher synthetic demand.
    is_premium = sample_df[SCHEMA.MODEL].isin(["X5", "M5"])
    assert y[is_premium].mean() > y[~is_premium].mean()


def test_permutation_detects_injected_signal(sample_df: pd.DataFrame) -> None:
    """With a real feature->target relationship, the test should flag SIGNAL."""
    signal_df = sample_df.copy()
    signal_df[SCHEMA.SALES_VOLUME] = make_signal_bearing_target(sample_df).to_numpy()
    res = permutation_test(signal_df, "regression", n_permutations=8, sample=1200)
    assert res.observed > res.null_mean
    assert res.p_value < 0.2


def test_permutation_structure(sample_df: pd.DataFrame) -> None:
    res = permutation_test(sample_df, "classification", n_permutations=6, sample=1000)
    assert len(res.null_scores) == 6
    assert 0.0 < res.p_value <= 1.0
    assert res.metric == "ROC-AUC"


def test_uniformity_and_chi2(sample_df: pd.DataFrame) -> None:
    unif = uniformity_tests(sample_df)
    assert {u.feature for u in unif} == set(SCHEMA.NUMERIC)
    p = chi2_independence(sample_df, SCHEMA.REGION, SCHEMA.FUEL_TYPE)
    assert 0.0 <= p <= 1.0

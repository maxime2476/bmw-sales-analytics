"""Tests for the predictive-capability demonstration."""

from __future__ import annotations

import pandas as pd

from bmw_sales.audit.capability import build_report, demonstrate


def test_pipeline_is_skilful_on_signal(sample_df: pd.DataFrame) -> None:
    """On a signal-bearing target the pipeline must achieve real, validated skill."""
    res = demonstrate(sample_df, sample=1400, cv=3)
    assert res.is_skilful
    assert res.cv_mean > 0.5
    # CV and held-out agree -> not overfitting.
    assert abs(res.cv_mean - res.test_r2) < 0.25
    assert res.cv_std < 0.2


def test_learning_curve_improves(sample_df: pd.DataFrame) -> None:
    res = demonstrate(sample_df, sample=1400, cv=3)
    lc = res.learning_curve_test
    assert lc[-1] >= lc[0]  # more data -> at least as good
    assert len(res.learning_curve_sizes) == len(lc)


def test_report_mentions_validation(sample_df: pd.DataFrame) -> None:
    res = demonstrate(sample_df, sample=1400, cv=3)
    md = build_report(res)
    assert "Cross-validated R" in md
    assert "Learning curve" in md
    assert "recovers the TRUE drivers" in md

"""Tests for split-conformal prediction intervals."""

from __future__ import annotations

import pandas as pd

from bmw_sales.models.conformal import build_report, conformal_intervals


def test_coverage_guarantee_holds(sample_df: pd.DataFrame) -> None:
    """Conformal intervals should achieve ~ (1 - alpha) coverage regardless."""
    res = conformal_intervals(sample_df, signal_bearing=False, alpha=0.1, sample=1400)
    assert 0.80 <= res.coverage <= 1.0


def test_real_intervals_are_wide_and_honest(sample_df: pd.DataFrame) -> None:
    """On signal-free data the interval should span most of the target range."""
    res = conformal_intervals(sample_df, signal_bearing=False, sample=1400)
    assert res.relative_width > 0.6
    assert not res.is_informative


def test_signal_intervals_are_tighter(sample_df: pd.DataFrame) -> None:
    """On signal-bearing data the interval should collapse and be informative."""
    real = conformal_intervals(sample_df, signal_bearing=False, sample=1400)
    signal = conformal_intervals(sample_df, signal_bearing=True, sample=1400)
    assert signal.relative_width < real.relative_width
    assert signal.is_informative
    assert 0.80 <= signal.coverage <= 1.0


def test_report_builds(sample_df: pd.DataFrame) -> None:
    real = conformal_intervals(sample_df, signal_bearing=False, sample=1200)
    signal = conformal_intervals(sample_df, signal_bearing=True, sample=1200)
    md = build_report(real, signal)
    assert "Conformal Prediction" in md
    assert "coverage" in md.lower()

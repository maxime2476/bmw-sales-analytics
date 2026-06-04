"""Tests for the data-integrity analysis."""

from __future__ import annotations

import pandas as pd

from bmw_sales.data.validation import analyse


def test_detects_leakage(sample_df: pd.DataFrame) -> None:
    report = analyse(sample_df)
    assert report.leakage_detected is True
    assert report.leakage_threshold == 7000


def test_reports_leakage_verdict(sample_df: pd.DataFrame) -> None:
    # The signal-free verdict on numeric/ANOVA is a large-sample property and is
    # asserted on the real 50k dataset in the integration suite; here we verify
    # the deterministic leakage verdict holds.
    report = analyse(sample_df)
    verdicts = {f.title: f.verdict for f in report.findings}
    assert verdicts["Target leakage"] == "LEAKAGE"


def test_structural_counts(sample_df: pd.DataFrame) -> None:
    report = analyse(sample_df)
    assert report.n_rows == len(sample_df)
    assert report.n_nulls == 0

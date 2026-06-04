"""Smoke tests for the markdown report builders (fast, on the small fixture)."""

from __future__ import annotations

import pandas as pd

from bmw_sales.data.validation import analyse, to_markdown
from bmw_sales.econometrics.report import build_report as econ_report
from bmw_sales.sql.report import build_report as sql_report


def test_integrity_markdown_has_sections(sample_df: pd.DataFrame) -> None:
    md = to_markdown(analyse(sample_df))
    assert "# Data Integrity Report" in md
    assert "Target leakage" in md
    assert "| Check | Verdict |" in md


def test_econometric_report_builds(sample_df: pd.DataFrame) -> None:
    md = econ_report(sample_df)
    assert "# Econometric Analysis" in md
    assert "Price elasticity of demand" in md
    assert "Target-leakage proof" in md


def test_sql_report_builds() -> None:
    md = sql_report()
    assert "# SQL Business Insights" in md
    assert "Sales volume by region" in md
    assert "Year-over-year" in md

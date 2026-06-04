"""Integration tests touching the real dataset (skipped if absent).

Run with the full suite, or exclude via ``pytest -m 'not integration'``.
"""

from __future__ import annotations

import pytest

from bmw_sales.config import RAW_DATASET_PATH, SCHEMA
from bmw_sales.data.loader import load_raw
from bmw_sales.data.validation import analyse

pytestmark = pytest.mark.integration

_HAS_DATA = RAW_DATASET_PATH.exists()
_skip = pytest.mark.skipif(not _HAS_DATA, reason="raw dataset not present")


@_skip
def test_real_dataset_shape() -> None:
    df = load_raw()
    assert df.shape == (50_000, 11)
    assert int(df.isna().sum().sum()) == 0


@_skip
def test_real_dataset_leakage_threshold_is_7000() -> None:
    report = analyse(load_raw())
    assert report.leakage_detected
    assert report.leakage_threshold == SCHEMA.CLASSIFICATION_THRESHOLD == 7000


@_skip
def test_real_dataset_has_no_numeric_signal() -> None:
    report = analyse(load_raw())
    verdicts = {f.title: f.verdict for f in report.findings}
    assert verdicts["Numeric signal (correlation)"] == "NO SIGNAL"

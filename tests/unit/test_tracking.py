"""Tests for the optional MLflow experiment-tracking helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from bmw_sales.models import tracking
from bmw_sales.models.ml_models import TrainedModel


def _model() -> TrainedModel:
    return TrainedModel(
        name="XGBoost",
        task="regression",
        pipeline=None,  # type: ignore[arg-type]  # only metadata is logged
        metrics={"r2": -0.01, "rmse": 2800.0, "mae": 2400.0},
        best_params={"model__n_estimators": 200},
    )


def test_log_models_writes_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracking, "MLRUNS_DIR", tmp_path / "mlruns")
    logged = tracking.log_models([_model()], experiment="test-exp")
    assert logged is True
    assert (tmp_path / "mlruns").exists()

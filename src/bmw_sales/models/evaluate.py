"""Evaluation metrics for the regression and classification tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass
class RegressionMetrics:
    """Held-out regression metrics."""

    r2: float
    rmse: float
    mae: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ClassificationMetrics:
    """Held-out classification metrics."""

    roc_auc: float
    accuracy: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute R², RMSE and MAE."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return RegressionMetrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=rmse,
        mae=float(mean_absolute_error(y_true, y_pred)),
    )


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> ClassificationMetrics:
    """Compute ROC-AUC, accuracy and F1 (binary)."""
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
    )

"""SHAP explainability for the tree-based models.

Provides feature attributions for a fitted sklearn ``Pipeline`` (preprocessor +
gradient-boosted tree). We explain the model in its *transformed* feature space
and map attributions back to readable feature names for the dashboard.

Because the data is signal-free (ADR-0002), SHAP magnitudes are expectedly tiny
and noisy — the explainability tab uses this honestly to show that no feature
systematically drives predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class ShapResult:
    """SHAP attributions for a sample of rows."""

    feature_names: list[str]
    values: np.ndarray  # (n_rows, n_features)
    base_value: float
    data: np.ndarray  # transformed feature matrix (n_rows, n_features)

    def importance(self) -> pd.DataFrame:
        """Mean absolute SHAP value per feature, descending."""
        mean_abs = np.abs(self.values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


def _transformed_feature_names(preprocessor: Any) -> list[str]:
    """Best-effort readable names for the ColumnTransformer output."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # noqa: BLE001 — fall back to positional names
        return []


def compute_shap(pipeline: Pipeline, x_sample: pd.DataFrame, *, max_rows: int = 500) -> ShapResult:
    """Compute SHAP values for ``x_sample`` against a fitted tree pipeline.

    Parameters
    ----------
    pipeline:
        A fitted ``Pipeline`` whose steps are ``("pre", ColumnTransformer)`` and
        ``("model", <tree model>)``.
    x_sample:
        Raw feature rows (pre-transform) to explain.
    max_rows:
        Cap the number of rows for tractable computation.
    """
    import shap

    sample = x_sample.iloc[:max_rows]
    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]

    x_trans = pre.transform(sample)
    names = _transformed_feature_names(pre) or [f"f{i}" for i in range(x_trans.shape[1])]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_trans)
    # Binary classifiers may return a list (per class) or a 3-D array; take class 1.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., -1]

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.ravel(base)[-1])

    return ShapResult(
        feature_names=list(names),
        values=shap_values,
        base_value=float(base),
        data=np.asarray(x_trans),
    )

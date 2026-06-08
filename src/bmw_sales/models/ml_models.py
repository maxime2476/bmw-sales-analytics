"""XGBoost, LightGBM and CatBoost wrappers with a small randomised search.

Each model shares the same preprocessor inside a sklearn Pipeline so the
comparison is fair and the result is serialisable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from bmw_sales.config import get_settings
from bmw_sales.models.evaluate import (
    ClassificationMetrics,
    RegressionMetrics,
    classification_metrics,
    regression_metrics,
)
from bmw_sales.models.preprocessing import Dataset, build_preprocessor

# Model & hyperparameter registry
SEED = get_settings().random_seed


def _regressors() -> dict[str, Callable[[], Any]]:
    return {
        "XGBoost": lambda: XGBRegressor(
            tree_method="hist", random_state=SEED, n_jobs=-1, verbosity=0
        ),
        "LightGBM": lambda: LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1),
        "CatBoost": lambda: CatBoostRegressor(random_seed=SEED, verbose=0),
    }


def _classifiers() -> dict[str, Callable[[], Any]]:
    return {
        "XGBoost": lambda: XGBClassifier(
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
        ),
        "LightGBM": lambda: LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
        "CatBoost": lambda: CatBoostClassifier(random_seed=SEED, verbose=0),
    }


#: Compact, model-specific search spaces (prefixed for the pipeline step "model").
PARAM_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "XGBoost": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
    },
    "LightGBM": {
        "model__n_estimators": [200, 400, 600],
        "model__num_leaves": [15, 31, 63],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
    },
    "CatBoost": {
        "model__iterations": [200, 400, 600],
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.02, 0.05, 0.1],
    },
}


@dataclass
class TrainedModel:
    """A fitted pipeline with its held-out metrics and best params."""

    name: str
    task: str
    pipeline: Pipeline
    metrics: dict[str, float]
    best_params: dict[str, Any] = field(default_factory=dict)


def _build_pipeline(estimator: Any, dataset: Dataset) -> Pipeline:
    pre = build_preprocessor(dataset.numeric, dataset.categorical)
    return Pipeline([("pre", pre), ("model", estimator)])


def train_one(
    name: str,
    dataset: Dataset,
    *,
    tune: bool = True,
    n_iter: int = 6,
    cv: int = 3,
) -> TrainedModel:
    """Train a single named model for ``dataset.task`` and evaluate on the test set."""
    factory = (_regressors() if dataset.task == "regression" else _classifiers())[name]
    pipeline = _build_pipeline(factory(), dataset)

    if tune:
        scoring = "r2" if dataset.task == "regression" else "roc_auc"
        search = RandomizedSearchCV(
            pipeline,
            PARAM_DISTRIBUTIONS[name],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=SEED,
            n_jobs=-1,
            refit=True,
        )
        search.fit(dataset.X_train, dataset.y_train)
        pipeline = search.best_estimator_
        best_params = {k: v for k, v in search.best_params_.items()}
    else:
        pipeline.fit(dataset.X_train, dataset.y_train)
        best_params = {}

    metrics = _evaluate(pipeline, dataset)
    return TrainedModel(name, dataset.task, pipeline, metrics, best_params)


def _evaluate(pipeline: Pipeline, dataset: Dataset) -> dict[str, float]:
    if dataset.task == "regression":
        pred = pipeline.predict(dataset.X_test)
        m: RegressionMetrics = regression_metrics(dataset.y_test.to_numpy(), pred)
        return m.as_dict()
    proba = pipeline.predict_proba(dataset.X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm: ClassificationMetrics = classification_metrics(dataset.y_test.to_numpy(), pred, proba)
    return cm.as_dict()


def train_all(dataset: Dataset, *, tune: bool = True, n_iter: int = 6) -> list[TrainedModel]:
    """Train XGBoost, LightGBM and CatBoost for the dataset's task."""
    names = list((_regressors() if dataset.task == "regression" else _classifiers()))
    return [train_one(n, dataset, tune=tune, n_iter=n_iter) for n in names]


def best_model(models: list[TrainedModel]) -> TrainedModel:
    """Pick the best model by the task-appropriate primary metric."""
    if not models:
        raise ValueError("No models to choose from")
    key = "r2" if models[0].task == "regression" else "roc_auc"
    return max(models, key=lambda m: m.metrics[key])

"""Optional MLflow logging for the training run.

Logs each model (params, metrics, tags) to a local mlruns/ store. If MLflow
isn't installed it just does nothing. Browse with
``mlflow ui --backend-store-uri ./mlruns``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from bmw_sales.config import PROJECT_ROOT

if TYPE_CHECKING:  # avoid a hard import cycle at runtime
    from bmw_sales.models.ml_models import TrainedModel

MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def log_models(
    models: "Iterable[TrainedModel]",
    *,
    experiment: str = "bmw-sales",
    run_group: str = "benchmark",
) -> bool:
    """Log each model's params & metrics to MLflow. Returns ``True`` if logged.

    Silently no-ops (returning ``False``) when MLflow is unavailable, so the
    training CLI never fails because of an optional tracking dependency.
    """
    try:
        import mlflow
    except Exception:  # noqa: BLE001 - optional dependency
        return False

    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment(experiment)

    for model in models:
        run_name = f"{run_group}-{model.task}-{model.name}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"task": model.task, "model": model.name, "group": run_group})
            if model.best_params:
                mlflow.log_params({k: v for k, v in model.best_params.items()})
            mlflow.log_metrics({k: float(v) for k, v in model.metrics.items()})
    return True

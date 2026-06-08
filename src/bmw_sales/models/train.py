"""End-to-end training & benchmarking orchestrator.

Trains the three gradient-boosting models for both the regression and the
(leakage-free) classification task, runs an explicit leakage demonstration,
persists the best pipelines, and writes a markdown benchmark report.

Run as a script::

    python -m bmw_sales.models.train            # full run (tuned)
    python -m bmw_sales.models.train --fast      # no tuning, quick smoke
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date

import joblib
import pandas as pd

from bmw_sales.apis.enrichment import enrich_dataset
from bmw_sales.config import MODELS_DIR, REPORTS_DIR
from bmw_sales.data.loader import load_raw
from bmw_sales.models.ml_models import TrainedModel, best_model, train_all
from bmw_sales.models.preprocessing import make_dataset
from bmw_sales.models.tracking import log_models


def _metrics_table(models: list[TrainedModel]) -> str:
    rows = [m.metrics for m in models]
    df = pd.DataFrame(rows, index=[m.name for m in models]).round(4)
    return df.to_markdown()


def _persist(model: TrainedModel, filename: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, MODELS_DIR / filename)


def run(*, tune: bool = True, n_iter: int = 6) -> str:
    """Train everything, persist artefacts and return the benchmark markdown."""
    os.environ.setdefault("BMW_OFFLINE_MODE", "true")  # reproducible inputs
    df = enrich_dataset(load_raw()).data

    # --- Regression: predict Sales_Volume ---
    reg_ds = make_dataset(df, "regression")
    reg_models = train_all(reg_ds, tune=tune, n_iter=n_iter)
    reg_best = best_model(reg_models)
    _persist(reg_best, "regression_best.joblib")

    # --- Classification: predict Sales_Classification (leakage-free) ---
    clf_ds = make_dataset(df, "classification")
    clf_models = train_all(clf_ds, tune=tune, n_iter=n_iter)
    clf_best = best_model(clf_models)
    _persist(clf_best, "classification_best.joblib")

    # --- Leakage demonstration (include Sales_Volume) ---
    leak_ds = make_dataset(df, "classification", include_leakage=True)
    leak_models = train_all(leak_ds, tune=False)

    # --- Experiment tracking (best-effort; no-op if MLflow absent) ---
    if log_models(reg_models, run_group="regression"):
        log_models(clf_models, run_group="classification")
        log_models(leak_models, run_group="leakage-demo")
        print("Logged runs to MLflow (./mlruns)")

    # --- Persist a machine-readable metrics summary ---
    summary = {
        "regression": {m.name: m.metrics for m in reg_models},
        "classification_leakage_free": {m.name: m.metrics for m in clf_models},
        "classification_with_leakage": {m.name: m.metrics for m in leak_models},
        "best": {
            "regression": reg_best.name,
            "classification": clf_best.name,
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return _build_markdown(reg_models, clf_models, leak_models, reg_best, clf_best)


def _build_markdown(
    reg_models: list[TrainedModel],
    clf_models: list[TrainedModel],
    leak_models: list[TrainedModel],
    reg_best: TrainedModel,
    clf_best: TrainedModel,
) -> str:
    leak_best = best_model(leak_models)
    return (
        f"# Model Benchmark - BMW Sales (2010–2024)\n\n"
        f"*Generated: {date.today().isoformat()} · Author: Maxime GOURGUECHON*\n\n"
        f"> Gradient-boosting benchmark on the API-enriched dataset. Metrics are "
        f"held-out (test split). Read with the "
        f"[Data Integrity Report](data_integrity_report.md).\n\n"
        f"## Executive summary\n\n"
        f"Three tuned gradient-boosting families (XGBoost, LightGBM, CatBoost) are "
        f"benchmarked. As predicted by the data-integrity analysis, **regression "
        f"R² ≈ 0** and **leakage-free classification ROC-AUC ≈ 0.5**: there is no "
        f"signal to learn. The contrast with the leakage run (ROC-AUC ≈ 1.0) "
        f"validates that our honest setup correctly excludes the leaked column.\n\n"
        f"## 1. Regression - target `Sales_Volume`\n\n"
        f"{_metrics_table(reg_models)}\n\n"
        f"**Best:** {reg_best.name} (R² = {reg_best.metrics['r2']:.4f}). "
        f"An R² at/below zero means the models do not beat predicting the mean - "
        f"the honest, expected outcome on noise.\n\n"
        f"## 2. Classification - target `Sales_Classification` (leakage-free)\n\n"
        f"{_metrics_table(clf_models)}\n\n"
        f"**Best:** {clf_best.name} (ROC-AUC = {clf_best.metrics['roc_auc']:.4f}). "
        f"ROC-AUC ≈ 0.5 confirms no discriminative signal once the leaked "
        f"`Sales_Volume` is correctly removed.\n\n"
        f"## 3. Leakage demonstration - `Sales_Volume` left in as a feature\n\n"
        f"{_metrics_table(leak_models)}\n\n"
        f"**Result:** {leak_best.name} reaches ROC-AUC = "
        f"{leak_best.metrics['roc_auc']:.4f}. This near-perfect score is **not a "
        f"success** - it is the signature of target leakage (the label is a "
        f"deterministic threshold on this feature) and is shown here only to make "
        f"the failure mode explicit.\n\n"
        f"## Takeaways\n\n"
        f"- Tuning cannot manufacture signal that the data does not contain.\n"
        f"- A >0.99 classification score on this data is a **red flag**, not a win.\n"
        f"- Decision value is delivered by the Scenario Simulator, not these "
        f"in-sample models - see ADR-0002.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & benchmark BMW models.")
    parser.add_argument(
        "--fast", action="store_true", help="Skip hyperparameter tuning (quick run)."
    )
    parser.add_argument("--n-iter", type=int, default=6, help="Randomised-search iters.")
    args = parser.parse_args()

    markdown = run(tune=not args.fast, n_iter=args.n_iter)
    out_path = REPORTS_DIR / "model_benchmark.md"
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Model benchmark written to {out_path}")
    print(f"Artefacts saved under {MODELS_DIR}")


if __name__ == "__main__":
    main()

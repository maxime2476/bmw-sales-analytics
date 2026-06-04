"""Formal statistical tests for *whether a dataset contains learnable signal*.

These turn the project's central claim ("the data is signal-free noise") from an
assertion into **falsifiable, transferable evidence**. The module is dataset-
agnostic: point it at any frame and it will quantify how much exploitable signal
exists, using three complementary lenses.

1. **Permutation (label-shuffle) test** — the gold standard. Train a model on the
   real labels, then on many label-shuffled copies, and compare the real held-out
   score to the null distribution. A high p-value means the real score is
   indistinguishable from chance: *no signal*.
2. **Kolmogorov–Smirnov uniformity test** — for each numeric feature, test whether
   it is drawn from a Uniform distribution (a fingerprint of synthetic data).
3. **Chi-squared independence** — test pairwise independence of categoricals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import r2_score, roc_auc_score

from bmw_sales.config import SCHEMA, get_settings
from bmw_sales.models.preprocessing import (
    Dataset,
    Task,
    build_preprocessor,
    make_dataset,
)


@dataclass
class PermutationResult:
    """Outcome of a label-permutation signal test."""

    task: str
    metric: str
    observed: float
    null_mean: float
    null_std: float
    p_value: float
    n_permutations: int
    null_scores: list[float] = field(default_factory=list)

    @property
    def has_signal(self) -> bool:
        """Signal is present only if the real score beats the null at 5%."""
        return self.p_value < 0.05


def _fit_score(dataset: Dataset, y_train: np.ndarray) -> float:
    """Fit a fast GBDT on ``y_train`` and score it on the real held-out target."""
    pre = build_preprocessor(dataset.numeric, dataset.categorical)
    x_train = pre.fit_transform(dataset.X_train)
    x_test = pre.transform(dataset.X_test)
    if dataset.task == "regression":
        model = HistGradientBoostingRegressor(random_state=0, max_iter=150)
        model.fit(x_train, y_train)
        return float(r2_score(dataset.y_test, model.predict(x_test)))
    model = HistGradientBoostingClassifier(random_state=0, max_iter=150)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(dataset.y_test, proba))


def permutation_test(
    df: pd.DataFrame, task: Task, *, n_permutations: int = 30, sample: int = 6000
) -> PermutationResult:
    """Run a label-permutation test for exploitable signal on ``task``.

    Parameters
    ----------
    df:
        Source dataset.
    task:
        ``"regression"`` (R²) or ``"classification"`` (ROC-AUC).
    n_permutations:
        Size of the null distribution (each is a full model fit).
    sample:
        Sub-sample size for tractable runtime.
    """
    seed = get_settings().random_seed
    rng = np.random.default_rng(seed)
    work = df.sample(min(sample, len(df)), random_state=seed)
    dataset = make_dataset(work, task)

    metric = "R²" if task == "regression" else "ROC-AUC"
    observed = _fit_score(dataset, dataset.y_train.to_numpy())

    y_tr = dataset.y_train.to_numpy()
    null_scores: list[float] = []
    for _ in range(n_permutations):
        permuted = rng.permutation(y_tr)
        null_scores.append(_fit_score(dataset, permuted))

    null = np.asarray(null_scores)
    # One-sided p-value: P(null >= observed). +1 smoothing avoids p=0.
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return PermutationResult(
        task=task,
        metric=metric,
        observed=observed,
        null_mean=float(null.mean()),
        null_std=float(null.std()),
        p_value=p_value,
        n_permutations=n_permutations,
        null_scores=null_scores,
    )


@dataclass
class UniformityResult:
    """KS test of one numeric feature against a fitted Uniform distribution."""

    feature: str
    ks_statistic: float
    p_value: float

    @property
    def looks_uniform(self) -> bool:
        """Cannot reject Uniform at 5% ⇒ consistent with synthetic uniform data."""
        return self.p_value > 0.05


def uniformity_tests(df: pd.DataFrame) -> list[UniformityResult]:
    """KS-test every numeric feature against Uniform(min, max)."""
    results: list[UniformityResult] = []
    for col in SCHEMA.NUMERIC:
        x = df[col].astype(float).to_numpy()
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            continue
        stat, p = stats.kstest(x, "uniform", args=(lo, hi - lo))
        results.append(UniformityResult(col, float(stat), float(p)))
    return results


def chi2_independence(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    """Return the chi-squared independence p-value between two categoricals."""
    table = pd.crosstab(df[col_a], df[col_b])
    _, p, _, _ = stats.chi2_contingency(table)
    return float(p)

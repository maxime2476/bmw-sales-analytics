"""Split-conformal prediction intervals.

Gives distribution-free intervals with ~(1-alpha) coverage regardless of model
quality. Fit on a proper-train split, take the (1-alpha) quantile of the
calibration-set absolute residuals as the half-width, then measure coverage and
width on the test split. On the signal-free data the interval ends up covering
most of the target range; on a signal-bearing target it is much tighter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from bmw_sales.config import SCHEMA, get_settings
from bmw_sales.models.preprocessing import build_preprocessor, make_dataset


@dataclass
class ConformalResult:
    """Outcome of a split-conformal experiment on one target."""

    alpha: float
    half_width: float
    coverage: float
    target_std: float
    target_range: float

    @property
    def interval_width(self) -> float:
        return 2.0 * self.half_width

    @property
    def relative_width(self) -> float:
        """Interval width as a fraction of the target's full range (0–1+)."""
        return self.interval_width / self.target_range if self.target_range else 0.0

    @property
    def is_informative(self) -> bool:
        """An interval covering < 60% of the range actually tells you something."""
        return self.relative_width < 0.6


def conformal_intervals(
    df: pd.DataFrame, *, signal_bearing: bool = False, alpha: float = 0.1, sample: int = 12000
) -> ConformalResult:
    """Compute split-conformal intervals for the regression target.

    Parameters
    ----------
    signal_bearing:
        If ``True``, replace the (signal-free) real target with the labelled
        synthetic signal-bearing target - to contrast informative vs honest-wide
        intervals.
    alpha:
        Miscoverage level; intervals target ``1 − alpha`` coverage (e.g. 0.1 → 90%).
    """
    seed = get_settings().random_seed
    work = df.sample(min(sample, len(df)), random_state=seed).copy()
    if signal_bearing:
        from bmw_sales.audit.control import make_signal_bearing_target

        work[SCHEMA.SALES_VOLUME] = make_signal_bearing_target(work).to_numpy()

    ds = make_dataset(work, "regression")

    # Carve a calibration set out of the training split.
    rng = np.random.default_rng(seed)
    n_train = len(ds.X_train)
    idx = rng.permutation(n_train)
    cut = n_train // 2
    fit_idx, cal_idx = idx[:cut], idx[cut:]

    pre = build_preprocessor(ds.numeric, ds.categorical)
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(ds.X_train.iloc[fit_idx], ds.y_train.iloc[fit_idx])

    # Nonconformity scores on the calibration set.
    cal_pred = pipe.predict(ds.X_train.iloc[cal_idx])
    scores = np.abs(ds.y_train.iloc[cal_idx].to_numpy() - cal_pred)
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    half_width = float(np.quantile(scores, q_level))

    # Empirical coverage on the held-out test split.
    test_pred = pipe.predict(ds.X_test)
    y_test = ds.y_test.to_numpy()
    coverage = float(np.mean(np.abs(y_test - test_pred) <= half_width))

    return ConformalResult(
        alpha=alpha,
        half_width=half_width,
        coverage=coverage,
        target_std=float(np.std(y_test)),
        target_range=float(y_test.max() - y_test.min()),
    )


def build_report(real: ConformalResult, signal: ConformalResult) -> str:
    """Render the conformal comparison (real vs signal-bearing) as markdown."""
    from datetime import date

    def row(label: str, r: ConformalResult) -> str:
        verdict = "informative" if r.is_informative else "honest *'I don't know'*"
        return (
            f"| {label} | {r.coverage:.0%} | ±{r.half_width:,.0f} "
            f"({r.interval_width:,.0f} wide) | {r.relative_width:.0%} of range | {verdict} |"
        )

    target = 1 - real.alpha
    return (
        f"# Conformal Prediction - calibrated, honest uncertainty\n\n"
        f"*Generated: {date.today().isoformat()} · Author: Maxime GOURGUECHON*\n\n"
        f"> Split-conformal intervals targeting **{target:.0%} coverage**. Reproduce "
        f"with `make conformal`.\n\n"
        f"| Target | Empirical coverage | Interval (half-width) | Relative width | Verdict |\n"
        f"|---|---|---|---|---|\n"
        f"{row('Real `Sales_Volume`', real)}\n"
        f"{row('Synthetic signal-bearing', signal)}\n\n"
        f"## Reading this\n\n"
        f"- Both targets achieve **≈ {target:.0%} coverage** - the conformal "
        f"guarantee holds regardless of model quality.\n"
        f"- On the **real** data the interval spans **{real.relative_width:.0%}** of "
        f"the target range: the model honestly says *'I don't know'* (there is no "
        f"signal to narrow it).\n"
        f"- On **signal-bearing** data the interval collapses to "
        f"**{signal.relative_width:.0%}** of the range - calibrated, *useful* "
        f"uncertainty.\n\n"
        f"The interval **width** is thus a principled measure of how much the data "
        f"actually permits you to say - the perfect complement to the project's "
        f"honest-analytics thesis (see the Signal Audit and Predictive Capability).\n"
    )


def main() -> None:
    import os

    from bmw_sales.config import REPORTS_DIR
    from bmw_sales.data.loader import load_raw

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    df = load_raw()
    real = conformal_intervals(df, signal_bearing=False)
    signal = conformal_intervals(df, signal_bearing=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "conformal_prediction.md"
    out.write_text(build_report(real, signal), encoding="utf-8")
    print(f"Conformal report written to {out}")
    print(
        f"     real: {real.coverage:.0%} cov, {real.relative_width:.0%} width | "
        f"signal: {signal.coverage:.0%} cov, {signal.relative_width:.0%} width"
    )


if __name__ == "__main__":
    main()

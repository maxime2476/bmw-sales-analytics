"""Positive-control experiment: prove the pipeline works when signal exists.

A null R² could mean two things: (a) the data has no signal, or (b) the pipeline
is broken. To rule out (b), we run the *identical* modelling pipeline on a
**synthetic target engineered to be a known function of the features** (clearly
labelled as such). If the pipeline recovers that signal (high R²) while scoring
~0 on the real target, the conclusion is unambiguous: **the pipeline is sound;
the real data is empty.**
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bmw_sales.config import SCHEMA, get_settings
from bmw_sales.features.engineering import add_engineered_features
from bmw_sales.models.ml_models import train_one
from bmw_sales.models.preprocessing import make_dataset

#: Per-region demand offsets used to build the synthetic, signal-bearing target.
_REGION_EFFECT: dict[str, float] = {
    "Europe": 1500,
    "North America": 1200,
    "Asia": 900,
    "Middle East": 300,
    "South America": 0,
    "Africa": -400,
}


def make_signal_bearing_target(df: pd.DataFrame, *, noise_sd: float = 400.0) -> pd.Series:
    """Construct a synthetic demand target that genuinely depends on the features.

    ``synthetic_demand = f(region, premium tier, engine, year, price, electrified)
    + Gaussian noise``. This is **not** a claim about the real world — it exists
    only to verify the pipeline can learn a relationship that is known to exist.
    """
    seed = get_settings().random_seed
    rng = np.random.default_rng(seed)
    data = add_engineered_features(df)

    region = data[SCHEMA.REGION].astype(str).map(_REGION_EFFECT).fillna(0.0)
    base = 5000.0
    target = (
        base
        + region.to_numpy()
        + 600.0 * data["is_premium_tier"].to_numpy()
        + 350.0 * data[SCHEMA.ENGINE_SIZE_L].to_numpy()
        + 40.0 * (data[SCHEMA.YEAR].astype(int).to_numpy() - 2010)
        - 0.02 * data[SCHEMA.PRICE_USD].astype(float).to_numpy()
        + 500.0 * data["is_electrified"].to_numpy()
        + rng.normal(0.0, noise_sd, size=len(data))
    )
    return pd.Series(np.clip(target, 100, None), index=df.index, name="synthetic_demand")


@dataclass
class ControlResult:
    """Held-out R² of the same pipeline on the real vs synthetic target."""

    r2_real: float
    r2_synthetic: float
    model: str

    @property
    def pipeline_validated(self) -> bool:
        """The pipeline is sound if it clearly learns the known synthetic signal."""
        return self.r2_synthetic > 0.5

    @property
    def verdict(self) -> str:
        return (
            f"Pipeline VALIDATED: it recovers a known signal (R²={self.r2_synthetic:.3f}) "
            f"but scores ~0 on the real target (R²={self.r2_real:.3f}) — the null "
            f"result is a property of the data, not a modelling failure."
            if self.pipeline_validated
            else "Inconclusive: synthetic signal not recovered."
        )


def run_control(
    df: pd.DataFrame, *, model_name: str = "LightGBM", sample: int = 12000
) -> ControlResult:
    """Train the same model on the real and a synthetic signal-bearing target."""
    seed = get_settings().random_seed
    work = df.sample(min(sample, len(df)), random_state=seed).copy()

    real_ds = make_dataset(work, "regression")
    r2_real = train_one(model_name, real_ds, tune=False).metrics["r2"]

    synth = work.copy()
    synth[SCHEMA.SALES_VOLUME] = make_signal_bearing_target(work).to_numpy()
    synth_ds = make_dataset(synth, "regression")
    r2_synth = train_one(model_name, synth_ds, tune=False).metrics["r2"]

    return ControlResult(r2_real=float(r2_real), r2_synthetic=float(r2_synth), model=model_name)

"""Monte-Carlo uncertainty propagation for the Scenario Simulator.

A point estimate ("demand +15.8%") is not a decision-grade answer — a board
needs the *range*. Elasticities are themselves uncertain, so we place priors on
them and propagate that uncertainty through the constant-elasticity model with a
Monte-Carlo simulation, yielding a **distribution** of projected demand and
**credible intervals** rather than a single number.

This is a transparent Bayesian-flavoured treatment: priors are explicit and
adjustable, sampling is seeded and reproducible, and the output is a full
posterior-predictive-style distribution of the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bmw_sales.config import get_settings
from bmw_sales.simulation.scenario import ELECTRIFIED_FUELS, ScenarioInput


@dataclass(frozen=True)
class ElasticityPriors:
    """Gaussian priors (mean ± sd) on each elasticity — segment-specific.

    Means match the deterministic model's **standard**-segment priors; the
    standard deviations encode honest parameter uncertainty. Use
    :meth:`for_segment` for the luxury/premium tier.
    """

    own_price_mean: float = -0.7
    own_price_sd: float = 0.2
    income_mean: float = 1.3
    income_sd: float = 0.4
    fuel_combustion_mean: float = -0.15
    fuel_electrified_mean: float = 0.10
    fuel_sd: float = 0.08
    regulation_mean: float = 0.08
    regulation_sd: float = 0.03

    @classmethod
    def for_segment(cls, premium: bool) -> "ElasticityPriors":
        """Priors for the premium/luxury tier (Veblen-leaning) or the standard tier."""
        if premium:
            return cls(
                own_price_mean=-0.3,
                own_price_sd=0.15,
                income_mean=2.2,
                income_sd=0.5,
                fuel_combustion_mean=-0.08,
                fuel_electrified_mean=0.06,
            )
        return cls()


@dataclass
class ScenarioDistribution:
    """Monte-Carlo distribution of projected demand for a scenario."""

    base_volume: float
    samples: np.ndarray  # projected volumes, one per draw

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.samples, q))

    @property
    def median(self) -> float:
        return self.percentile(50)

    @property
    def ci80(self) -> tuple[float, float]:
        return self.percentile(10), self.percentile(90)

    @property
    def ci95(self) -> tuple[float, float]:
        return self.percentile(2.5), self.percentile(97.5)

    def pct_change_ci(self) -> tuple[float, float]:
        """80% credible interval expressed as % change vs baseline."""
        lo, hi = self.ci80
        if self.base_volume == 0:
            return 0.0, 0.0
        return (lo / self.base_volume - 1) * 100, (hi / self.base_volume - 1) * 100


def simulate_mc(
    scenario: ScenarioInput,
    priors: ElasticityPriors | None = None,
    *,
    n_draws: int = 5000,
) -> ScenarioDistribution:
    """Propagate elasticity uncertainty through the demand model via Monte Carlo.

    Each draw samples the elasticities from their priors and recomputes projected
    demand; the collection of draws forms the predictive distribution.
    """
    pr = priors or ElasticityPriors()
    rng = np.random.default_rng(get_settings().random_seed)
    electrified = scenario.fuel_type in ELECTRIFIED_FUELS

    dp = scenario.price_change_pct / 100.0
    dy = scenario.gdp_growth_pct / 100.0
    df = scenario.fuel_price_change_pct / 100.0
    dfx = scenario.fx_depreciation_pct / 100.0

    own_price = rng.normal(pr.own_price_mean, pr.own_price_sd, n_draws)
    income = rng.normal(pr.income_mean, pr.income_sd, n_draws)
    fuel_mean = pr.fuel_electrified_mean if electrified else pr.fuel_combustion_mean
    fuel_e = rng.normal(fuel_mean, pr.fuel_sd, n_draws)
    reg_sens = rng.normal(pr.regulation_mean, pr.regulation_sd, n_draws)

    price_mult = np.power(1.0 + dp, own_price)
    income_mult = np.power(1.0 + dy, income)
    fuel_mult = np.power(1.0 + df, fuel_e)
    fx_mult = np.power(1.0 + dfx, own_price)
    reg_dir = 1.0 if electrified else -1.0
    reg_mult = np.clip(
        1.0 + reg_dir * reg_sens * (scenario.regulation_change_pts / 10.0), 0.0, None
    )

    projected = scenario.base_volume * price_mult * income_mult * fuel_mult * fx_mult * reg_mult
    return ScenarioDistribution(base_volume=scenario.base_volume, samples=projected)

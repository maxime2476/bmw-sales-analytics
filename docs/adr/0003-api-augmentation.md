# ADR-0003 — Hybrid external-data augmentation strategy

- **Status:** Accepted
- **Date:** 2026-06-03
- **Author:** Maxime GOURGUECHON
- **Related:** ADR-0001, ADR-0002

## Context

The brief calls for confronting the sales data with external sources. The raw
dataset is also signal-free (ADR-0002), so external macro/energy/regulatory
context is where genuine economic narrative comes from. We need this enrichment
to be **reproducible** (CI must not flake on a third-party API) yet demonstrably
capable of using **real** data.

## Decision

Implement a **hybrid client architecture** (`bmw_sales.apis`):

- A `BaseAPIClient` abstract base providing, for every source:
  - disk **cache** (parquet, keyed by request params),
  - **retry with exponential backoff** (`tenacity`) on transient HTTP errors,
  - a per-instance **circuit breaker**: the first hard failure flips the client
    to mock for the rest of the run instead of repeatedly timing out,
  - **provenance** tagging (`live` / `cache` / `mock`) surfaced to the UI.
- Subclasses implement only `_fetch_live` and a deterministic `_mock`.

Four sources, mapped to the dataset's six regions via official **World Bank
aggregate codes** (EAS, NAC, MEA, LCN, EMU, SSF) and representative currencies:

| Client | Real endpoint | Mock |
|---|---|---|
| `WorldBankClient` | World Bank Indicators API (keyless): inflation `FP.CPI.TOTL.ZG`, GDP/cap `NY.GDP.PCAP.CD` | growth/inflation baselines per region |
| `FXRateClient` | exchangerate.host (key optional) | realistic USD-per-unit anchors + drift |
| `FuelPriceClient` | mock-first (no reliable keyless historical pump-price API) | regional USD/L baselines × fuel-type multipliers |
| `CO2RegulationClient` | mock-first (no canonical free API) | curated stringency schedule (Europe leads) |

`bmw_sales.apis.enrichment` assembles a region×year(×fuel_type) panel and
**left-joins** it onto the sales data, never dropping rows.

## Decisions of note

- **Offline by default for modelling.** The training pipeline runs with
  `BMW_OFFLINE_MODE=true` so model inputs are 100% deterministic. The Streamlit
  app may attempt live calls (with graceful fallback) for freshness.
- **Real path is proven, not theoretical.** `WorldBankClient` was validated
  end-to-end against the live API (returns real Euro-area GDP/capita). Where an
  aggregate lacks an indicator (e.g. EMU inflation), the value is `NaN` and the
  system degrades gracefully — an honest reflection of real-world data gaps.
- **Mock-first ≠ fake.** Mock generators are seeded deterministically and
  anchored to realistic magnitudes, so analyses are stable and defensible.

## Consequences

- **+** CI and Docker need no secrets and never flake.
- **+** Clean extension point: adding a source = one subclass.
- **−** Two sources are mock-only today; documented transparently, with the real
  hook (`_fetch_live`) ready to be implemented when a provider is chosen.

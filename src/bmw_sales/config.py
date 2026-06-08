"""Central, typed configuration for the BMW Sales Analytics platform.

Single source of truth for paths, dataset schema and runtime knobs. Values are
read from environment variables / ``.env`` via :class:`pydantic_settings`, so the
same code runs identically in local dev, CI and Docker without edits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Filesystem layout (resolved relative to the repository root, not the CWD).
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Final[Path] = DATA_DIR / "external"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

RAW_DATASET_PATH: Final[Path] = RAW_DATA_DIR / "BMW_sales_data_(2010-2024).csv"


class DatasetSchema:
    """Canonical column names - referenced everywhere instead of string literals.

    Keeping the schema in one place means a column rename is a one-line change
    and ``mypy`` / IDEs can catch typos that bare strings never would.
    """

    MODEL: Final[str] = "Model"
    YEAR: Final[str] = "Year"
    REGION: Final[str] = "Region"
    COLOR: Final[str] = "Color"
    FUEL_TYPE: Final[str] = "Fuel_Type"
    TRANSMISSION: Final[str] = "Transmission"
    ENGINE_SIZE_L: Final[str] = "Engine_Size_L"
    MILEAGE_KM: Final[str] = "Mileage_KM"
    PRICE_USD: Final[str] = "Price_USD"
    SALES_VOLUME: Final[str] = "Sales_Volume"
    SALES_CLASSIFICATION: Final[str] = "Sales_Classification"

    CATEGORICAL: Final[tuple[str, ...]] = (
        MODEL,
        REGION,
        COLOR,
        FUEL_TYPE,
        TRANSMISSION,
    )
    NUMERIC: Final[tuple[str, ...]] = (
        YEAR,
        ENGINE_SIZE_L,
        MILEAGE_KM,
        PRICE_USD,
        SALES_VOLUME,
    )
    #: Regression target.
    TARGET_REGRESSION: Final[str] = SALES_VOLUME
    #: Classification target.
    TARGET_CLASSIFICATION: Final[str] = SALES_CLASSIFICATION
    #: Empirically discovered leakage threshold (see Data Integrity Report / ADR-0002).
    CLASSIFICATION_THRESHOLD: Final[int] = 7000


class Settings(BaseSettings):
    """Runtime configuration sourced from environment / ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Network / API behaviour ---
    offline_mode: bool = Field(default=False, alias="BMW_OFFLINE_MODE")
    http_timeout: float = Field(default=8.0, alias="BMW_HTTP_TIMEOUT")
    http_max_retries: int = Field(default=3, alias="BMW_HTTP_MAX_RETRIES")
    cache_dir: Path = Field(default=Path(".cache/api"), alias="BMW_CACHE_DIR")

    worldbank_base_url: str = Field(
        default="https://api.worldbank.org/v2", alias="WORLDBANK_BASE_URL"
    )
    fx_base_url: str = Field(default="https://api.exchangerate.host", alias="FX_BASE_URL")
    fx_api_key: str = Field(default="", alias="FX_API_KEY")

    # --- Reproducibility ---
    random_seed: int = Field(default=42, alias="BMW_RANDOM_SEED")


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` instance (cheap; avoids global state in tests)."""
    return Settings()


SCHEMA: Final[DatasetSchema] = DatasetSchema()

"""Base class for the external API clients.

Each client either hits a real endpoint or returns a deterministic mock, so the
project runs offline with no keys. Live calls retry with backoff (tenacity), and
after a failure a per-client circuit breaker falls back to the mock. Successful
responses are cached to parquet, and every result carries its provenance
(live / cache / mock). Subclasses implement ``_fetch_live`` and ``_mock``.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from bmw_sales.config import Settings, get_settings


class DataSource(str, Enum):
    """Provenance of a returned dataset."""

    LIVE = "live"
    CACHE = "cache"
    MOCK = "mock"


@dataclass
class APIResult:
    """A dataset plus its provenance metadata."""

    data: pd.DataFrame
    source: DataSource
    client: str

    @property
    def is_live(self) -> bool:
        return self.source is DataSource.LIVE


class BaseAPIClient(ABC):
    """Abstract hybrid client with caching, retries and a circuit breaker."""

    #: Short, unique client name (used for cache keys and logging).
    name: str = "base"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._cache_dir = Path(self.settings.cache_dir) / self.name
        self._circuit_open = False  # once True, skip live calls for this instance

    # Public API
    def fetch(self, **params: Any) -> APIResult:
        """Return data for ``params``, preferring cache → live → mock.

        The method never raises on network failure: it degrades gracefully to a
        deterministic mock so downstream code always receives a valid frame.
        """
        cache_path = self._cache_path(params)

        cached = self._read_cache(cache_path)
        if cached is not None:
            return APIResult(cached, DataSource.CACHE, self.name)

        if self.settings.offline_mode or self._circuit_open:
            return APIResult(self._mock(**params), DataSource.MOCK, self.name)

        try:
            data = self._fetch_live(**params)
            self._write_cache(cache_path, data)
            return APIResult(data, DataSource.LIVE, self.name)
        except Exception:  # noqa: BLE001 - any failure must fall back, not crash
            self._circuit_open = True
            return APIResult(self._mock(**params), DataSource.MOCK, self.name)

    # To implement in subclasses
    @abstractmethod
    def _fetch_live(self, **params: Any) -> pd.DataFrame:
        """Fetch real data from the upstream API. May raise on failure."""

    @abstractmethod
    def _mock(self, **params: Any) -> pd.DataFrame:
        """Return a deterministic, plausible mock for the same parameters."""

    # Shared HTTP helper (retry + timeout)
    def _http_get_json(self, url: str, params: Optional[dict[str, Any]] = None) -> Any:
        """GET ``url`` and return parsed JSON, with retry/backoff and timeout."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.settings.http_max_retries),
            wait=wait_exponential(multiplier=0.5, max=4),
            retry=retry_if_exception_type(requests.RequestException),
        )
        def _do_request() -> Any:
            resp = requests.get(url, params=params, timeout=self.settings.http_timeout)
            resp.raise_for_status()
            return resp.json()

        return _do_request()

    def _fetch_wb_indicator(
        self, area_code: str, indicator: str, start: int, end: int
    ) -> dict[int, float]:
        """Fetch one World Bank indicator for a country/aggregate as {year: value}.

        Shared by clients backed by the (keyless) World Bank Indicators API.
        Raises if the payload shape is unexpected; returns ``{}`` if the series
        exists but has no observations in range.
        """
        url = f"{self.settings.worldbank_base_url}/country/{area_code}/indicator/{indicator}"
        payload = self._http_get_json(
            url, params={"date": f"{start}:{end}", "format": "json", "per_page": "500"}
        )
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            raise ValueError(f"Unexpected World Bank payload for {indicator}")
        return {
            int(obs["date"]): float(obs["value"])
            for obs in payload[1]
            if obs.get("value") is not None
        }

    # Cache plumbing
    def _cache_key(self, params: dict[str, Any]) -> str:
        payload = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.name}_{digest}"

    def _cache_path(self, params: dict[str, Any]) -> Path:
        return self._cache_dir / f"{self._cache_key(params)}.parquet"

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - corrupt cache should not be fatal
            return None

    def _write_cache(self, path: Path, data: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data.to_parquet(path, index=False)
        except Exception:  # noqa: BLE001 - caching is best-effort
            pass

    @staticmethod
    def _seed_from(*parts: Any) -> int:
        """Stable integer seed derived from arbitrary params (for mock determinism)."""
        payload = "|".join(str(p) for p in parts)
        return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16)

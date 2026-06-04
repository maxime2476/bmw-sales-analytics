"""DuckDB-powered SQL analytics over the raw BMW dataset.

A lightweight analytical layer that runs versioned ``.sql`` files directly
against the CSV via DuckDB — no database server, no ETL. This keeps the
decision-oriented business queries in plain SQL (reviewable, portable) while the
Python layer just orchestrates execution.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from bmw_sales.config import PROJECT_ROOT, RAW_DATASET_PATH

QUERIES_DIR: Path = PROJECT_ROOT / "sql" / "queries"

#: The view name the ``.sql`` files reference.
VIEW_NAME = "bmw"


def _connect(dataset_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with the dataset exposed as a view."""
    path = dataset_path or RAW_DATASET_PATH
    # DuckDB cannot bind parameters inside CREATE VIEW, so the (trusted, internal)
    # path is inlined with single-quote escaping to stay injection-safe.
    safe_path = str(path).replace("'", "''")
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW {VIEW_NAME} AS " f"SELECT * FROM read_csv_auto('{safe_path}', header=true)"
    )
    return con


def list_queries() -> list[str]:
    """Return the available query names (``.sql`` file stems), sorted."""
    return sorted(p.stem for p in QUERIES_DIR.glob("*.sql"))


def run_query(name: str, *, dataset_path: Path | None = None) -> pd.DataFrame:
    """Execute the named query and return the result as a DataFrame.

    Parameters
    ----------
    name:
        A query name from :func:`list_queries` (the ``.sql`` file stem).
    dataset_path:
        Optional dataset override (for tests/fixtures).
    """
    sql_path = QUERIES_DIR / f"{name}.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"Unknown query '{name}'. Available: {list_queries()}")
    sql = sql_path.read_text(encoding="utf-8")
    con = _connect(dataset_path)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def run_all(*, dataset_path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Execute every query and return ``{name: DataFrame}``."""
    return {name: run_query(name, dataset_path=dataset_path) for name in list_queries()}

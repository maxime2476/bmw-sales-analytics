"""Generate the SQL business-insights report from the DuckDB queries.

Run as a script::

    python -m bmw_sales.sql.report
"""

from __future__ import annotations

from datetime import date

from bmw_sales.config import REPORTS_DIR
from bmw_sales.sql.analytics import run_all

_TITLES = {
    "top_regions_by_volume": "Sales volume by region",
    "price_stats_by_model": "Price distribution by model (USD)",
    "electrification_by_region": "Electrification share by region",
    "yoy_volume": "Year-over-year total volume",
    "high_rate_by_region": "'High' classification rate by region",
}


def build_report() -> str:
    """Run every query and render a markdown report of the results."""
    results = run_all()
    blocks = []
    for name, df in results.items():
        title = _TITLES.get(name, name.replace("_", " ").title())
        blocks.append(f"## {title}\n\n`{name}.sql`\n\n{df.to_markdown(index=False)}\n")
    body = "\n".join(blocks)
    return (
        f"# SQL Business Insights — BMW Sales\n\n"
        f"*Generated: {date.today().isoformat()} · Author: Maxime GOURGUECHON*\n\n"
        f"> Decision-oriented analytics run with **DuckDB** directly over the raw "
        f"CSV (no ETL). Queries live in `sql/queries/`. Reproduce with `make sql`.\n\n"
        f"{body}\n"
        f"## Reading the results\n\n"
        f"The flat distributions across regions (~16.7% each) and models (~$75k "
        f"each) are the SQL view of the same finding the audit proves statistically: "
        f"the data is uniform noise (see ADR-0002 and `signal_audit.md`).\n"
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "sql_insights.md"
    out.write_text(build_report(), encoding="utf-8")
    print(f"[OK] SQL insights written to {out}")


if __name__ == "__main__":
    main()

"""Guard tests — keep README claims in sync with the single source of truth.

The project's thesis is precision, so its own numbers must not drift. These
tests fail CI if the README's stated coverage gate or ADR count diverges from
the actual configuration / files.
"""

from __future__ import annotations

import re

from bmw_sales.config import PROJECT_ROOT

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


def test_coverage_gate_matches_ci() -> None:
    """README's stated coverage gate must equal the CI `--cov-fail-under` value."""
    ci = (PROJECT_ROOT / ".github" / "workflows" / "main.yml").read_text(encoding="utf-8")
    m = re.search(r"--cov-fail-under=(\d+)", ci)
    assert m, "no --cov-fail-under in the CI workflow"
    gate = m.group(1)
    assert f"{gate}%" in README, f"README must mention the {gate}% coverage gate"


def test_adr_count_matches_files() -> None:
    """README's stated ADR count must equal the number of ADR files."""
    n_adr = len(list((PROJECT_ROOT / "docs" / "adr").glob("[0-9]*.md")))
    assert f"{n_adr} ADR" in README, f"README should state {n_adr} ADRs"

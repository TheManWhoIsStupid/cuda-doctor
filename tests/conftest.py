"""Shared test fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def load_fixture(fixtures_dir: Path, relative: str) -> str:
    return (fixtures_dir / relative).read_text(encoding="utf-8")

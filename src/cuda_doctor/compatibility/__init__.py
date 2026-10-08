"""Compatibility knowledge, loaded from versioned JSON data files.

Keeping this knowledge in data (not code) lets it be updated without touching
diagnostic logic. The JSON files ship inside the package so installed tools
can read them via importlib.resources.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from cuda_doctor.compatibility.compiler_cuda import CompilerCompatibility, CompilerVerdict
from cuda_doctor.compatibility.cuda_driver import DriverCompatibility, DriverVerdict
from cuda_doctor.compatibility.pytorch_cuda import (
    KnownIssue,
    KnownIssues,
    TorchCudaInterpretation,
    interpret_torch_cuda,
)
from cuda_doctor.core.exceptions import CompatibilityDataError

__all__ = [
    "CompatibilityData",
    "CompatibilityDataError",
    "CompilerCompatibility",
    "CompilerVerdict",
    "DriverCompatibility",
    "DriverVerdict",
    "KnownIssue",
    "KnownIssues",
    "TorchCudaInterpretation",
    "interpret_torch_cuda",
    "load_compatibility_data",
]


@dataclass(frozen=True)
class CompatibilityData:
    """All compatibility tables used by the diagnosis engine."""

    driver: DriverCompatibility
    compiler: CompilerCompatibility
    known_issues: KnownIssues


def load_compatibility_data() -> CompatibilityData:
    """Load the bundled compatibility tables (raises CompatibilityDataError)."""
    try:
        driver = DriverCompatibility.from_json(
            _read_json("cuda_driver_compatibility.json")
        )
        compiler = CompilerCompatibility.from_json(
            _read_json("cuda_compiler_compatibility.json")
        )
        known = KnownIssues.from_json(_read_json("known_issues.json"))
    except (OSError, ValueError, TypeError) as exc:
        raise CompatibilityDataError(f"Cannot load compatibility data: {exc}") from exc
    return CompatibilityData(driver=driver, compiler=compiler, known_issues=known)


def _read_json(name: str) -> Any:
    resource = files("cuda_doctor").joinpath("data").joinpath(name)
    if not resource.is_file():
        raise CompatibilityDataError(f"Bundled data file missing: {name}")
    return json.loads(resource.read_text(encoding="utf-8"))

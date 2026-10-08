"""Diagnostic context passed to every check."""

from __future__ import annotations

from dataclasses import dataclass

from cuda_doctor.compatibility import CompatibilityData
from cuda_doctor.core.enums import Platform
from cuda_doctor.core.models import EnvironmentSnapshot


@dataclass(frozen=True)
class DiagnosticContext:
    """A snapshot plus the compatibility knowledge needed to judge it.

    ``compatibility`` may be ``None`` (e.g. bundled data unreadable); checks
    degrade gracefully instead of failing.
    """

    snapshot: EnvironmentSnapshot
    compatibility: CompatibilityData | None = None

    @property
    def platform(self) -> Platform:
        return self.snapshot.system.platform

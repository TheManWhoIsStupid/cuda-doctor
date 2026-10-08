"""Base interface for diagnostic checks.

A Check evaluates the normalized snapshot (plus compatibility knowledge) and
returns structured issues. Checks are read-only and side-effect free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Severity
from cuda_doctor.diagnosis.issue import Issue


class Check(ABC):
    """One diagnostic rule with a stable code."""

    code: ClassVar[str]
    category: ClassVar[str]

    @abstractmethod
    def run(self, ctx: DiagnosticContext) -> list[Issue]:
        """Evaluate the snapshot; return [] when the rule does not apply."""

    def issue(
        self,
        severity: Severity,
        title: str,
        description: str = "",
        evidence: list[str] | None = None,
        recommendations: list[str] | None = None,
    ) -> Issue:
        """Build an Issue bound to this check's code and category."""
        return Issue(
            code=self.code,
            category=self.category,
            severity=severity,
            title=title,
            description=description,
            evidence=evidence or [],
            recommendations=recommendations or [],
        )

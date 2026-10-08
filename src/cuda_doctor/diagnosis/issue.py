"""Structured diagnostic issue model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cuda_doctor.core.enums import EnvironmentStatus, Severity
from cuda_doctor.utils.redact import redact_value

__all__ = ["Issue", "Summary"]


@dataclass
class Issue:
    """One diagnostic finding with a stable issue code."""

    code: str
    severity: Severity
    title: str
    description: str = ""
    evidence: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    category: str = "general"

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        data = asdict(self)
        return redact_value(data) if redact else redact_value(data, home="")


@dataclass
class Summary:
    """Issue counts and the overall environment verdict."""

    critical: int = 0
    errors: int = 0
    warnings: int = 0
    info: int = 0
    status: EnvironmentStatus = EnvironmentStatus.HEALTHY

    @classmethod
    def from_issues(cls, issues: list[Issue]) -> Summary:
        summary = cls(
            critical=sum(1 for i in issues if i.severity is Severity.CRITICAL),
            errors=sum(1 for i in issues if i.severity is Severity.ERROR),
            warnings=sum(1 for i in issues if i.severity is Severity.WARNING),
            info=sum(1 for i in issues if i.severity is Severity.INFO),
        )
        summary.status = cls._status_of(summary)
        return summary

    @staticmethod
    def _status_of(summary: Summary) -> EnvironmentStatus:
        if summary.critical:
            return EnvironmentStatus.BROKEN
        if summary.errors:
            return EnvironmentStatus.DEGRADED
        if summary.warnings:
            return EnvironmentStatus.USABLE_WITH_WARNINGS
        return EnvironmentStatus.HEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "critical": self.critical,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "status": self.status.value,
        }

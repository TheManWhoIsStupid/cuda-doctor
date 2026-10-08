"""The diagnosis engine: snapshot in, issues out."""

from __future__ import annotations

from dataclasses import dataclass, field

from cuda_doctor.checks import Check, default_checks
from cuda_doctor.compatibility import CompatibilityData, load_compatibility_data
from cuda_doctor.core.context import DiagnosticContext
from cuda_doctor.core.enums import Severity
from cuda_doctor.core.exceptions import CompatibilityDataError
from cuda_doctor.core.models import EnvironmentSnapshot
from cuda_doctor.diagnosis.issue import Issue, Summary

_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.ERROR: 1,
    Severity.WARNING: 2,
    Severity.INFO: 3,
}


@dataclass
class DiagnosisResult:
    """Everything the engine concluded about one snapshot."""

    issues: list[Issue] = field(default_factory=list)
    summary: Summary = field(default_factory=Summary)
    check_errors: dict[str, str] = field(default_factory=dict)


class DiagnosisEngine:
    """Runs every registered check; one crashing check never stops the rest."""

    def __init__(
        self,
        checks: list[Check] | None = None,
        compatibility: CompatibilityData | None = None,
    ) -> None:
        self.checks = checks if checks is not None else default_checks()
        self._compatibility = compatibility

    def run(self, snapshot: EnvironmentSnapshot) -> DiagnosisResult:
        result = DiagnosisResult()
        compatibility = self._compatibility or self._load_compat(result)
        ctx = DiagnosticContext(snapshot=snapshot, compatibility=compatibility)

        issues: list[Issue] = []
        for check in self.checks:
            try:
                issues.extend(check.run(ctx))
            except Exception as exc:
                result.check_errors[check.code] = f"{type(exc).__name__}: {exc}"
        issues.sort(key=lambda issue: (_SEVERITY_ORDER[issue.severity], issue.code))
        result.issues = issues
        result.summary = Summary.from_issues(issues)
        return result

    @staticmethod
    def _load_compat(result: DiagnosisResult) -> CompatibilityData | None:
        try:
            return load_compatibility_data()
        except CompatibilityDataError as exc:
            # Unreadable bundled data must not disable diagnosis entirely.
            result.check_errors["compatibility-data"] = str(exc)
            return None

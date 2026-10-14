"""Reporter interface shared by all output formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from cuda_doctor.core.models import EnvironmentSnapshot
from cuda_doctor.diagnosis.engine import DiagnosisResult
from cuda_doctor.diagnosis.models import Diagnosis


@dataclass(frozen=True)
class ReportInputs:
    """Everything a reporter needs: snapshot + verdict + metadata.

    ``generated_at`` is injected (ISO-8601 UTC) so output is testable.
    """

    snapshot: EnvironmentSnapshot
    result: DiagnosisResult
    generated_at: str


def finalized_diagnoses(diagnoses: Sequence[Diagnosis]) -> list[Diagnosis]:
    """The diagnoses eligible for report output (frozen model contract).

    ``role=None`` is internal pre-resolver state — not a fifth role — and
    never reaches report output. A malformed role-less entry reaching a
    reporter is skipped in every format (the accepted terminal/Markdown
    behavior and the JSON schema-v2 boundary alike): reporters never
    invent a role for it, and the stored ``Diagnosis`` is never modified.
    """
    return [diagnosis for diagnosis in diagnoses if diagnosis.role is not None]


def suppressed_issue_codes(diagnoses: Sequence[Diagnosis]) -> set[str]:
    """The effective human-presentation suppression set (architecture §25.2).

    Suppression activates only from the explicit
    ``presentation_suppressed_issue_codes`` carrier of diagnoses that are
    actually rendered by the human-facing reporters — role-finalized
    entries only, since a role-less pre-resolver candidate never reaches
    report output and must not activate its carrier. The set is computed
    from finalized diagnoses exactly as stored: this helper (and its
    callers) never mutate ``DiagnosisResult.issues`` or the diagnoses
    themselves; suppression is presentation-only by construction.
    """
    return {
        code
        for diagnosis in finalized_diagnoses(diagnoses)
        for code in diagnosis.presentation_suppressed_issue_codes
    }


class Reporter(ABC):
    """One output format; renderers are pure functions of their inputs."""

    @abstractmethod
    def render(self, inputs: ReportInputs) -> str:
        """Return the full report as text."""


def format_memory(memory_mb: int | None) -> str:
    """Render a MiB quantity as human-readable GiB ("" when unknown)."""
    if not memory_mb:
        return ""
    return f"{memory_mb / 1024:.1f} GiB"

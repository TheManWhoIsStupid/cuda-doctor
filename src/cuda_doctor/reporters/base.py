"""Reporter interface shared by all output formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from cuda_doctor.core.models import EnvironmentSnapshot
from cuda_doctor.diagnosis.engine import DiagnosisResult


@dataclass(frozen=True)
class ReportInputs:
    """Everything a reporter needs: snapshot + verdict + metadata.

    ``generated_at`` is injected (ISO-8601 UTC) so output is testable.
    """

    snapshot: EnvironmentSnapshot
    result: DiagnosisResult
    generated_at: str


class Reporter(ABC):
    """One output format; renderers are pure functions of their inputs."""

    @abstractmethod
    def render(self, inputs: ReportInputs) -> str:
        """Return the full report as text."""

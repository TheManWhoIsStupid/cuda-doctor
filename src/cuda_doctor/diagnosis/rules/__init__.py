"""Diagnosis rules: pure evaluators over normalized ``DiagnosisFacts``.

Each rule consumes already-normalized facts and emits at most one
``DiagnosisCandidate`` whose ``Diagnosis`` keeps ``role=None`` — final
roles, relationships, and engine wiring belong to later phases.
"""

from cuda_doctor.diagnosis.rules.base import DiagnosisRule
from cuda_doctor.diagnosis.rules.d001 import D001ToolkitResolutionConflict

__all__ = ["D001ToolkitResolutionConflict", "DiagnosisRule"]

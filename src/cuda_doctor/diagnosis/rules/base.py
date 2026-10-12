"""Minimal shared diagnosis-rule contract (frozen architecture §15).

A rule is a pure evaluator over already-normalized ``DiagnosisFacts``:
it never collects, probes host state, mutates its inputs, or assigns a
final report role. This module carries only the shared Protocol — no
rule engine, no base-class policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from cuda_doctor.diagnosis.facts import DiagnosisFacts
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.models import DiagnosisCandidate

__all__ = ["DiagnosisRule"]

# Shared immutable default: rules that need no upstream candidate
# context may ignore ``prior_candidates`` entirely.
NO_PRIOR_CANDIDATES: Mapping[str, DiagnosisCandidate] = MappingProxyType({})


@runtime_checkable
class DiagnosisRule(Protocol):
    """One diagnosis rule (architecture §15).

    ``evaluate`` inspects pre-collected normalized evidence only and
    returns at most one candidate, or ``None`` when the rule does not
    apply. ``issues`` are the already-run low-level issues — rules never
    re-derive check logic. ``prior_candidates`` maps the codes of rules
    evaluated earlier in the frozen order (D001 → D002 → D004 → D003)
    to their candidates; a missing code means "upstream state
    unavailable", never negative evidence.
    """

    code: str

    def evaluate(
        self,
        facts: DiagnosisFacts,
        issues: Sequence[Issue],
        prior_candidates: Mapping[str, DiagnosisCandidate] = NO_PRIOR_CANDIDATES,
    ) -> DiagnosisCandidate | None:
        ...

"""Structured diagnosis output models (v0.2).

Passive data containers only: they describe what a concluded diagnosis
looks like. Deciding when a diagnosis exists, which diagnosis is
PRIMARY, how confident a rule may be, or how issues are presented is
later-phase resolver/reporter work — never here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from cuda_doctor.core.enums import Severity
from cuda_doctor.utils.redact import redact_value

__all__ = [
    "Confidence",
    "Diagnosis",
    "DiagnosisCandidate",
    "DiagnosisRelationship",
    "DiagnosisRole",
    "EvidenceItem",
    "ImpactItem",
    "RemediationSafety",
    "RemediationStep",
]


class Confidence(str, Enum):
    """How strongly the observed facts support a diagnosis.

    Deliberately ordinal-free: the resolver compares candidates by its
    own frozen ordering, never by enum identity or numeric value.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DiagnosisRole(str, Enum):
    """Presentation role of one diagnosis among its peers."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    CONTRIBUTING = "contributing"
    POSSIBLE = "possible"


class RemediationSafety(str, Enum):
    """Safety classification of executing one remediation step."""

    SAFE = "safe"
    CAUTION = "caution"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class EvidenceItem:
    """One bounded, presentation-ready fact backing a diagnosis.

    Exactly key/value/source strings — never an ``EnvironmentSnapshot``,
    raw collector object, or arbitrary nested dict. Bounded evidence is
    derived by the diagnosis rules (later phases) from already-redacted
    facts.
    """

    key: str
    value: str
    source: str


@dataclass(frozen=True)
class ImpactItem:
    """One affected scope with its observed status.

    ``scope`` names an affected area (for example PyTorch runtime,
    extension build, local CUDA development); ``status`` describes the
    observed/expected effect in that scope. Both are plain strings —
    no inference happens here.
    """

    scope: str
    status: str


@dataclass(frozen=True)
class RemediationStep:
    """One suggested remediation instruction.

    Read-only by construction: ``command`` is a display string the user
    may choose to run themselves — cuda-doctor never executes it, and
    this model carries no callables.
    """

    action: str
    why: str
    safety: RemediationSafety
    command: str | None = None


@dataclass(frozen=True)
class DiagnosisRelationship:
    """A data-only link from one diagnosis to another, by code.

    ``kind`` is a relationship name such as ``"explains"`` or
    ``"related_to"``. The resolver (later phase) creates relationships;
    this model performs no graph traversal, cycle detection, or
    parent selection.
    """

    kind: str
    target_code: str


@dataclass
class Diagnosis:
    """One concluded root-cause diagnosis (passive output model).

    Carries the frozen diagnosis structure only. It has no policy
    properties or methods: no effective severity/confidence, no
    is-primary predicate, no ranking, no suppression behavior.
    ``role`` is assigned by the resolver (later phase), never here.
    """

    code: str
    title: str
    severity: Severity
    confidence: Confidence
    role: DiagnosisRole
    summary: str
    cause_class: str | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    impact: list[ImpactItem] = field(default_factory=list)
    remediation: list[RemediationStep] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    related_issues: list[str] = field(default_factory=list)
    relationships: list[DiagnosisRelationship] = field(default_factory=list)
    # Rule-11 carrier: healthy/excluded causes, kept separate from
    # evidence by frozen architecture.
    exclusions: list[str] = field(default_factory=list)
    # Presentation-only suppression carrier (TORCH002→D003, TORCH004→D002
    # in later phases). Removes and mutates nothing by itself.
    presentation_suppressed_issue_codes: list[str] = field(default_factory=list)

    def to_dict(self, *, redact: bool = True) -> dict[str, Any]:
        data = asdict(self)
        return redact_value(data) if redact else redact_value(data, home="")


@dataclass
class DiagnosisCandidate:
    """Internal resolver input: a diagnosis plus resolution-neutral hints.

    Never serialized into reports. The resolver (later phase) consumes
    candidates to assign final roles and relationships; the hints carry
    no policy themselves.
    """

    diagnosis: Diagnosis
    explains_active_failure: bool = False
    risk_only: bool = False
    causal_parent_codes: tuple[str, ...] = ()

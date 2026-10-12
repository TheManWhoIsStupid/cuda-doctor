"""D004 — CUDA runtime library conflict.

Diagnoses a statically modeled runtime-library resolution conflict: within
the explicitly modeled ``LD_LIBRARY_PATH`` search order, a reliably
identified candidate of a supported family (``libcudart`` / ``libcublas`` /
``libcudnn``) from a conflicting family is preferred over the environment's
expected family, which is itself present later in the same modeled ordering
(frozen semantics §23, truth tables §19). The conclusion is deliberately
static — it never claims which object a process actually loaded, never
invents loader precedence between origins (package-local, toolkit, conda,
ldconfig stay inventory-only), and never exceeds MEDIUM confidence (§19.6:
HIGH is reserved for future direct loaded-object evidence). Multiplicity
without modeled precedence stays informational (§19.7); UNKNOWN never
becomes mismatch (§19.5). The rule is pure over ``DiagnosisFacts`` — no
collection, no host state, no ELF inspection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cuda_doctor.core.enums import Severity
from cuda_doctor.diagnosis.facts import (
    DiagnosisFacts,
    IdentityReliability,
    RuntimeLibraryCandidateFact,
    RuntimeLibraryFacts,
)
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.models import (
    Confidence,
    Diagnosis,
    DiagnosisCandidate,
    EvidenceItem,
    ImpactItem,
    RemediationSafety,
    RemediationStep,
)
from cuda_doctor.diagnosis.rules.base import NO_PRIOR_CANDIDATES

__all__ = ["D004RuntimeLibraryConflict"]

_CODE = "D004"
_TITLE = "CUDA runtime library conflict"

# Frozen family scope (truth tables §19.2) in the deterministic aggregation
# order (plan §14: multiple families aggregate into one candidate; this is
# also the alphabetical order of the three frozen family names).
_SUPPORTED_FAMILIES = ("libcudart", "libcublas", "libcudnn")

# The only origin whose entry order the collector establishes (facts carry
# ``modeled_search_order`` exclusively for it — architecture §10.4). The
# literal matches the frozen origin vocabulary without importing the
# collector module.
_MODELED_ORIGIN = "ld_library_path"

# Package-local candidates establish environment-observed baselines (the
# current Python environment's own expected library families).
_BASELINE_ORIGIN = "python_package"

# Version identities reliable enough to support a modeled conflict claim
# (semantics §23.7 "with reliable version identities"). Weakly derived or
# unknown identities cannot prove which family a candidate belongs to.
_RELIABLE_RELIABILITIES = frozenset(
    (IdentityReliability.DIRECT, IdentityReliability.DERIVED_STRONG)
)


class D004RuntimeLibraryConflict:
    """Emit D004 when the modeled search order prefers a conflicting family.

    A family conflict requires three things the static model can actually
    establish: an expected family baseline, a reliably identified
    ``ld_library_path`` candidate of a conflicting family at the earliest
    modeled position, and the expected family itself appearing later in the
    same modeled ordering (truth tables §19.8). Anything less — multiplicity
    without precedence, package-local-vs-LDP ambiguity needing
    RPATH/RUNPATH, ldconfig-only differences, unknown identities or
    baselines — is UNKNOWN and emits nothing.
    """

    code = _CODE

    def evaluate(
        self,
        facts: DiagnosisFacts,
        issues: Sequence[Issue],
        prior_candidates: Mapping[str, DiagnosisCandidate] = NO_PRIOR_CANDIDATES,
    ) -> DiagnosisCandidate | None:
        # D004 is independent of D001/D002 (truth tables §20.6) and is not
        # yet consumed by D003 (Phase 9): no issue prose and no upstream
        # context is consumed — both parameters stay intentionally unused.
        if not facts.applicability.d004:
            # Not evaluated on this platform (Linux-only, §19.1): no
            # candidate, no error — never guessed, never "conflict-free".
            return None
        libraries = facts.runtime_libraries
        findings = tuple(
            finding
            for finding in (
                _family_finding(family, libraries) for family in _SUPPORTED_FAMILIES
            )
            if finding is not None
        )
        if not findings:
            # Includes every §19.7 no-conflict case and every UNKNOWN
            # resolution: LOW-confidence ambiguity stays informational.
            return None
        diagnosis = Diagnosis(
            code=_CODE,
            title=_TITLE,
            severity=Severity.WARNING,
            confidence=Confidence.MEDIUM,
            summary=_summary(findings),
            # The resolver (Phase 10) owns final roles; whether D004
            # explains a D003 symptom is decided by Phase 9 wiring, not here.
            role=None,
            evidence=_evidence(findings),
            impact=_impact(),
            remediation=_remediation(),
            verification=_verification(),
            related_issues=[],
            presentation_suppressed_issue_codes=[],
        )
        # Static analysis alone: a modeled resolution risk, not an observed
        # runtime failure (semantics §23.3, §23.7).
        return DiagnosisCandidate(diagnosis=diagnosis, risk_only=True)


@dataclass(frozen=True)
class _FamilyFinding:
    """One family's modeled conflict: preferred candidate vs baseline.

    Carries only bounded, presentation-safe identity (family, version,
    modeled position, reliability) — never a path, entry text, or any raw
    inventory content.
    """

    family: str
    preferred: RuntimeLibraryCandidateFact
    baseline: str


def _reliably_known(candidate: RuntimeLibraryCandidateFact) -> bool:
    """Whether the candidate's family version can prove anything (§13).

    An absent version or a weakly/unknown-derived identity is not a
    comparable identity: known-vs-unknown never becomes mismatch.
    """
    return (
        candidate.family_version is not None
        and candidate.reliability in _RELIABLE_RELIABILITIES
    )


def _baseline(family: str, libraries: RuntimeLibraryFacts) -> str | None:
    """The family's expected version baseline, or None for UNKNOWN.

    libcudart compares against the PyTorch CUDA runtime family directly
    (§11A/§19.8). libcublas may use it only through a reliable mapping —
    and the only reliable mapping evidence is the environment itself: the
    current Python environment's own package-local libcublas family,
    observed alongside the known torch family, establishes what family
    this PyTorch runtime ships (§19.9/§23.8). No numeric CUDA↔SONAME
    table is assumed; without the observed pair the relation is UNKNOWN.
    libcudnn never touches the torch family — its baseline is the
    environment-local package cuDNN family already normalized in the
    facts (§11C/§19.10).
    """
    if family == "libcudart":
        return libraries.torch_runtime_family
    if family == "libcudnn":
        return libraries.package_local_cudnn_family
    # libcublas: the environment-observed torch-family → package-local pair.
    if not libraries.torch_runtime_family:
        return None
    observed = {
        candidate.family_version
        for candidate in libraries.candidates
        if candidate.family == family
        and candidate.origin == _BASELINE_ORIGIN
        and _reliably_known(candidate)
    }
    if len(observed) != 1:
        # Zero or several disagreeing package-local families: the mapping
        # is not reliable, so the libcublas relation is UNKNOWN (§19.9).
        return None
    return next(iter(observed))


def _family_finding(family: str, libraries: RuntimeLibraryFacts) -> _FamilyFinding | None:
    """Establish one family's modeled conflict, or None (§19.7-§19.11).

    Conflict means: the earliest modeled ``ld_library_path`` candidate is a
    reliably identified member of a family that conflicts with the
    baseline, while the expected family itself appears later in the same
    modeled ordering (§19.8 — if the expected copy exists only
    package-local, deciding the winner needs RPATH/RUNPATH and the
    resolution is UNKNOWN, §19.11). Candidates from every other origin,
    and anything the collector could not order, carry no precedence and
    are never dressed up as winners (§19.4, §19.12).
    """
    baseline = _baseline(family, libraries)
    if baseline is None:
        return None
    # Exact duplicate facts collapse before any ordering decision, so a
    # repeated candidate neither duplicates evidence nor blocks emission.
    modeled = sorted(
        {
            candidate
            for candidate in libraries.candidates
            if candidate.family == family
            and candidate.origin == _MODELED_ORIGIN
            and candidate.modeled_search_order is not None
        },
        key=lambda candidate: (
            candidate.modeled_search_order or 0,
            candidate.family_version or "",
        ),
    )
    if not modeled:
        return None
    earliest_order = modeled[0].modeled_search_order or 0
    earliest = [c for c in modeled if (c.modeled_search_order or 0) == earliest_order]
    if len(earliest) != 1:
        # Two distinct candidates share the earliest modeled position:
        # choosing between them needs per-library loader semantics the
        # model does not cover (§19.5) — resolution UNKNOWN.
        return None
    preferred = earliest[0]
    if not _reliably_known(preferred):
        # The preferred candidate's family is not reliably identifiable;
        # skipping it would fabricate precedence for the next entry.
        return None
    if preferred.family_version == baseline:
        # The expected family is what the modeled order prefers (§19.7
        # N01-N03: matching copies, however many, are no conflict).
        return None
    expected_later = any(
        (candidate.modeled_search_order or 0) > earliest_order
        and candidate.family_version == baseline
        and _reliably_known(candidate)
        for candidate in modeled
    )
    if not expected_later:
        # The expected copy is not a candidate within the modeled ordering
        # (§19.8): if it exists only package-local / toolkit / conda /
        # ldconfig, the winner depends on RPATH/RUNPATH and friends (§19.11,
        # §19.12) — UNKNOWN, no MEDIUM D004.
        return None
    return _FamilyFinding(family=family, preferred=preferred, baseline=baseline)


def _summary(findings: Sequence[_FamilyFinding]) -> str:
    """Modeled-order wording only (semantics §23.7).

    Every clause stays within "preferred among the modeled candidates" /
    "strongly indicates a conflict within the modeled search order" — no
    claim about which object any process actually loaded.
    """
    described = ", ".join(
        f"{finding.family} family {finding.preferred.family_version} "
        f"(expected family {finding.baseline})"
        for finding in findings
    )
    return (
        "Within the modeled LD_LIBRARY_PATH search order, a conflicting "
        f"library candidate is preferred over the environment's expected "
        f"family: {described}. The static inventory strongly indicates a "
        "runtime-library resolution conflict; which object the process "
        "actually loads is not observed by this analysis."
    )


def _evidence(findings: Sequence[_FamilyFinding]) -> list[EvidenceItem]:
    """Bounded, privacy-safe evidence (plan §18).

    Family, version, origin class, reliability, modeled position, the
    expected family, and the normalized static relation — never a library
    path, LD_LIBRARY_PATH contents, conda prefix, site-packages location,
    or scan-error text.
    """
    items: list[EvidenceItem] = []
    for finding in findings:
        preferred = finding.preferred
        items.append(
            EvidenceItem(
                key=f"{finding.family}_conflicting",
                value=(
                    f"family {preferred.family_version} at "
                    f"LD_LIBRARY_PATH entry {preferred.modeled_search_order} "
                    f"({preferred.reliability.value})"
                ),
                source="runtime_libraries",
            )
        )
        items.append(
            EvidenceItem(
                key=f"{finding.family}_expected",
                value=f"family {finding.baseline}",
                source=(
                    "torch"
                    if finding.family == "libcudart"
                    else "runtime_libraries"
                ),
            )
        )
    items.append(
        EvidenceItem(
            key="static_relation",
            value="conflicting candidate preferred within the modeled search order",
            source="compatibility",
        )
    )
    return items


def _impact() -> list[ImpactItem]:
    """Expected impact, preserving the §23.3 existence/precedence/loaded
    distinction inside the diagnosis output itself."""
    return [
        ImpactItem(
            scope="CUDA runtime library resolution",
            status="conflicting family preferred within the modeled search order",
        ),
        ImpactItem(
            scope="PyTorch CUDA initialization",
            status="at risk if the modeled resolution applies",
        ),
        ImpactItem(
            scope="actually loaded library",
            status="not observed (static analysis only)",
        ),
    ]


def _remediation() -> list[RemediationStep]:
    """Read-only SAFE guidance (truth tables §19.15, semantics plan §19).

    D004 cannot know which runtime-library source the user intends or
    which entry is safe to change, so every step carries ``command=None``
    and no path, version, entry text, or package-manager action is
    invented. Deleting or overwriting shared libraries is never
    recommended; only the search configuration may be simplified.
    """
    return [
        RemediationStep(
            action=(
                "Temporarily remove or reorder the conflicting "
                "LD_LIBRARY_PATH entry so the environment's expected "
                "library family is preferred within the search order"
            ),
            why=(
                "An earlier entry wins within the modeled search order; "
                "moving or dropping the conflicting entry lets the "
                "expected family resolve first without touching any files"
            ),
            safety=RemediationSafety.SAFE,
        ),
        RemediationStep(
            action=(
                "Start from a clean shell (or one without the inherited "
                "LD_LIBRARY_PATH override) before running CUDA workloads"
            ),
            why=(
                "A fresh shell avoids inheriting stale library-path "
                "overrides from the surrounding environment"
            ),
            safety=RemediationSafety.SAFE,
        ),
        RemediationStep(
            action="Re-run cuda-doctor after adjusting the environment",
            why=(
                "A re-run shows whether the modeled runtime-library "
                "conflict is gone and PyTorch CUDA initialization behaves "
                "as expected"
            ),
            safety=RemediationSafety.SAFE,
        ),
    ]


def _verification() -> list[str]:
    """Frozen verification minimum (truth tables §19.16), display-only.

    D004 is Linux-only, so no platform keying is needed; direct
    loaded-object inspection is intentionally out of scope in v0.2.0.
    """
    return [
        "echo $LD_LIBRARY_PATH",
        "cuda-doctor",
        "python -c \"import torch; print(torch.cuda.is_available())\"",
    ]

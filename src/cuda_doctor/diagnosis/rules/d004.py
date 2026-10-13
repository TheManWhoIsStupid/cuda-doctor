"""D004 — CUDA runtime library conflict.

Diagnoses a statically modeled runtime-library alias conflict over
OBSERVED EXACT versioned load names (frozen Revision-3 semantics, truth
tables §19.8/§19.8a). For a supported family (libcudart, libcublas,
libcudnn) with a known family baseline, an LD_LIBRARY_PATH candidate
whose exact observed versioned name claims the baseline family resolves
to a different canonical target family:

- Class A — same-name shadowing: a provider of the SAME exact name with
  a baseline-compatible target appears strictly later in the same
  modeled ordering, so within the modeled search the incompatible
  target precedes the compatible one (rows C/D).
- Class B — independently corroborated misleading alias: a current-Python
  package-local provider of the SAME exact name is baseline-compatible
  (row F). A conditional static risk about environment content — never
  a cross-origin loader winner claim.

Everything else stays conservative (UNKNOWN, no candidate): bare names
(row E), different names (rows A/B, row S), irrelevant name families
(rows Q/R), an unknown earliest target which is never skipped over (row
H), an ambiguous earliest position (§19.5), an uncorroborated alias (row
G), and any conclusion that would need ELF inspection or precedence
between origins. ``candidate.reliability`` never enters a conflict
predicate; its only use is package-local FAMILY BASELINE derivation
(amendment §8). The conclusion is static: MEDIUM confidence ceiling,
WARNING severity, and the actually loaded object is never claimed. The
rule is pure over ``DiagnosisFacts`` — no collection, no host state, no
ELF inspection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

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

# Frozen family scope (truth tables §19.2) in the deterministic
# aggregation order (amendment §20: family order libcudart, libcublas,
# libcudnn — also the alphabetical order of the three frozen names).
_SUPPORTED_FAMILIES = ("libcudart", "libcublas", "libcudnn")

# The only origin whose entry order the collector establishes (facts
# carry ``modeled_search_order`` exclusively for it — architecture
# §10.4). The literal matches the frozen origin vocabulary without
# importing the collector module.
_MODELED_ORIGIN = "ld_library_path"

# Package-local candidates are the only cross-origin evidence the frozen
# contract allows: the current-Python same-name corroboration behind
# Class B and the environment-observed family baselines (amendment
# §11/§20). They never establish loader precedence.
_BASELINE_ORIGIN = "python_package"

# Identity reliabilities accepted for package-local FAMILY BASELINE
# derivation only (amendment §7/§8): a name-claimed family (DIRECT) or a
# bare name's reliably derived canonical target family (DERIVED_STRONG).
# This is the ONLY place ``candidate.reliability`` is read in D004 — it
# never qualifies a conflict target, a load name's relevance, or loader
# precedence.
_BASELINE_RELIABILITIES = frozenset(
    (IdentityReliability.DIRECT, IdentityReliability.DERIVED_STRONG)
)


class D004RuntimeLibraryConflict:
    """Emit D004 when an observed exact versioned load name misleads.

    A positive requires, per family with a known baseline, an
    LD_LIBRARY_PATH candidate whose exact observed versioned name claims
    the baseline family while its canonical target family differs, plus
    either a strictly later same-name baseline-compatible provider
    (Class A) or an independently observed current-Python same-name
    baseline-compatible provider (Class B). Anything less — bare names,
    different names, irrelevant name families, unknown or ambiguous
    earliest targets, uncorroborated aliases, cross-origin winner
    questions — is UNKNOWN and emits nothing.
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
class _ExactNameFinding:
    """One family's conflict over one observed exact load name.

    Carries only bounded, presentation-safe identity — family, the exact
    observed name, the baseline family it claims, the conflicting
    canonical target family, the conflict class, and the modeled
    position — never a path, entry text, or any raw inventory content
    (R2 plan §24).
    """

    family: str
    load_name: str
    baseline: str
    conflicting_target_family: str
    conflict_class: Literal["A", "B"]
    modeled_order: int


def _baseline_identity(candidate: RuntimeLibraryCandidateFact) -> str | None:
    """A package-local candidate's FAMILY BASELINE identity (plan §6).

    Scoped to family-baseline derivation and nothing else: the observed
    name's claimed family when versioned, else the reliably derived
    canonical target family for a bare alias, else nothing. This is not
    a conflict identity — conflict predicates read the split identity
    fields directly and never this helper.
    """
    if candidate.reliability not in _BASELINE_RELIABILITIES:
        return None
    if candidate.load_name_family_version is not None:
        return candidate.load_name_family_version
    return candidate.target_family_version


def _baseline(family: str, libraries: RuntimeLibraryFacts) -> str | None:
    """The family's expected family baseline, or None for UNKNOWN.

    libcudart compares against the PyTorch CUDA runtime family directly
    (§11A/§19.8) — a family label only, never a filename, and never a
    name synthesized from it. libcublas is contextualized to the current
    PyTorch CUDA environment: the frozen v0.2 mapping is the environment
    itself, so the torch family must be known and exactly one distinct
    reliable package-local libcublas family must be observed (§19.9);
    zero or several disagreeing ones are UNKNOWN, and no numeric
    CUDA-major equivalence is ever assumed. libcudnn never touches the
    torch family — its baseline is the environment-local package cuDNN
    family already normalized in the facts (§11C/§19.10).
    """
    if family == "libcudart":
        return libraries.torch_runtime_family
    if family == "libcudnn":
        return libraries.package_local_cudnn_family
    # libcublas: the environment-observed package-local family, under a
    # known torch family.
    if not libraries.torch_runtime_family:
        return None
    observed = {
        identity
        for identity in (
            _baseline_identity(candidate)
            for candidate in libraries.candidates
            if candidate.family == family and candidate.origin == _BASELINE_ORIGIN
        )
        if identity is not None
    }
    if len(observed) != 1:
        # Zero or several disagreeing package-local families: the mapping
        # is not reliable, so the libcublas relation is UNKNOWN (§19.9).
        return None
    return next(iter(observed))


def _family_finding(
    family: str, libraries: RuntimeLibraryFacts
) -> _ExactNameFinding | None:
    """Establish one family's conflict over exact load names (§19.8).

    Scans each distinct exact versioned load name observed in the
    modeled LD_LIBRARY_PATH ordering (amendment §20): relevance requires
    the name itself to claim the family baseline (rows Q/R); the unique
    earliest provider must resolve to a known target family different
    from the baseline — an unknown earliest target is never skipped over
    (row H), two distinct candidates at the earliest position are
    ambiguous (§19.5), and an earliest target equal to the baseline is
    the healthy or normal minor-symlink case (N07/N08). Class A fires
    when a strictly later same-name provider is baseline-compatible;
    Class B when an independently observed current-Python package-local
    provider of the same exact name is. When several names are
    independently positive, one representative is selected
    deterministically (plan §17): Class A before Class B, then lower
    modeled order, then the lexical exact name — presentation
    determinism that never changes whether the family is positive.
    """
    baseline = _baseline(family, libraries)
    if baseline is None:
        return None
    # Exact duplicate facts collapse before any multiplicity decision,
    # so duplicates neither manufacture providers nor block emission
    # (row P); distinct exact names never collapse into one group
    # (row S). Only ordered, versioned, modeled-origin candidates carry
    # precedence — everything else is inventory.
    modeled = sorted(
        {
            candidate
            for candidate in libraries.candidates
            if candidate.family == family
            and candidate.origin == _MODELED_ORIGIN
            and candidate.modeled_search_order is not None
            and candidate.load_name_family_version is not None
        },
        key=lambda candidate: (
            candidate.load_name,
            candidate.modeled_search_order or 0,
            candidate.target_family_version or "",
        ),
    )
    package_local = [
        candidate
        for candidate in libraries.candidates
        if candidate.family == family and candidate.origin == _BASELINE_ORIGIN
    ]
    positives: list[_ExactNameFinding] = []
    for name in sorted({candidate.load_name for candidate in modeled}):
        providers = [
            candidate for candidate in modeled if candidate.load_name == name
        ]
        earliest_order = min(
            candidate.modeled_search_order or 0 for candidate in providers
        )
        earliest = [
            candidate
            for candidate in providers
            if (candidate.modeled_search_order or 0) == earliest_order
        ]
        if len(earliest) != 1:
            # Two distinct candidates share the earliest modeled position:
            # choosing between them needs per-library loader semantics the
            # model does not cover (§19.5) — resolution UNKNOWN.
            continue
        preferred = earliest[0]
        # Relevance gate (Class A/B condition 3): the observed name itself
        # must claim the family baseline — validation of an observed name,
        # never generation of an expected one (rows Q/R).
        if preferred.load_name_family_version != baseline:
            continue
        if preferred.target_family_version is None:
            # Row H: the earliest provider's target family is unknown;
            # skipping ahead would fabricate precedence — UNKNOWN.
            continue
        if preferred.target_family_version == baseline:
            # The earliest same-name target already is the expected
            # family: the healthy case, or a normal minor-version symlink
            # whose divergent later same-name aliases stay shadowed
            # (N07/N08).
            continue
        later_compatible = any(
            (candidate.modeled_search_order or 0) > earliest_order
            and candidate.target_family_version == baseline
            for candidate in providers
        )
        if later_compatible:
            # Class A (rows C/D): same-name shadowing established entirely
            # inside the modeled ordering — no cross-origin precedence.
            positives.append(
                _ExactNameFinding(
                    family=family,
                    load_name=name,
                    baseline=baseline,
                    conflicting_target_family=preferred.target_family_version,
                    conflict_class="A",
                    modeled_order=earliest_order,
                )
            )
            continue
        corroborated = any(
            package_candidate.load_name == name
            and package_candidate.target_family_version == baseline
            for package_candidate in package_local
        )
        if corroborated:
            # Class B (row F): an independently observed current-Python
            # provider of the same exact name is baseline-compatible — a
            # conditional static risk, never a cross-origin winner claim.
            positives.append(
                _ExactNameFinding(
                    family=family,
                    load_name=name,
                    baseline=baseline,
                    conflicting_target_family=preferred.target_family_version,
                    conflict_class="B",
                    modeled_order=earliest_order,
                )
            )
        # else: an uncorroborated misleading alias (row G) or a later
        # same-name target that is unknown (row I) — UNKNOWN for this
        # name, no D004.
    if not positives:
        return None
    representative = min(
        positives,
        key=lambda finding: (
            finding.conflict_class != "A",
            finding.modeled_order,
            finding.load_name,
        ),
    )
    return representative


def _describe(finding: _ExactNameFinding) -> str:
    """One finding's clause, per class, in the conditional register."""
    if finding.conflict_class == "A":
        return (
            f"for {finding.family}, within the modeled LD_LIBRARY_PATH "
            f"lookup for the observed versioned name {finding.load_name} "
            f"(claims family {finding.baseline}), the incompatible target "
            f"family {finding.conflicting_target_family} at entry "
            f"{finding.modeled_order} precedes a compatible same-name "
            "target"
        )
    return (
        f"for {finding.family}, an LD_LIBRARY_PATH provider for the "
        f"observed versioned alias {finding.load_name} (claims family "
        f"{finding.baseline}) resolves to family "
        f"{finding.conflicting_target_family}, a different target family "
        "than an independently observed current-Python compatible "
        "provider of the same alias"
    )


def _summary(findings: Sequence[_ExactNameFinding]) -> str:
    """Conditional modeled-lookup wording only (semantics §23.7, §25).

    Every clause stays within the modeled same-name lookup (Class A) or
    the corroborated-alias relation (Class B) — correct for both
    classes, with no claim about which object any process actually
    loaded and no cross-origin winner claim.
    """
    described = "; ".join(_describe(finding) for finding in findings)
    return (
        "Runtime-library alias resolution conflict within the modeled "
        f"LD_LIBRARY_PATH ordering: {described}. The static inventory "
        "indicates a conditional resolution risk; the actual loaded "
        "object is not observed by this analysis."
    )


def _evidence(findings: Sequence[_ExactNameFinding]) -> list[EvidenceItem]:
    """Bounded, privacy-safe evidence (plan §26, amendment §20 formats).

    Family, the exact observed load-name basename, the claimed baseline
    family, the conflicting canonical target family, the shadowed or
    corroborated expected provider, and the normalized static relation —
    never a library path, LD_LIBRARY_PATH contents, conda prefix,
    site-packages location, scan-error text, or any reliability label
    presented as conflict evidence.
    """
    items: list[EvidenceItem] = []
    for finding in findings:
        items.append(
            EvidenceItem(
                key=f"{finding.family}_conflicting",
                value=(
                    f"observed versioned load name {finding.load_name} "
                    f"(claims family {finding.baseline}) at "
                    f"LD_LIBRARY_PATH entry {finding.modeled_order} "
                    f"resolves to family {finding.conflicting_target_family} "
                    "(derived strongly from its symlink target)"
                ),
                source="runtime_libraries",
            )
        )
        expected = (
            f"family {finding.baseline} provider of {finding.load_name} "
            "shadowed at a later entry"
            if finding.conflict_class == "A"
            else (
                f"family {finding.baseline} package-local provider of "
                f"{finding.load_name}"
            )
        )
        items.append(
            EvidenceItem(
                key=f"{finding.family}_expected",
                value=expected,
                source=(
                    "torch" if finding.family == "libcudart" else "runtime_libraries"
                ),
            )
        )
    items.append(
        EvidenceItem(
            key="static_relation",
            value=(
                "same-name shadowing / misleading alias on an observed "
                "versioned load name within the modeled LD_LIBRARY_PATH "
                "ordering"
            ),
            source="compatibility",
        )
    )
    return items


def _impact() -> list[ImpactItem]:
    """Expected impact, preserving the §23.3 existence/precedence/loaded
    distinction inside the diagnosis output itself (plan §27: alias
    resolution at risk; no certainty claim)."""
    return [
        ImpactItem(
            scope="CUDA runtime library resolution",
            status=(
                "alias resolution at risk under the modeled or "
                "conditional LD_LIBRARY_PATH semantics"
            ),
        ),
        ImpactItem(
            scope="PyTorch CUDA initialization",
            status="may be affected if the modeled or conditional resolution applies",
        ),
        ImpactItem(
            scope="actually loaded library",
            status="not observed (static analysis only)",
        ),
    ]


def _remediation() -> list[RemediationStep]:
    """Read-only SAFE guidance (truth tables §19.15, R2 plan §28).

    D004 cannot know which runtime-library source the user intends or
    which entry is safe to change, so every step carries ``command=None``
    and no path, version, entry text, or package-manager action is
    invented. Deleting or overwriting shared libraries is never
    recommended; only the search configuration may be simplified. For a
    Class B finding nothing asserts that the package-local copy would
    win once the override is removed — that precedence is not modeled.
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

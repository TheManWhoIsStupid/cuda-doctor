"""D001 — CUDA toolkit resolution conflict.

Diagnoses *active* CUDA development selectors that meaningfully disagree
about the selected CUDA toolkit: two or more active, valid selectors
with derivable identities whose reliably known normalized toolkit
versions differ. Installation multiplicity, root-path inequality, and
UNKNOWN identities never trigger it (frozen semantics §20, truth tables
§16). The rule is pure over ``DiagnosisFacts`` — no collection, no host
state, no issue re-derivation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cuda_doctor.core.enums import Platform, Severity
from cuda_doctor.diagnosis.facts import (
    DiagnosisFacts,
    IdentityReliability,
    ToolkitSelectorFact,
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

__all__ = ["D001ToolkitResolutionConflict"]

_CODE = "D001"
_TITLE = "CUDA toolkit resolution conflict"

# Reliabilities that support the strongest frozen D001 confidence when
# both sides of a version disagreement carry them (truth tables §7,
# §16.3: directly or strongly and reliably derived).
_STRONG_RELIABILITIES = frozenset(
    (IdentityReliability.DIRECT, IdentityReliability.DERIVED_STRONG)
)


class D001ToolkitResolutionConflict:
    """Emit D001 when active selectors disagree on the CUDA toolkit.

    Conflict means reliably different normalized toolkit versions among
    active, valid, identity-known selectors — never installation
    multiplicity or differing roots with agreeing versions.
    """

    code = _CODE

    def evaluate(
        self,
        facts: DiagnosisFacts,
        issues: Sequence[Issue],
        prior_candidates: Mapping[str, DiagnosisCandidate] = NO_PRIOR_CANDIDATES,
    ) -> DiagnosisCandidate | None:
        # D001 consumes no issue or upstream-candidate context
        # (architecture §15; Phase-6 plan §11) — both parameters stay
        # intentionally unused.
        if not facts.applicability.d001:
            # Not evaluated on this platform: no candidate, no error.
            return None
        participating = _identity_known_selectors(facts.toolkit_selectors)
        disagreement = _disagreeing_pairs(participating)
        if not disagreement:
            return None
        diagnosis = Diagnosis(
            code=_CODE,
            title=_TITLE,
            severity=Severity.WARNING,
            confidence=_confidence(disagreement),
            summary=_summary(participating),
            # The resolver (Phase 10) owns final roles: candidates are
            # constructed without one.
            role=None,
            evidence=_evidence(participating),
            impact=_impact(),
            remediation=_remediation(participating),
            verification=_verification(facts.platform),
        )
        # D001 explains a configuration risk, not an observed failure
        # (truth tables §12.2, §20.4): risk-only, no causal parents.
        return DiagnosisCandidate(diagnosis=diagnosis, risk_only=True)


def _identity_known_selectors(
    selectors: Sequence[ToolkitSelectorFact],
) -> tuple[ToolkitSelectorFact, ...]:
    """Selectors eligible for D001 conflict reasoning.

    Frozen trigger preconditions (truth tables §16.1): ``active`` and
    ``valid`` both TRUE, plus a derivable identity — a known normalized
    toolkit version with non-UNKNOWN reliability. Inactive selectors,
    invalid selectors, and unknown identities never participate.
    """

    def has_known_identity(fact: ToolkitSelectorFact) -> bool:
        return (
            fact.active is True
            and fact.valid is True
            and fact.toolkit_version is not None
            and fact.reliability is not IdentityReliability.UNKNOWN
        )

    return tuple(fact for fact in selectors if has_known_identity(fact))


def _disagreeing_pairs(
    participating: Sequence[ToolkitSelectorFact],
) -> tuple[tuple[ToolkitSelectorFact, ToolkitSelectorFact], ...]:
    """All participating pairs whose normalized toolkit versions differ.

    Only normalized "major.minor" version strings are compared; canonical
    roots never participate — path inequality alone is never a version
    disagreement (semantics §20, truth tables §16.4). Pair order follows
    the facts tuple, so the result is deterministic.
    """
    pairs: list[tuple[ToolkitSelectorFact, ToolkitSelectorFact]] = []
    for index, left in enumerate(participating):
        for right in participating[index + 1 :]:
            if left.toolkit_version != right.toolkit_version:
                pairs.append((left, right))
    return tuple(pairs)


def _confidence(
    disagreement: Sequence[tuple[ToolkitSelectorFact, ToolkitSelectorFact]],
) -> Confidence:
    """Frozen confidence mapping (truth tables §7, §16.3).

    HIGH when some disagreeing pair is directly or strongly and
    reliably derived on both sides; MEDIUM once every disagreement
    involves weaker (derived_weak) identity evidence.
    """
    for left, right in disagreement:
        if (
            left.reliability in _STRONG_RELIABILITIES
            and right.reliability in _STRONG_RELIABILITIES
        ):
            return Confidence.HIGH
    return Confidence.MEDIUM


def _ordered(participating: Sequence[ToolkitSelectorFact]) -> list[ToolkitSelectorFact]:
    """Deterministic presentation order: by selector name."""
    return sorted(participating, key=lambda fact: fact.name)


def _summary(participating: Sequence[ToolkitSelectorFact]) -> str:
    """Bounded narrative over selector names and versions only."""
    resolutions = "; ".join(
        f"{fact.name} resolves CUDA {fact.toolkit_version}"
        for fact in _ordered(participating)
    )
    return (
        "Active CUDA development selectors disagree about the selected "
        f"CUDA toolkit: {resolutions}."
    )


def _evidence(participating: Sequence[ToolkitSelectorFact]) -> list[EvidenceItem]:
    """Bounded, privacy-safe evidence (Phase-6 plan §9, semantics §8.3).

    Selector name, normalized CUDA version, and reliability only — one
    item per participating selector, deterministically ordered; never a
    host path, environment dump, or collector error text.
    """
    return [
        EvidenceItem(
            key=fact.name,
            value=f"CUDA {fact.toolkit_version} ({fact.reliability.value})",
            source="toolkit_selectors",
        )
        for fact in _ordered(participating)
    ]


def _impact() -> list[ImpactItem]:
    """Frozen expected impact (truth tables §16.3)."""
    return [
        ImpactItem(scope="system CUDA configuration", status="inconsistent"),
        ImpactItem(scope="CUDA development toolchain", status="inconsistent"),
        ImpactItem(scope="CUDA extension build", status="at risk"),
        ImpactItem(scope="PyTorch runtime", status="not automatically broken"),
    ]


def _remediation(participating: Sequence[ToolkitSelectorFact]) -> list[RemediationStep]:
    """Read-only SAFE guidance (truth tables §16.12, semantics §16-§18).

    D001 proves only that the active selectors disagree — it cannot know
    which conflicting toolkit the user intends to keep. No toolkit
    version or root is selected, synthesized, or hard-coded, so every
    configuration-changing step carries ``command=None``: there is no
    facts-supported concrete target value. Steps are emitted per
    participating selector (evidence-tied and selector-specific — never
    a CUDA_PATH instruction for a CUDA_HOME-only conflict); removing
    installations is never recommended. Nothing is executed.
    """
    names = {fact.name for fact in participating}
    steps: list[RemediationStep] = [
        RemediationStep(
            action=(
                "Choose the CUDA toolkit you intend this shell/build "
                "environment to use; the alignment steps below assume "
                "that choice"
            ),
            why=(
                "The active CUDA development selectors disagree about the "
                "selected CUDA toolkit; the diagnosis cannot know which "
                "toolkit you intend to keep"
            ),
            safety=RemediationSafety.SAFE,
        )
    ]
    if "CUDA_HOME" in names:
        steps.append(
            RemediationStep(
                action=(
                    "Align CUDA_HOME to the intended toolkit so the "
                    "environment selector agrees with the other active "
                    "selectors"
                ),
                why=(
                    "CUDA extension builds commonly use CUDA_HOME and nvcc "
                    "to locate CUDA development components; a CUDA_HOME that "
                    "resolves a different CUDA version makes them select "
                    "components from an unintended toolkit"
                ),
                safety=RemediationSafety.SAFE,
            )
        )
    if "CUDA_PATH" in names:
        steps.append(
            RemediationStep(
                action=(
                    "Align CUDA_PATH to the intended toolkit so the "
                    "environment selector agrees with the other active "
                    "selectors"
                ),
                why=(
                    "CUDA extension builds commonly use CUDA_PATH and nvcc "
                    "to locate CUDA development components; a CUDA_PATH that "
                    "resolves a different CUDA version makes them select "
                    "components from an unintended toolkit"
                ),
                safety=RemediationSafety.SAFE,
            )
        )
    if "CUDACXX" in names:
        steps.append(
            RemediationStep(
                action=(
                    "Align CUDACXX to the intended toolkit's nvcc so the "
                    "explicitly selected compiler agrees with the other "
                    "active selectors"
                ),
                why=(
                    "Build systems honoring CUDACXX invoke the explicitly "
                    "selected compiler; a CUDACXX from a different CUDA "
                    "version produces an inconsistent toolchain"
                ),
                safety=RemediationSafety.SAFE,
            )
        )
    if "NVCC" in names or "PATH_CUDA_BIN" in names:
        steps.append(
            RemediationStep(
                action=(
                    "Adjust PATH so nvcc resolves from the intended "
                    "toolkit, then open a fresh shell"
                ),
                why=(
                    "The PATH-resolved nvcc is an active selector; which "
                    "PATH entry wins determines its toolkit, and a fresh "
                    "shell picks up the adjusted environment"
                ),
                safety=RemediationSafety.SAFE,
            )
        )
    return steps


def _verification(platform: Platform) -> list[str]:
    """Frozen verification minimum (truth tables §16.13): the
    platform-applicable subset of the four commands, in Windows syntax
    on Windows (architecture §40).
    """
    if platform is Platform.WINDOWS:
        return ["echo %CUDA_PATH%", "echo %CUDACXX%", "where nvcc", "nvcc --version"]
    return ["echo $CUDA_HOME", "echo $CUDACXX", "which nvcc", "nvcc --version"]

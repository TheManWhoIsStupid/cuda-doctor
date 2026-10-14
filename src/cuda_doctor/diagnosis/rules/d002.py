"""D002 — PyTorch/local CUDA toolkit build mismatch.

Diagnoses a material CUDA major-version mismatch between the installed
PyTorch build and the active local CUDA development toolkit
(``facts.reference_toolkit``, the frozen Phase-5 precedence result).
This is not a claim that normal PyTorch runtime use is broken — official
PyTorch wheels bundle their own CUDA runtime — so the mismatch matters
primarily for compiling CUDA extensions and local CUDA development.
Exact and same-major differences never trigger it (the legacy
informational TORCH004 issue stays authoritative there), and UNKNOWN
comparison information never becomes mismatch (frozen semantics §21,
truth tables §17). The rule is pure over ``DiagnosisFacts`` — no
collection, no host state, no issue re-derivation, no selector
re-ranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cuda_doctor.core.enums import Platform, Severity
from cuda_doctor.diagnosis.facts import (
    DiagnosisFacts,
    IdentityReliability,
    ReferenceToolkit,
    ToolkitCompatibility,
    TorchDiagnosisFacts,
    normalize_torch_toolkit_relation,
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
from cuda_doctor.utils.versions import parse_cuda_version

__all__ = ["D002TorchToolkitMismatch"]

_CODE = "D002"
_TITLE = "PyTorch/local toolkit build mismatch"

# Architecture §25.2 assigns TORCH004 presentation-suppression ownership
# to D002 (v0.2.0 family map): the carrier is populated on every emitted
# D002. Suppression itself is a later presentation concern — the issue
# always stays machine-readable in ``issues``.
_LEGACY_ISSUE_CODE = "TORCH004"


class D002TorchToolkitMismatch:
    """Emit D002 when PyTorch and the reference toolkit differ by CUDA major.

    The reference is consumed from ``facts.reference_toolkit`` (valid
    CUDACXX → resolved active NVCC → usable CUDA_HOME/CUDA_PATH: frozen
    Phase-5 precedence); the rule never re-ranks selectors. Compatibility
    is decided by the shared Phase-5 adapter, never a second comparison
    table.
    """

    code = _CODE

    def evaluate(
        self,
        facts: DiagnosisFacts,
        issues: Sequence[Issue],
        prior_candidates: Mapping[str, DiagnosisCandidate] = NO_PRIOR_CANDIDATES,
    ) -> DiagnosisCandidate | None:
        # D002 is independent of D001 under the frozen tables (truth
        # tables §20.1-§20.3): no issue prose and no upstream-candidate
        # context is consumed — both parameters stay intentionally unused.
        if not facts.applicability.d002:
            # Not evaluated on this platform: no candidate, no error.
            return None
        comparison = _comparison(facts.torch, facts.reference_toolkit)
        if comparison is None:
            return None
        torch_version, toolkit_version, reference = comparison
        relation = normalize_torch_toolkit_relation(
            parse_cuda_version(torch_version), parse_cuda_version(toolkit_version)
        )
        if relation is not ToolkitCompatibility.MAJOR_INCOMPATIBLE:
            # EXACT and same-major differences stay informational (the
            # legacy TORCH004 issue remains authoritative there);
            # UNKNOWN is not mismatch (truth tables §17.3-§17.6).
            return None
        confidence = _confidence(reference.reliability)
        if confidence is None:
            # Weakly derived reference identity is LOW-confidence
            # evidence, and LOW-confidence D002 is not emitted at all,
            # in any role (truth tables §17.10).
            return None
        diagnosis = Diagnosis(
            code=_CODE,
            title=_TITLE,
            severity=Severity.WARNING,
            confidence=confidence,
            summary=_summary(torch_version, toolkit_version, reference),
            # The resolver (Phase 10) owns final roles: candidates are
            # constructed without one.
            role=None,
            evidence=_evidence(torch_version, reference),
            impact=_impact(),
            remediation=_remediation(),
            verification=_verification(facts.platform),
            related_issues=_related_issues(facts.issue_codes),
            presentation_suppressed_issue_codes=[_LEGACY_ISSUE_CODE],
        )
        # D002 describes an extension-build risk, not an observed runtime
        # failure (truth tables §17.4, §20.1): risk-only, no causal
        # parents.
        return DiagnosisCandidate(diagnosis=diagnosis, risk_only=True)


def _comparison(
    torch: TorchDiagnosisFacts, reference_toolkit: ReferenceToolkit | None
) -> tuple[str, str, ReferenceToolkit] | None:
    """Establish the frozen comparison context (truth tables §17.1).

    Returns the PyTorch CUDA build version, the reference toolkit's
    normalized version, and the reference itself — or None whenever
    either comparison side is unavailable: PyTorch not importable
    (or absent), a CPU-only or unknown torch CUDA build, no reference
    toolkit, or a reference whose version is underivable or whose
    reliability cannot prove anything. Absence never implies mismatch;
    healthy wheel-only use without a local toolkit is not D002.
    """
    if not torch.importable or torch.cuda_build is not True or not torch.cuda_version:
        return None
    if (
        reference_toolkit is None
        or not reference_toolkit.toolkit_version
        or reference_toolkit.reliability is IdentityReliability.UNKNOWN
    ):
        return None
    return torch.cuda_version, reference_toolkit.toolkit_version, reference_toolkit


def _confidence(reliability: IdentityReliability) -> Confidence | None:
    """Frozen confidence mapping (truth tables §17.10).

    HIGH when the torch CUDA version (always directly known here) is
    compared against a directly known reference version; MEDIUM when the
    reference version is strongly derived. Weakly derived identity is
    LOW-confidence evidence and LOW-confidence D002 is not emitted in
    v0.2.0 — None means "do not emit".
    """
    if reliability is IdentityReliability.DIRECT:
        return Confidence.HIGH
    if reliability is IdentityReliability.DERIVED_STRONG:
        return Confidence.MEDIUM
    return None  # DERIVED_WEAK → LOW → not emitted (truth tables §17.10)


def _summary(
    torch_version: str, toolkit_version: str, reference: ReferenceToolkit
) -> str:
    """Bounded narrative over names and normalized versions only."""
    return (
        "The installed PyTorch build targets CUDA "
        f"{torch_version} while the active local development toolkit "
        f"({reference.source}) resolves CUDA {toolkit_version}; the "
        "differing CUDA major versions make CUDA extension builds "
        "against this toolkit likely incompatible."
    )


def _evidence(torch_version: str, reference: ReferenceToolkit) -> list[EvidenceItem]:
    """Bounded, privacy-safe evidence (Phase-7 plan §11, semantics §8.3).

    PyTorch CUDA build version, reference toolkit source/version/
    reliability, and the normalized compatibility relation only — never
    a canonical root, raw path, environment dump, or selector value.
    """
    return [
        EvidenceItem(
            key="torch_cuda",
            value=f"CUDA {torch_version}",
            source="torch",
        ),
        EvidenceItem(
            key="reference_source",
            value=reference.source,
            source="toolkit_selectors",
        ),
        EvidenceItem(
            key="reference_toolkit",
            value=f"CUDA {reference.toolkit_version} ({reference.reliability.value})",
            source="toolkit_selectors",
        ),
        EvidenceItem(
            key="compatibility",
            value=ToolkitCompatibility.MAJOR_INCOMPATIBLE.value,
            source="compatibility",
        ),
    ]


def _impact() -> list[ImpactItem]:
    """Frozen expected impact (truth tables §17.4)."""
    return [
        ImpactItem(scope="PyTorch runtime", status="may remain healthy"),
        ImpactItem(scope="CUDA extension build", status="high risk / likely incompatible"),
        ImpactItem(scope="local CUDA development", status="mismatched"),
    ]


def _remediation() -> list[RemediationStep]:
    """Read-only SAFE guidance (truth tables §17.11, semantics §21.11).

    D002 knows a major-version mismatch exists but not which side the
    user intends to change, so it presents the two legitimate strategies
    without selecting one: align the local development toolkit to the
    PyTorch build when compiling extensions, or use a PyTorch build
    compatible with the intended toolkit. Ordinary prebuilt-wheel use
    needs no change (wheels bundle their own CUDA runtime); no toolkit
    version, root, URL, or command is invented — there is no
    facts-supported concrete target value, so every step carries
    ``command=None``. Nothing is executed.
    """
    return [
        RemediationStep(
            action=(
                "No change is required for ordinary prebuilt-wheel "
                "PyTorch use; official PyTorch builds bundle their own "
                "CUDA runtime libraries"
            ),
            why=(
                "A PyTorch/local toolkit version difference does not by "
                "itself break PyTorch runtime use; it matters when "
                "compiling CUDA extensions or developing locally "
                "against the toolkit"
            ),
            safety=RemediationSafety.SAFE,
        ),
        RemediationStep(
            action=(
                "When compiling CUDA extensions, use a local CUDA "
                "development toolkit whose CUDA major version matches "
                "the PyTorch build"
            ),
            why=(
                "Extension builds compile and link against the local "
                "development toolkit; differing CUDA major versions "
                "risk ABI incompatibilities"
            ),
            safety=RemediationSafety.SAFE,
        ),
        RemediationStep(
            action=(
                "Alternatively, install or use a PyTorch build whose "
                "CUDA version is compatible with the intended local "
                "development toolkit"
            ),
            why=(
                "Aligning the PyTorch build with the toolkit you intend "
                "to keep also removes the extension-build mismatch; "
                "this is a choice for extension development, not a "
                "requirement for wheel-only runtime use"
            ),
            safety=RemediationSafety.SAFE,
        ),
    ]


def _verification(platform: Platform) -> list[str]:
    """Frozen verification minimum (truth tables §17.12): the
    platform-applicable commands, in Windows syntax on Windows
    (architecture §40).
    """
    if platform is Platform.WINDOWS:
        return [
            "python -c \"import torch; print(torch.version.cuda); "
            "print(torch.cuda.is_available())\"",
            "where nvcc",
            "nvcc --version",
            "echo %CUDACXX%",
        ]
    return [
        "python -c \"import torch; print(torch.version.cuda); "
        "print(torch.cuda.is_available())\"",
        "which nvcc",
        "nvcc --version",
        "echo $CUDACXX",
    ]


def _related_issues(issue_codes: frozenset[str]) -> list[str]:
    """The legacy informational issue, when the environment produced it.

    TORCH004 stays machine-readable and remains the authoritative
    rendering for same-major informational differences (semantics
    §21.6); it is listed as related evidence only when it actually
    fired.
    """
    return [_LEGACY_ISSUE_CODE] if _LEGACY_ISSUE_CODE in issue_codes else []

"""D003 — PyTorch CUDA unavailable.

Diagnoses the observed symptom ``torch.cuda.is_available() == False`` by
selecting exactly one root cause from the frozen deterministic cause
order (truth tables §18.3, semantics §22.4): CPU-only PyTorch build,
independently invisible GPU, unusable driver, driver incompatible with
the PyTorch CUDA runtime family, direct runtime-initialization failure
evidence, a supported D004 runtime-library conflict, an explicit
CUDA_VISIBLE_DEVICES mask-all, or UNKNOWN. Later causes are considered
only when every earlier branch fails to establish.

The rule never re-derives the symptom: a clean ``False`` is required to
enter the tree at all. ``True`` means no symptom, and ``None`` — the
probe produced no reliable boolean, including the raised-probe state
carried by the snapshot's probe-error field — means D003 is not
evaluated (truth tables §18.1/§18.8); ``None`` never collapses onto
``False``, and a torch import failure is never treated as CUDA
unavailability. Because direct runtime-init failure evidence only
exists in the ``None`` state, the RUNTIME_INITIALIZATION_FAILURE cause
stays unreachable in v0.2.0, which the frozen table explicitly accepts
(§18.2). GPU visibility comes from the independent nvidia-smi path only,
driver compatibility is the already-normalized Phase-5 verdict, and D004
is consumed solely through ``prior_candidates`` presence as MEDIUM
static evidence — never as proof of the actually loaded object. The raw
probe-error text is never read: it can contain private paths, and no
branch of the tree needs it. The rule is pure over ``DiagnosisFacts`` —
no collection, no host state, no issue-prose classification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cuda_doctor.core.enums import Platform, Severity
from cuda_doctor.diagnosis.facts import (
    CUDAVisibleDevicesState,
    DiagnosisFacts,
    TorchDiagnosisFacts,
    Trinary,
)
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.models import (
    Confidence,
    Diagnosis,
    DiagnosisCandidate,
    DiagnosisRelationship,
    EvidenceItem,
    ImpactItem,
    RemediationSafety,
    RemediationStep,
)
from cuda_doctor.diagnosis.rules.base import NO_PRIOR_CANDIDATES

__all__ = ["D003PyTorchCudaUnavailable"]

_CODE = "D003"
_TITLE = "PyTorch CUDA unavailable"

# Architecture §25.2 assigns TORCH002 presentation-suppression ownership
# to D003 (v0.2.0 family map: TORCH002 -> D003): the carrier is populated
# on every emitted D003, including the UNKNOWN-cause variant. Suppression
# is presentation metadata only — the issue always stays machine-readable.
_LEGACY_ISSUE_CODE = "TORCH002"

# D004 is consumed only through the frozen prior_candidates contract
# (architecture §16/§18.1): presence means "evaluated and established",
# absence means "upstream state unavailable" and is never negative
# evidence (plan §17.4). D003 never invokes D004 itself.
_PRIOR_CONFLICT_CODE = "D004"

# Frozen v0.2.0 normalized cause classes (truth tables §18.2).
_CAUSE_CPU_ONLY = "CPU_ONLY_PYTORCH"
_CAUSE_GPU_NOT_VISIBLE = "GPU_NOT_VISIBLE"
_CAUSE_DRIVER_UNUSABLE = "DRIVER_UNUSABLE"
_CAUSE_DRIVER_INCOMPATIBLE = "DRIVER_INCOMPATIBLE"
_CAUSE_RUNTIME_LIBRARY_CONFLICT = "RUNTIME_LIBRARY_CONFLICT"
_CAUSE_EXPLICIT_VISIBILITY_MASK = "EXPLICIT_VISIBILITY_MASK"
_CAUSE_UNKNOWN = "UNKNOWN"


class D003PyTorchCudaUnavailable:
    """Emit D003 when CUDA is cleanly observed unavailable.

    Applicability requires a supported platform, a successful torch
    import, and a reliable ``torch.cuda.is_available() is False``
    observation. The cause is then selected by the frozen order over
    independent normalized facts plus prior-candidate presence; exactly
    one candidate is emitted, ``role=None`` (the Phase-10 resolver owns
    final roles), ``explains_active_failure=True`` and ``risk_only=False``
    because D003 explains an observed active failure rather than a mere
    risk.
    """

    code = _CODE

    def evaluate(
        self,
        facts: DiagnosisFacts,
        issues: Sequence[Issue],
        prior_candidates: Mapping[str, DiagnosisCandidate] = NO_PRIOR_CANDIDATES,
    ) -> DiagnosisCandidate | None:
        # D003 classifies from normalized facts and prior-candidate
        # presence only: issue prose never classifies a cause (semantics
        # §22.6) and D001/D002 are never ad-hoc CUDA-runtime causes (plan
        # §12) — the issues parameter stays intentionally unused.
        if not facts.applicability.d003:
            # Not evaluated on this platform: no candidate, no error.
            return None
        torch_facts = facts.torch
        if not torch_facts.importable:
            # Import failure (N05/§22.7): the existing import issue stays
            # authoritative; import failure is never CUDA unavailability.
            return None
        if torch_facts.cuda_available is not False:
            # True: no symptom (N07). None: the probe produced no reliable
            # boolean — the raised-probe state included (U04/§18.8) — so
            # D003 is not evaluated; None never becomes False.
            return None
        cause = _cause(facts, prior_candidates)
        conflict: DiagnosisCandidate | None = None
        if cause == _CAUSE_RUNTIME_LIBRARY_CONFLICT:
            conflict = prior_candidates[_PRIOR_CONFLICT_CODE]
        diagnosis = Diagnosis(
            code=_CODE,
            title=_TITLE,
            severity=Severity.ERROR,
            confidence=_confidence(cause),
            summary=_summary(cause),
            # The resolver (Phase 10) owns final roles: candidates are
            # constructed without one.
            role=None,
            cause_class=cause,
            evidence=_evidence(cause, facts, conflict),
            impact=_impact(cause),
            remediation=_remediation(cause),
            verification=_verification(facts.platform),
            exclusions=_exclusions(cause, facts),
            related_issues=_related_issues(facts.issue_codes),
            # §25.2: D003 owns the TORCH002 presentation-suppression
            # carrier on every emission, UNKNOWN included.
            presentation_suppressed_issue_codes=[_LEGACY_ISSUE_CODE],
            relationships=(
                [DiagnosisRelationship(
                    kind="explains", target_code=_PRIOR_CONFLICT_CODE
                )]
                if conflict is not None
                else []
            ),
        )
        # D003 explains the directly observed unavailability itself, so it
        # outranks risk-only candidates in later resolution (§12.2) and
        # claims a causal parent only when D004 is the selected cause.
        return DiagnosisCandidate(
            diagnosis=diagnosis,
            explains_active_failure=True,
            risk_only=False,
            causal_parent_codes=(
                (_PRIOR_CONFLICT_CODE,) if conflict is not None else ()
            ),
        )


def _cause(facts: DiagnosisFacts, prior_candidates: Mapping[str, DiagnosisCandidate]) -> str:
    """The frozen deterministic cause selection (truth tables §18.3).

    Branches are evaluated exactly in the frozen order; a later cause is
    considered only when every earlier branch fails to establish. Unknown
    tri-states never become negative facts — they simply fail to
    establish their branch and fall through.
    """
    if facts.torch.cuda_build is False:
        # Branch 1 (P01/P02): a confirmed CPU-only build makes CUDA
        # unavailable regardless of GPU, driver, or visibility state.
        return _CAUSE_CPU_ONLY
    if facts.gpu.visible is Trinary.FALSE:
        # Branch 2 (P03): independent nvidia-smi-path observation only —
        # never derived from PyTorch itself or CUDA_VISIBLE_DEVICES.
        return _CAUSE_GPU_NOT_VISIBLE
    if facts.driver.usable is Trinary.FALSE:
        # Branch 3 (P04).
        return _CAUSE_DRIVER_UNUSABLE
    if facts.driver.compatible_with_torch_runtime is Trinary.FALSE:
        # Branch 4 (P05/P06/P08): the already-normalized Phase-5 verdict
        # of the shared compatibility evaluator — no second table here.
        return _CAUSE_DRIVER_INCOMPATIBLE
    # Branch 5 (RUNTIME_INITIALIZATION_FAILURE) is unreachable in v0.2.0:
    # the only direct runtime-initialization failure evidence is a raised
    # availability probe, which the collector carries with
    # cuda_available=None — a state that never enters the cause tree
    # (§18.1/§18.8). Truth tables §18.2 explicitly accept this.
    if _PRIOR_CONFLICT_CODE in prior_candidates:
        # Branch 6: supported D004 evidence. Presence in the mapping means
        # D004 was evaluated and established a conflict; absence is never
        # negative evidence and simply falls through.
        return _CAUSE_RUNTIME_LIBRARY_CONFLICT
    if facts.visibility.cuda_visible_devices is CUDAVisibleDevicesState.MASK_ALL:
        # Branch 7 (P07): only the two Phase-0-verified mask-all tokens
        # (empty value, -1) are classified MASK_ALL by the fact layer;
        # restricted-nonempty and unparseable never reach this branch
        # (N06/N08/U05).
        return _CAUSE_EXPLICIT_VISIBILITY_MASK
    # Branch 8: UNKNOWN is a legitimate diagnosis state (§18.13) — never
    # a fabricated cause from missing toolkits, D001/D002, PATH
    # multiplicity, library multiplicity, or unknown GPU/driver states.
    return _CAUSE_UNKNOWN


def _confidence(cause: str) -> Confidence:
    """Frozen confidence matrix (truth tables §18.4/§18.6/§18.11/§18.13).

    HIGH for every directly observed cause: CPU-only build identity
    (§18.4), an independently invisible GPU (a direct observation on the
    independent collection path, P03), an unusable driver (§18.6), a
    driver below the documented minimum (semantics §22.10 example), and a
    directly established mask-all state (§18.11). The D004-based cause is
    MEDIUM: D004 is MEDIUM static evidence and is never upgraded to
    loaded-object proof. UNKNOWN is LOW — root-cause attribution only;
    the symptom itself is directly observed.
    """
    if cause == _CAUSE_RUNTIME_LIBRARY_CONFLICT:
        return Confidence.MEDIUM
    if cause == _CAUSE_UNKNOWN:
        return Confidence.LOW
    return Confidence.HIGH


def _summary(cause: str) -> str:
    """Bounded narrative; no raw values, paths, or exception text.

    The UNKNOWN text must contain the frozen sentence "Unable to
    determine the root cause from available evidence." (truth tables
    §18.13) and must not invent a runtime-library explanation.
    """
    if cause == _CAUSE_CPU_ONLY:
        return (
            "PyTorch reports CUDA unavailable because the installed "
            "PyTorch build is CPU-only: torch.version.cuda is None after "
            "a successful import, so the build contains no CUDA support "
            "and torch.cuda.is_available() is false regardless of GPU, "
            "driver, or device-visibility state."
        )
    if cause == _CAUSE_GPU_NOT_VISIBLE:
        return (
            "PyTorch reports CUDA unavailable and cuda-doctor "
            "independently observes no NVIDIA GPU through its GPU and "
            "driver collection path: without a visible GPU there is no "
            "CUDA device for this PyTorch build to initialize."
        )
    if cause == _CAUSE_DRIVER_UNUSABLE:
        return (
            "PyTorch reports CUDA unavailable and the NVIDIA driver is "
            "observed unusable through cuda-doctor's independent driver "
            "collection path: PyTorch cannot initialize CUDA without a "
            "working driver, whatever the PyTorch build."
        )
    if cause == _CAUSE_DRIVER_INCOMPATIBLE:
        return (
            "PyTorch reports CUDA unavailable because the installed "
            "NVIDIA driver does not satisfy the documented minimum for "
            "this PyTorch build's CUDA runtime family (a documented "
            "family-minimum violation or a cross-generation gap). The "
            "verdict comes from cuda-doctor's existing driver-"
            "compatibility evaluation, not from the CUDA version "
            "reported by nvidia-smi."
        )
    if cause == _CAUSE_RUNTIME_LIBRARY_CONFLICT:
        return (
            "PyTorch reports CUDA unavailable and a supported D004 "
            "runtime-library conflict diagnosis is present as MEDIUM "
            "static evidence of a modeled library-resolution conflict. "
            "This is a plausible contributing cause of the observed "
            "unavailability; the actually loaded object is not observed "
            "by this analysis."
        )
    if cause == _CAUSE_EXPLICIT_VISIBILITY_MASK:
        return (
            "PyTorch reports CUDA unavailable because CUDA_VISIBLE_DEVICES"
            " explicitly masks all GPUs in this environment (an empty "
            "value or -1): with every device masked, CUDA initialization "
            "finds no visible GPU."
        )
    return (
        "PyTorch reports CUDA unavailable and none of the supported "
        "root-cause branches (CPU-only build, independent GPU "
        "visibility, driver usability, driver compatibility, direct "
        "runtime-initialization evidence, supported runtime-library "
        "conflict, explicit device-visibility mask) could be established "
        "from the collected evidence. Unable to determine the root cause "
        "from available evidence. The next diagnostic area is CUDA "
        "runtime initialization / library resolution."
    )


def _torch_build_evidence(torch_facts: TorchDiagnosisFacts) -> EvidenceItem:
    """The normalized PyTorch build state, worded per state.

    A CPU-only build is stated only from the confirmed boolean fact
    (§18.4: after a successful import, torch.version.cuda is None) —
    never inferred from an absent cuda_version alone (plan §7).
    """
    if torch_facts.cuda_build is False:
        value = "CPU-only build (torch.version.cuda is None)"
    elif torch_facts.cuda_build is True:
        version = torch_facts.cuda_version
        value = (
            f"CUDA-enabled build (CUDA {version})"
            if version
            else "CUDA-enabled build (CUDA version unknown)"
        )
    else:
        value = "CUDA build state unknown"
    return EvidenceItem(key="torch_cuda_build", value=value, source="torch")


def _evidence(
    cause: str,
    facts: DiagnosisFacts,
    conflict: DiagnosisCandidate | None,
) -> list[EvidenceItem]:
    """Bounded, privacy-safe evidence (plan §19 permitted examples only).

    Normalized states — the directly observed symptom, the build state,
    the independent GPU/driver verdicts, the normalized mask-all
    classification, and prior-candidate D004 code/confidence — never the
    raw CUDA_VISIBLE_DEVICES value, raw exception text, filesystem
    paths, environment contents, D004 library paths, or issue prose.
    """
    items = [
        EvidenceItem(
            key="cuda_available",
            value="false (directly observed)",
            source="torch",
        ),
        _torch_build_evidence(facts.torch),
    ]
    if cause == _CAUSE_GPU_NOT_VISIBLE:
        items.append(
            EvidenceItem(
                key="gpu_visible",
                value="no NVIDIA GPU independently visible",
                source="gpu",
            )
        )
    elif cause == _CAUSE_DRIVER_UNUSABLE:
        items.append(
            EvidenceItem(
                key="driver_usable",
                value="false (independent driver collection path)",
                source="driver",
            )
        )
    elif cause == _CAUSE_DRIVER_INCOMPATIBLE:
        items.append(
            EvidenceItem(
                key="driver_compatible_with_torch_runtime",
                value=(
                    "false (below the documented family minimum or a "
                    "cross-generation gap)"
                ),
                source="compatibility",
            )
        )
    elif cause == _CAUSE_RUNTIME_LIBRARY_CONFLICT:
        assert conflict is not None  # the cause implies presence
        items.append(
            EvidenceItem(
                key="runtime_library_conflict",
                value=(
                    "D004 present ("
                    f"{conflict.diagnosis.confidence.value} confidence, "
                    "static inventory evidence)"
                ),
                source="prior_diagnosis",
            )
        )
    elif cause == _CAUSE_EXPLICIT_VISIBILITY_MASK:
        items.append(
            EvidenceItem(
                key="cuda_visible_devices",
                value="explicit mask-all (empty value or -1)",
                source="visibility",
            )
        )
    if (
        cause != _CAUSE_EXPLICIT_VISIBILITY_MASK
        and facts.visibility.cuda_visible_devices is CUDAVisibleDevicesState.MASK_ALL
    ):
        # §18.12 Example A: an outranked mask may be reported as a
        # contributing fact without becoming the selected root cause.
        items.append(
            EvidenceItem(
                key="cuda_visible_devices",
                value=(
                    "explicit mask-all also present (a contributing "
                    "fact; not the selected root cause under the frozen "
                    "cause order)"
                ),
                source="visibility",
            )
        )
    return items


def _impact(cause: str) -> list[ImpactItem]:
    """The directly observed symptom plus one cause-specific scope."""
    cause_impact = {
        _CAUSE_CPU_ONLY: ImpactItem(
            scope="CUDA-accelerated PyTorch workloads",
            status="not possible with this build on any machine",
        ),
        _CAUSE_GPU_NOT_VISIBLE: ImpactItem(
            scope="CUDA device access",
            status="no independently visible NVIDIA GPU",
        ),
        _CAUSE_DRIVER_UNUSABLE: ImpactItem(
            scope="NVIDIA driver",
            status="observed unusable through the independent driver path",
        ),
        _CAUSE_DRIVER_INCOMPATIBLE: ImpactItem(
            scope="NVIDIA driver / PyTorch CUDA runtime relation",
            status="driver below the documented minimum for the CUDA family",
        ),
        _CAUSE_RUNTIME_LIBRARY_CONFLICT: ImpactItem(
            scope="CUDA runtime library resolution",
            status="static conflict evidence present; plausible, not proven",
        ),
        _CAUSE_EXPLICIT_VISIBILITY_MASK: ImpactItem(
            scope="CUDA device visibility",
            status="all GPUs explicitly masked by the environment",
        ),
        _CAUSE_UNKNOWN: ImpactItem(
            scope="root-cause attribution",
            status="undetermined from available evidence",
        ),
    }
    return [
        ImpactItem(
            scope="PyTorch CUDA availability",
            status="torch.cuda.is_available() is false (directly observed)",
        ),
        cause_impact[cause],
    ]


def _remediation(cause: str) -> list[RemediationStep]:
    """Cause-specific SAFE guidance (truth tables §18.15, plan §20).

    No destructive commands, no invented targets; cuda-doctor never
    installs or removes PyTorch, drivers, toolkits, or libraries, so
    every step carries ``command=None``. The D004-based cause defers to
    the D004 remediation without strengthening its loader claims, and
    the UNKNOWN cause prescribes only bounded verification — never a
    speculative fix.
    """
    if cause == _CAUSE_CPU_ONLY:
        return [
            RemediationStep(
                action=(
                    "The installed PyTorch build contains no CUDA "
                    "support: torch.version.cuda is None after a "
                    "successful import, so CUDA is unavailable on any "
                    "machine with this build"
                ),
                why=(
                    "This is a property of the installed build — not a "
                    "GPU, driver, toolkit, or environment problem — so no "
                    "host configuration change can enable CUDA for it"
                ),
                safety=RemediationSafety.SAFE,
            ),
            RemediationStep(
                action=(
                    "To run CUDA-accelerated PyTorch workloads, select "
                    "and install a CUDA-capable PyTorch build from the "
                    "official PyTorch distribution for the CUDA version "
                    "you intend to use"
                ),
                why=(
                    "Choosing the CUDA variant is a user decision about "
                    "which CUDA generation to target; cuda-doctor does "
                    "not install, remove, or modify PyTorch installations"
                ),
                safety=RemediationSafety.SAFE,
            ),
        ]
    if cause == _CAUSE_GPU_NOT_VISIBLE:
        return [
            RemediationStep(
                action=(
                    "Verify an NVIDIA GPU is present, powered, and "
                    "correctly installed in this machine, independent of "
                    "PyTorch"
                ),
                why=(
                    "cuda-doctor observed no NVIDIA GPU through its "
                    "independent GPU and driver collection path, so "
                    "PyTorch has no CUDA device to initialize"
                ),
                safety=RemediationSafety.SAFE,
            ),
            RemediationStep(
                action=(
                    "On a machine that should have a GPU, check GPU "
                    "visibility at the system level (hardware presence "
                    "and driver binding) before changing any software "
                    "configuration"
                ),
                why=(
                    "A missing GPU observation is a hardware and "
                    "platform state; reinstalling toolkits or PyTorch "
                    "cannot create a GPU"
                ),
                safety=RemediationSafety.SAFE,
            ),
        ]
    if cause == _CAUSE_DRIVER_UNUSABLE:
        return [
            RemediationStep(
                action=(
                    "Investigate NVIDIA driver health: confirm the "
                    "driver is installed, loaded, and communicating "
                    "(for example through nvidia-smi)"
                ),
                why=(
                    "cuda-doctor could not obtain a usable driver state "
                    "through its independent driver collection path; "
                    "PyTorch cannot initialize CUDA without a working "
                    "driver"
                ),
                safety=RemediationSafety.SAFE,
            ),
            RemediationStep(
                action=(
                    "If the driver is broken, repairing or reinstalling "
                    "it is a system-level decision for the user"
                ),
                why=(
                    "Driver repair steps vary by platform and "
                    "installation method; cuda-doctor never modifies "
                    "drivers"
                ),
                safety=RemediationSafety.SAFE,
            ),
        ]
    if cause == _CAUSE_DRIVER_INCOMPATIBLE:
        return [
            RemediationStep(
                action=(
                    "Check the documented minimum NVIDIA driver "
                    "requirement for this PyTorch build's CUDA runtime "
                    "family in the official CUDA documentation"
                ),
                why=(
                    "The incompatibility verdict comes from "
                    "cuda-doctor's documented driver-compatibility data "
                    "(family minimum or generation gap), not from the "
                    "CUDA version reported by nvidia-smi"
                ),
                safety=RemediationSafety.SAFE,
            ),
            RemediationStep(
                action=(
                    "Either update the NVIDIA driver to at least the "
                    "documented minimum for that CUDA family, or use a "
                    "PyTorch build whose CUDA family matches the "
                    "installed driver"
                ),
                why=(
                    "Both are legitimate resolutions and the choice "
                    "depends on which component you intend to keep; "
                    "cuda-doctor does not modify drivers or PyTorch "
                    "installations"
                ),
                safety=RemediationSafety.SAFE,
            ),
        ]
    if cause == _CAUSE_RUNTIME_LIBRARY_CONFLICT:
        return [
            RemediationStep(
                action=(
                    "Review the D004 runtime-library conflict diagnosis "
                    "and follow its remediation for the modeled "
                    "library-resolution conflict"
                ),
                why=(
                    "D004 is MEDIUM-confidence static inventory evidence "
                    "of a plausible library conflict; it is consumed here "
                    "as a plausible contributing cause, never as proof "
                    "of the actually loaded object"
                ),
                safety=RemediationSafety.SAFE,
            ),
            RemediationStep(
                action=(
                    "Re-run cuda-doctor after adjusting the environment "
                    "to see whether the library conflict and the CUDA "
                    "unavailability resolve together"
                ),
                why=(
                    "Coordinated resolution would corroborate the "
                    "library conflict as the cause; continued failure "
                    "points the investigation elsewhere"
                ),
                safety=RemediationSafety.SAFE,
            ),
        ]
    if cause == _CAUSE_EXPLICIT_VISIBILITY_MASK:
        return [
            RemediationStep(
                action=(
                    "Start PyTorch workloads from a shell without the "
                    "CUDA_VISIBLE_DEVICES mask (an empty value or -1 "
                    "masks every GPU)"
                ),
                why=(
                    "The current environment explicitly hides all GPUs "
                    "from CUDA, which by itself explains the observed "
                    "unavailability"
                ),
                safety=RemediationSafety.SAFE,
            ),
            RemediationStep(
                action=(
                    "Check where the mask is set: a launcher script, "
                    "container entrypoint, or job scheduler may export "
                    "it rather than the user's shell profile"
                ),
                why=(
                    "Removing the mask only in the current shell may not "
                    "persist if the value is inherited from a parent "
                    "process"
                ),
                safety=RemediationSafety.SAFE,
            ),
        ]
    return [
        RemediationStep(
            action=(
                "Gather further diagnostic evidence rather than changing "
                "drivers, toolkits, or libraries: collect the "
                "verification output below from the failing environment"
            ),
            why=(
                "No supported root cause could be established; modifying "
                "the system without an established cause risks "
                "unnecessary and possibly harmful changes"
            ),
            safety=RemediationSafety.SAFE,
        ),
        RemediationStep(
            action=(
                "Direct the next investigation at CUDA runtime "
                "initialization and library resolution in the failing "
                "environment"
            ),
            why=(
                "The build, GPU visibility, driver usability, driver "
                "compatibility, explicit masking, and supported "
                "library-conflict evidence were all checked without "
                "establishing a cause"
            ),
            safety=RemediationSafety.SAFE,
        ),
    ]


def _verification(platform: Platform) -> list[str]:
    """Frozen verification minimum (truth tables §18.16), display-only.

    The same read-only triple serves every cause class ("depending on
    cause" in the frozen table is permissive, not a narrower minimum);
    Windows syntax on Windows (architecture §40). The rule never
    executes any of these.
    """
    if platform is Platform.WINDOWS:
        return [
            "python -c \"import torch; print(torch.version.cuda); "
            "print(torch.cuda.is_available())\"",
            "nvidia-smi",
            "echo %CUDA_VISIBLE_DEVICES%",
        ]
    return [
        "python -c \"import torch; print(torch.version.cuda); "
        "print(torch.cuda.is_available())\"",
        "nvidia-smi",
        "echo $CUDA_VISIBLE_DEVICES",
    ]


def _exclusions(cause: str, facts: DiagnosisFacts) -> list[str]:
    """Concise healthy/excluded causes (plan §17.9, architecture §13.8).

    Each frozen exclusion string appears only when its branch was
    actually checked and positively established as healthy — an UNKNOWN
    tri-state never produces a healthy-string exclusion.
    """
    exclusions: list[str] = []
    if cause != _CAUSE_CPU_ONLY and facts.torch.cuda_build is True:
        exclusions.append("CUDA-enabled PyTorch build confirmed")
    if cause != _CAUSE_GPU_NOT_VISIBLE and facts.gpu.visible is Trinary.TRUE:
        exclusions.append("GPU independently visible")
    if cause != _CAUSE_DRIVER_UNUSABLE and facts.driver.usable is Trinary.TRUE:
        exclusions.append("driver usable")
    if (
        cause != _CAUSE_DRIVER_INCOMPATIBLE
        and facts.driver.compatible_with_torch_runtime is Trinary.TRUE
    ):
        exclusions.append("driver compatible")
    if (
        cause != _CAUSE_EXPLICIT_VISIBILITY_MASK
        and facts.visibility.cuda_visible_devices
        in (CUDAVisibleDevicesState.UNSET, CUDAVisibleDevicesState.RESTRICTED_NONEMPTY)
    ):
        # Restricted-nonempty is a device restriction, not a mask-all
        # (N06/N08); unparseable stays unknown and never appears here.
        exclusions.append("no explicit CUDA visibility mask")
    return exclusions


def _related_issues(issue_codes: frozenset[str]) -> list[str]:
    """The symptom-level legacy issue, when the environment produced it.

    TORCH002 is the observed-unavailability symptom/finding that D003
    interprets at the diagnosis layer (semantics §22.5); it stays
    machine-readable and is listed as related evidence only when it
    actually fired.
    """
    return [_LEGACY_ISSUE_CODE] if _LEGACY_ISSUE_CODE in issue_codes else []

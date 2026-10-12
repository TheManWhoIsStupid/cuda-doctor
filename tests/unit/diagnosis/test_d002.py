"""Tests for the D002 PyTorch/local toolkit build mismatch rule.

The rule is tested over synthetic ``DiagnosisFacts`` (truth tables §31):
collector and normalization behavior is pinned by the Phase-2/Phase-5
suites; here only the diagnosis decision over already-normalized facts
matters. Frozen truth-table row IDs appear in test names/comments
(D002-P01 … D002-P06, D002-N01 … D002-N09, D002-I01, D002-U01, plus the
combined §17.8/§20.1-§20.3 rows) so the coverage stays auditable against
the frozen tables.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from tests.factories import base_snapshot

from cuda_doctor.collectors.cuda import (
    VERSION_SOURCE_DERIVED_WEAK,
    VERSION_SOURCE_DIRECT,
)
from cuda_doctor.compatibility import load_compatibility_data
from cuda_doctor.core.enums import Platform, Severity
from cuda_doctor.core.models import CUDASelectorObservation, EnvironmentSnapshot
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.diagnosis.facts import (
    DiagnosisFactBuilder,
    DiagnosisFacts,
    IdentityReliability,
    PlatformApplicability,
    ReferenceToolkit,
    ToolkitCompatibility,
    ToolkitSelectorFact,
    TorchDiagnosisFacts,
    normalize_torch_toolkit_relation,
)
from cuda_doctor.diagnosis.issue import Issue
from cuda_doctor.diagnosis.models import (
    Confidence,
    Diagnosis,
    EvidenceItem,
    ImpactItem,
    RemediationSafety,
)
from cuda_doctor.diagnosis.rules import (
    D001ToolkitResolutionConflict,
    D002TorchToolkitMismatch,
    DiagnosisRule,
)
from cuda_doctor.diagnosis.rules import base as rules_base
from cuda_doctor.diagnosis.rules import d002 as d002_module
from cuda_doctor.utils.versions import parse_cuda_version

COMPATIBILITY = load_compatibility_data()

RULE = D002TorchToolkitMismatch()
D001 = D001ToolkitResolutionConflict()

# Frozen selector-class flags (Phase-5 normalization): name → (explicit, effective).
_SELECTOR_FLAGS = {
    "CUDA_HOME": (True, False),
    "CUDA_PATH": (True, False),
    "CUDACXX": (True, False),
    "NVCC": (False, True),
    "PATH_CUDA_BIN": (False, True),
    "USR_LOCAL_CUDA": (False, False),
}


def sel(
    name: str,
    version: str | None,
    reliability: IdentityReliability = IdentityReliability.DIRECT,
    *,
    canonical_root: str | None = None,
    valid: bool | None = True,
    active: bool = True,
) -> ToolkitSelectorFact:
    """Synthetic selector fact with the frozen class flags for ``name``."""
    explicit, effective = _SELECTOR_FLAGS[name]
    return ToolkitSelectorFact(
        name=name,
        explicit=explicit,
        effective=effective,
        active=active,
        valid=valid,
        canonical_root=canonical_root,
        toolkit_version=version,
        reliability=reliability,
    )


def torch_fact(
    importable: bool = True,
    cuda_build: bool | None = True,
    cuda_version: str | None = "12.4",
    cuda_available: bool | None = True,
) -> TorchDiagnosisFacts:
    """Synthetic torch facts; D002 only needs the comparison surface."""
    return TorchDiagnosisFacts(
        importable=importable,
        cuda_build=cuda_build,
        cuda_version=cuda_version,
        cuda_available=cuda_available,
        cuda_probe_error=None,
        cudnn_version=None,
    )


def ref(
    source: str = "NVCC",
    version: str | None = "11.8",
    reliability: IdentityReliability = IdentityReliability.DIRECT,
    *,
    canonical_root: str | None = None,
) -> ReferenceToolkit:
    """Synthetic reference toolkit (the authoritative Phase-5 result)."""
    return ReferenceToolkit(
        source=source,
        canonical_root=canonical_root,
        toolkit_version=version,
        reliability=reliability,
    )


def base_facts(platform: Platform = Platform.LINUX) -> DiagnosisFacts:
    snapshot = base_snapshot()
    if platform is not Platform.LINUX:
        snapshot.system = replace(snapshot.system, platform=platform)
    return DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)


def facts_with(
    torch: TorchDiagnosisFacts | None = None,
    reference: ReferenceToolkit | None = None,
    *,
    selectors: tuple[ToolkitSelectorFact, ...] = (),
    issue_codes: frozenset[str] = frozenset(),
    platform: Platform = Platform.LINUX,
    applicability: PlatformApplicability | None = None,
) -> DiagnosisFacts:
    base = base_facts(platform)
    return replace(
        base,
        torch=torch if torch is not None else base.torch,
        reference_toolkit=reference,
        toolkit_selectors=selectors,
        issue_codes=frozenset(issue_codes),
        applicability=applicability if applicability is not None else base.applicability,
    )


def observation(name: str, **overrides) -> CUDASelectorObservation:
    params: dict = {"name": name}
    params.update(overrides)
    return CUDASelectorObservation(**params)


def snapshot_with_selectors(
    *observations: CUDASelectorObservation,
    platform: Platform | None = None,
    cuda_version: str | None = None,
) -> EnvironmentSnapshot:
    """Base snapshot with explicit selector observations (and optionally a
    different torch CUDA build version)."""
    snapshot = base_snapshot()
    snapshot.cuda = replace(snapshot.cuda, selector_observations=list(observations))
    if platform is not None:
        snapshot.system = replace(snapshot.system, platform=platform)
    if cuda_version is not None:
        snapshot.pytorch = replace(snapshot.pytorch, cuda_version=cuda_version)
    return snapshot


def built_facts(snapshot: EnvironmentSnapshot) -> DiagnosisFacts:
    return DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)


def mismatch_facts() -> DiagnosisFacts:
    """The canonical D002 environment: torch CUDA 12.4 vs NVCC 11.8."""
    return facts_with(
        torch=torch_fact(cuda_version="12.4"),
        reference=ref("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
    )


def mismatch_diagnosis() -> Diagnosis:
    candidate = RULE.evaluate(mismatch_facts(), [])
    assert candidate is not None, "expected a D002 candidate"
    return candidate.diagnosis


class TestRuleContract:
    def test_code_is_d002(self):
        assert D002TorchToolkitMismatch.code == "D002"
        assert D002TorchToolkitMismatch().code == "D002"

    def test_satisfies_the_frozen_rule_protocol(self):
        assert isinstance(RULE, DiagnosisRule)

    def test_issue_and_prior_candidate_context_never_changes_the_result(self):
        # D002 is independent of D001 under the frozen tables (§20.1-
        # §20.3) and never inspects issue prose (plan §9): both
        # parameters must stay inert.
        facts = mismatch_facts()
        issues = [
            Issue(code="TORCH004", severity=Severity.INFO, title="torch cuda differs"),
            Issue(code="CUDA006", severity=Severity.WARNING, title="nvcc root differs"),
        ]
        d001_candidate = D001.evaluate(
            facts_with(
                selectors=(
                    sel("CUDA_HOME", "11.8", canonical_root="/opt/a"),
                    sel("NVCC", "12.4", canonical_root="/opt/b"),
                )
            ),
            [],
        )
        assert d001_candidate is not None
        assert RULE.evaluate(facts, issues) == RULE.evaluate(facts, [])
        assert RULE.evaluate(facts, [], {"D001": d001_candidate}) == RULE.evaluate(facts, [])


class TestPositiveMajorMismatchRows:
    def test_p01_torch_11_8_vs_reference_12_4(self):
        # D002-P01: torch CUDA 11.x against a CUDA 12.x reference toolkit.
        facts = facts_with(
            torch=torch_fact(cuda_version="11.8"),
            reference=ref("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
        )
        diagnosis = RULE.evaluate(facts, [])
        assert diagnosis is not None
        assert diagnosis.diagnosis.confidence is Confidence.HIGH
        assert diagnosis.diagnosis.severity is Severity.WARNING

    def test_p02_torch_12_4_vs_reference_11_8(self):
        # D002-P02: the mirrored direction.
        diagnosis = mismatch_diagnosis()
        assert diagnosis.confidence is Confidence.HIGH
        assert diagnosis.severity is Severity.WARNING

    def test_p03_torch_12_1_vs_reference_13_0(self):
        # D002-P03: 12.x vs 13.x — a major boundary that is not 11/12.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.1"),
            reference=ref("NVCC", "13.0", canonical_root="/opt/cuda-13.0"),
        )
        assert RULE.evaluate(facts, []) is not None

    def test_linux_major_mismatch_emits(self):
        facts = mismatch_facts()
        assert facts.applicability.d002 is True
        assert RULE.evaluate(facts, []) is not None

    def test_windows_major_mismatch_emits(self):
        # D002 is evaluated on Windows too (truth tables §16.11 analogue);
        # CUDA_PATH is a legitimate Windows reference source.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("CUDA_PATH", "11.8", canonical_root="C:/toolkits/v11.8"),
            platform=Platform.WINDOWS,
        )
        assert facts.applicability.d002 is True
        assert RULE.evaluate(facts, []) is not None

    def test_reference_source_cudacxx(self):
        # §17.2: an explicit CUDACXX is the highest-precedence reference.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("CUDACXX", "11.8", canonical_root="/opt/cuda-11.8"),
        )
        diagnosis = RULE.evaluate(facts, [])
        assert diagnosis is not None
        assert diagnosis.diagnosis.evidence[1] == EvidenceItem(
            key="reference_source", value="CUDACXX", source="toolkit_selectors"
        )

    def test_reference_source_nvcc_builder_based(self):
        # End-to-end through the fact builder: resolved nvcc becomes the
        # reference when no CUDACXX exists, and D002 fires on it.
        snapshot = snapshot_with_selectors(
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-11.8",
                toolkit_version="11.8",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            )
        )
        facts = built_facts(snapshot)
        assert facts.reference_toolkit is not None
        assert facts.reference_toolkit.source == "NVCC"
        assert RULE.evaluate(facts, []) is not None

    def test_p06_directory_reference_where_valid_under_phase5_precedence(self):
        # D002-P06: a usable CUDA_HOME development toolkit (no CUDACXX,
        # no NVCC) is a legitimate directory reference; a material
        # mismatch against it emits D002.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                raw_value="/opt/cuda-11.8",
                canonical_root="/opt/cuda-11.8",
                toolkit_version="11.8",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            )
        )
        facts = built_facts(snapshot)
        assert facts.reference_toolkit is not None
        assert facts.reference_toolkit.source == "CUDA_HOME"
        diagnosis = RULE.evaluate(facts, [])
        assert diagnosis is not None
        assert diagnosis.diagnosis.confidence is Confidence.HIGH

    def test_derived_strong_reference_combination_emits(self):
        # Accepted strong reliability combination (§17.10): a strongly
        # derived reference version still proves the mismatch, at MEDIUM.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8", IdentityReliability.DERIVED_STRONG),
        )
        diagnosis = RULE.evaluate(facts, [])
        assert diagnosis is not None
        assert diagnosis.diagnosis.confidence is Confidence.MEDIUM

    def test_cuda_probe_unavailable_does_not_block_the_comparison(self):
        # D002 is a build-tool mismatch, not a runtime-health verdict:
        # cuda_available=None (probe produced no boolean) with a known
        # torch CUDA build version still compares.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4", cuda_available=None),
            reference=ref("NVCC", "11.8"),
        )
        assert RULE.evaluate(facts, []) is not None


class TestD001Interaction:
    def test_20_1_torch_12_4_cudacxx_11_8_nvcc_12_4_both_fire(self):
        # Combined §20.1: D001 explains the CUDACXX/NVCC disagreement,
        # D002 describes torch-vs-CUDACXX — independent, no causal edge.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("CUDACXX", "11.8", canonical_root="/opt/cuda-11.8"),
            selectors=(
                sel("CUDACXX", "11.8", canonical_root="/opt/cuda-11.8"),
                sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            ),
        )
        d001 = D001.evaluate(facts, [])
        d002 = RULE.evaluate(facts, [])
        assert d001 is not None
        assert d002 is not None
        assert d002.causal_parent_codes == ()

    def test_20_2_torch_12_4_cudacxx_12_4_nvcc_11_8_d001_only(self):
        # Combined §20.2: the explicit build compiler matches PyTorch, so
        # no D002 — even though the broader shell toolchain is inconsistent.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("CUDACXX", "12.4", canonical_root="/opt/cuda-12.4"),
            selectors=(
                sel("CUDACXX", "12.4", canonical_root="/opt/cuda-12.4"),
                sel("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
            ),
        )
        assert D001.evaluate(facts, []) is not None
        assert RULE.evaluate(facts, []) is None

    def test_20_3_all_selectors_agree_11_8_torch_12_4_d002_only(self):
        # Combined §20.3: every local selector agrees on CUDA 11.8, so no
        # D001 — but the torch build mismatch is still D002.
        selectors = (
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
            sel("CUDACXX", "11.8", canonical_root="/opt/cuda-11.8"),
            sel("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
        )
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
            selectors=selectors,
        )
        assert D001.evaluate(facts, []) is None
        assert RULE.evaluate(facts, []) is not None


class TestNegativeNoDiagnosisRows:
    @pytest.mark.parametrize(
        ("torch_version", "toolkit_version"),
        [("12.4", "12.4"), ("11.8", "11.8"), ("12.1", "12.1")],
        ids=["n01-12_4", "n02-11_8", "n03-12_1"],
    )
    def test_exact_match_rows(self, torch_version, toolkit_version):
        # D002-N01, N02, N03: an exact torch/reference match never emits.
        facts = facts_with(
            torch=torch_fact(cuda_version=torch_version),
            reference=ref("NVCC", toolkit_version),
        )
        assert RULE.evaluate(facts, []) is None

    def test_n04_exact_relation_with_derived_reference(self):
        # D002-N04: relation EXACT — even a strongly derived reference
        # agreeing with the torch build is no mismatch.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("CUDA_HOME", "12.4", IdentityReliability.DERIVED_STRONG),
        )
        assert RULE.evaluate(facts, []) is None

    def test_i01_same_major_minor_difference_12_1_vs_12_4(self):
        # D002-I01: same CUDA major, minor differs — informational only
        # (the legacy TORCH004 issue stays authoritative), never D002.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.1"),
            reference=ref("NVCC", "12.4"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_same_major_minor_difference_reversed_12_4_vs_12_1(self):
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "12.1"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_torch_absent_or_not_importable(self):
        # Truth tables §17.1 preconditions: PyTorch absent and PyTorch
        # import failure both leave the comparison context unestablished.
        facts = facts_with(
            torch=torch_fact(importable=False, cuda_build=None, cuda_version=None),
            reference=ref("NVCC", "11.8"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_cpu_only_pytorch_build(self):
        facts = facts_with(
            torch=torch_fact(cuda_build=False, cuda_version=None),
            reference=ref("NVCC", "11.8"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_cpu_only_build_with_stale_version_field_excluded(self):
        # Defensive: a CPU-only flag dominates any stale version field.
        facts = facts_with(
            torch=torch_fact(cuda_build=False, cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_torch_cuda_version_unknown(self):
        facts = facts_with(
            torch=torch_fact(cuda_build=True, cuda_version=None),
            reference=ref("NVCC", "11.8"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_no_reference_toolkit_wheel_only(self):
        # D002-N05/N06: healthy wheel-only use with no local development
        # toolkit is NOT D002 — absence never implies mismatch.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4", cuda_available=True),
            reference=None,
        )
        assert RULE.evaluate(facts, []) is None

    def test_reference_toolkit_version_unknown(self):
        # D002-U-row: the reference exists but its version is underivable.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("CUDACXX", None, IdentityReliability.UNKNOWN),
        )
        assert RULE.evaluate(facts, []) is None

    @pytest.mark.parametrize("platform", [Platform.MACOS, Platform.OTHER])
    def test_not_evaluated_on_macos_or_other(self, platform):
        # Architecture §33: D002 runs on Linux and Windows only.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
            platform=platform,
        )
        assert facts.applicability.d002 is False
        assert RULE.evaluate(facts, []) is None

    def test_torch004_info_alone_on_minor_difference_causes_no_d002(self):
        # Plan §13: the legacy INFO issue by itself must not upgrade a
        # same-major informational difference into a diagnosis.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.1"),
            reference=ref("NVCC", "12.4"),
            issue_codes=frozenset({"TORCH004"}),
        )
        assert RULE.evaluate(facts, []) is None


class TestCudaBuildPrecondition:
    """Maintainer review follow-up: D002 requires an established CUDA
    build context. UNKNOWN (``cuda_build=None``) is not one — even with
    a known version string — while a confirmed CUDA build keeps emitting
    regardless of the availability probe's outcome."""

    def test_unknown_build_state_with_known_version_does_not_emit(self):
        # Malformed/partial facts: importable, cuda_version present, but
        # the build state itself is not established — no D002 even
        # though the version would major-mismatch the reference.
        facts = facts_with(
            torch=torch_fact(cuda_build=None, cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
        )
        assert RULE.evaluate(facts, []) is None

    @pytest.mark.parametrize(
        ("torch_version", "toolkit_version"),
        [("12.4", "11.8"), ("11.8", "12.4")],
        ids=["torch-newer", "torch-older"],
    )
    def test_unknown_build_state_never_emits_in_either_direction(
        self, torch_version, toolkit_version
    ):
        # A clear major mismatch in either direction stays
        # no-diagnosis while cuda_build is UNKNOWN.
        facts = facts_with(
            torch=torch_fact(cuda_build=None, cuda_version=torch_version),
            reference=ref("NVCC", toolkit_version),
        )
        assert RULE.evaluate(facts, []) is None

    def test_confirmed_cuda_build_still_emits(self):
        facts = facts_with(
            torch=torch_fact(cuda_build=True, cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.HIGH
        assert candidate.diagnosis.severity is Severity.WARNING

    @pytest.mark.parametrize("cuda_available", [True, False, None])
    def test_cuda_available_irrelevant_once_build_confirmed(self, cuda_available):
        # D002 is a build/toolchain diagnosis: once the CUDA build
        # identity is established, the availability probe's outcome
        # (True/False/None) never changes the decision or the output.
        facts = facts_with(
            torch=torch_fact(cuda_build=True, cuda_version="12.4", cuda_available=cuda_available),
            reference=ref("NVCC", "11.8"),
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        baseline = RULE.evaluate(
            facts_with(
                torch=torch_fact(cuda_build=True, cuda_version="12.4", cuda_available=True),
                reference=ref("NVCC", "11.8"),
            ),
            [],
        )
        assert candidate == baseline


class TestUnknownConservativeRows:
    def test_unknown_torch_cuda_with_known_reference(self):
        facts = facts_with(
            torch=torch_fact(cuda_build=True, cuda_version=None),
            reference=ref("NVCC", "11.8"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_known_torch_with_unknown_reference_identity(self):
        # D002-U01-shaped: reference present, identity/version unknown.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("CUDACXX", None, IdentityReliability.UNKNOWN),
        )
        assert RULE.evaluate(facts, []) is None

    def test_reference_reliability_unknown_excluded(self):
        # UNKNOWN reliability cannot prove a major mismatch (plan §7);
        # this also covers malformed facts carrying a version anyway.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8", IdentityReliability.UNKNOWN),
        )
        assert RULE.evaluate(facts, []) is None

    def test_malformed_torch_version_string_degrades_to_unknown(self):
        facts = facts_with(
            torch=torch_fact(cuda_version="banana"),
            reference=ref("NVCC", "11.8"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_malformed_reference_version_string_degrades_to_unknown(self):
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "weird-version"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_canonical_root_difference_alone_never_emits(self):
        # Path inequality is not version evidence: roots may differ wildly
        # while the versions agree — no mismatch is inferred from roots.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "12.4", canonical_root="/home/other-user/cuda"),
        )
        assert RULE.evaluate(facts, []) is None


class TestPrecedenceTrust:
    def test_cudacxx_unknown_version_blocks_nvcc_fallback_synthetic(self):
        # Plan §13 precedence: Phase-5 chose CUDACXX as the reference even
        # though its version is underivable — D002 must stay no-diagnosis
        # and must NOT bypass the reference to compare against NVCC.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("CUDACXX", None, IdentityReliability.UNKNOWN),
            selectors=(
                sel("CUDACXX", None, IdentityReliability.UNKNOWN, canonical_root="/opt/w"),
                sel("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
            ),
        )
        assert RULE.evaluate(facts, []) is None

    def test_cudacxx_unknown_version_blocks_nvcc_fallback_builder(self):
        # The same scenario end-to-end: the fact layer hands D002 the
        # version-less CUDACXX reference and the rule accepts it as-is.
        snapshot = snapshot_with_selectors(
            observation(
                "cudacxx",
                raw_value="/opt/wrapper/nvcc",
                canonical_root=None,
                valid=True,
                version_source=None,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-11.8",
                toolkit_version="11.8",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        facts = built_facts(snapshot)
        assert facts.reference_toolkit is not None
        assert facts.reference_toolkit.source == "CUDACXX"
        assert facts.reference_toolkit.toolkit_version is None
        assert RULE.evaluate(facts, []) is None

    def test_torch_matches_cudacxx_while_nvcc_differs_no_d002(self):
        # §17.8 direction 2: torch 11.8, CUDACXX 11.8, NVCC 12.4 — the
        # reference is CUDACXX and it agrees with torch, so no D002,
        # regardless of what NVCC resolves.
        facts = facts_with(
            torch=torch_fact(cuda_version="11.8"),
            reference=ref("CUDACXX", "11.8", canonical_root="/opt/cuda-11.8"),
            selectors=(
                sel("CUDACXX", "11.8", canonical_root="/opt/cuda-11.8"),
                sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            ),
        )
        assert RULE.evaluate(facts, []) is None

    def test_output_changes_only_with_reference_not_selector_ranking(self):
        # Identical toolkit_selectors, different authoritative reference:
        # D002's decision follows reference_toolkit alone — it never
        # re-ranks the selector facts to find another comparison side.
        selectors = (
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
            sel("CUDACXX", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
        )
        against_nvcc = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
            selectors=selectors,
        )
        against_cudacxx = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("CUDACXX", "12.4"),
            selectors=selectors,
        )
        assert RULE.evaluate(against_nvcc, []) is not None
        assert RULE.evaluate(against_cudacxx, []) is None

    def test_metadata_only_directory_is_never_a_reference(self):
        # D002-N08 (builder-based): CUDA_HOME exists with metadata-only
        # version evidence (weakly derived) — not a usable development
        # toolkit, so no reference, so no D002 even though the metadata
        # version majors away from torch.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                raw_value="/opt/not-really-a-toolkit",
                canonical_root="/opt/not-really-a-toolkit",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            ),
            cuda_version="11.8",
        )
        facts = built_facts(snapshot)
        assert facts.reference_toolkit is None
        assert RULE.evaluate(facts, []) is None

    def test_applicability_gate_trusted_over_platform(self):
        # The rule consumes the normalized applicability fact only; even a
        # mismatching environment on Linux stays unevaluated when the
        # gate is False.
        base = base_facts(Platform.LINUX)
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
            applicability=replace(base.applicability, d002=False),
        )
        assert RULE.evaluate(facts, []) is None


class TestConfidenceMapping:
    def test_direct_reference_is_high(self):
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8", IdentityReliability.DIRECT),
        )
        assert RULE.evaluate(facts, []).diagnosis.confidence is Confidence.HIGH

    def test_derived_strong_reference_is_medium(self):
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8", IdentityReliability.DERIVED_STRONG),
        )
        assert RULE.evaluate(facts, []).diagnosis.confidence is Confidence.MEDIUM

    def test_derived_weak_reference_is_not_emitted(self):
        # Truth tables §17.10: weakly derived identity is LOW-confidence
        # evidence and LOW-confidence D002 is not emitted at all, in any
        # role.
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8", IdentityReliability.DERIVED_WEAK),
        )
        assert RULE.evaluate(facts, []) is None

    def test_unknown_reliability_is_not_emitted(self):
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8", IdentityReliability.UNKNOWN),
        )
        assert RULE.evaluate(facts, []) is None


class TestCandidateOutputContent:
    def test_frozen_title_and_summary(self):
        diagnosis = mismatch_diagnosis()
        assert diagnosis.code == "D002"
        assert diagnosis.title == "PyTorch/local toolkit build mismatch"
        assert diagnosis.summary == (
            "The installed PyTorch build targets CUDA 12.4 while the "
            "active local development toolkit (NVCC) resolves CUDA 11.8; "
            "the differing CUDA major versions make CUDA extension "
            "builds against this toolkit likely incompatible."
        )

    def test_evidence_bounded_normalized_and_ordered(self):
        assert mismatch_diagnosis().evidence == [
            EvidenceItem(key="torch_cuda", value="CUDA 12.4", source="torch"),
            EvidenceItem(
                key="reference_source", value="NVCC", source="toolkit_selectors"
            ),
            EvidenceItem(
                key="reference_toolkit",
                value="CUDA 11.8 (direct)",
                source="toolkit_selectors",
            ),
            EvidenceItem(
                key="compatibility",
                value="major_incompatible",
                source="compatibility",
            ),
        ]

    def test_frozen_impact(self):
        assert mismatch_diagnosis().impact == [
            ImpactItem(scope="PyTorch runtime", status="may remain healthy"),
            ImpactItem(
                scope="CUDA extension build", status="high risk / likely incompatible"
            ),
            ImpactItem(scope="local CUDA development", status="mismatched"),
        ]

    def test_remediation_safe_guidance_without_commands(self):
        # Plan §12: no facts-supported concrete target exists, so every
        # step carries command=None; all steps stay SAFE and explained.
        steps = mismatch_diagnosis().remediation
        assert 2 <= len(steps) <= 5
        for step in steps:
            assert step.safety is RemediationSafety.SAFE
            assert step.command is None
            assert step.why
        actions = " ".join(step.action for step in steps)
        assert "No change is required" in actions  # wheel-only runtime stance
        assert "CUDA extensions" in actions  # strategy A present
        assert "PyTorch build" in actions  # strategy B present

    def test_remediation_distinguishes_runtime_from_extension_development(self):
        # Truth tables §17.11: the remediation must separate ordinary
        # PyTorch runtime use from CUDA extension development and must
        # not mandate any change for wheel-only use.
        steps = mismatch_diagnosis().remediation
        first = f"{steps[0].action} {steps[0].why}"
        assert "bundle their own CUDA runtime" in first
        assert "compiling CUDA extensions" in " ".join(
            f"{step.action} {step.why}" for step in steps
        )

    def test_remediation_invents_no_target_toolkit_version(self):
        # The Phase-6 lesson (plan §12): D002 must not promote either
        # observed version — or any never-observed one — as the target.
        facts = facts_with(
            torch=torch_fact(cuda_version="11.8"),
            reference=ref("NVCC", "12.1"),
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        for step in candidate.diagnosis.remediation:
            rendered = f"{step.action} {step.why}"
            for forbidden in ("11.8", "12.1", "12.4", "v12.4", "cuda-11", "cuda-12"):
                assert forbidden not in rendered, forbidden

    def test_no_removal_destructive_or_reinstall_remediation(self):
        rendered = str(asdict(mismatch_diagnosis()))
        for forbidden in ("rm -rf", "uninstall", "apt remove", "delete", "reinstall"):
            assert forbidden not in rendered, forbidden

    def test_linux_verification_commands(self):
        assert mismatch_diagnosis().verification == [
            "python -c \"import torch; print(torch.version.cuda); "
            "print(torch.cuda.is_available())\"",
            "which nvcc",
            "nvcc --version",
            "echo $CUDACXX",
        ]

    def test_windows_verification_commands(self):
        facts = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("CUDA_PATH", "11.8"),
            platform=Platform.WINDOWS,
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert candidate.diagnosis.verification == [
            "python -c \"import torch; print(torch.version.cuda); "
            "print(torch.cuda.is_available())\"",
            "where nvcc",
            "nvcc --version",
            "echo %CUDACXX%",
        ]

    def test_diagnosis_stays_pre_resolver(self):
        diagnosis = mismatch_diagnosis()
        assert diagnosis.role is None
        assert diagnosis.cause_class is None
        assert diagnosis.relationships == []
        assert diagnosis.exclusions == []

    def test_candidate_is_risk_only_without_causal_parents(self):
        candidate = RULE.evaluate(mismatch_facts(), [])
        assert candidate.risk_only is True
        assert candidate.explains_active_failure is False
        assert candidate.causal_parent_codes == ()

    def test_related_issues_lists_torch004_only_when_present(self):
        with_issue = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
            issue_codes=frozenset({"TORCH004", "CUDA006"}),
        )
        without_issue = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
        )
        assert RULE.evaluate(with_issue, []).diagnosis.related_issues == ["TORCH004"]
        assert RULE.evaluate(without_issue, []).diagnosis.related_issues == []


class TestTorch004Interaction:
    def test_suppression_carrier_owned_by_d002_on_every_emission(self):
        # Architecture §25.2 assigns TORCH004 presentation suppression
        # ownership to D002: the carrier is populated whenever D002 is
        # emitted, whether or not the legacy issue fired. Actual
        # suppression is a later presentation concern; the issue always
        # stays machine-readable.
        for codes in (frozenset(), frozenset({"TORCH004"})):
            facts = facts_with(
                torch=torch_fact(cuda_version="12.4"),
                reference=ref("NVCC", "11.8"),
                issue_codes=codes,
            )
            candidate = RULE.evaluate(facts, [])
            assert candidate is not None
            assert candidate.diagnosis.presentation_suppressed_issue_codes == ["TORCH004"]

    def test_torch004_presence_changes_only_related_issues(self):
        # With the underlying major mismatch present, the legacy issue's
        # presence adds related-issue linkage and nothing else.
        common = {"torch": torch_fact(cuda_version="12.4"), "reference": ref("NVCC", "11.8")}
        without = RULE.evaluate(facts_with(**common), [])
        with_issue = RULE.evaluate(
            facts_with(issue_codes=frozenset({"TORCH004"}), **common), []
        )
        assert without is not None and with_issue is not None
        assert without.diagnosis.related_issues == []
        assert with_issue.diagnosis.related_issues == ["TORCH004"]
        assert without.diagnosis.evidence == with_issue.diagnosis.evidence
        assert without.diagnosis.summary == with_issue.diagnosis.summary


class TestCompatibilityAdapterReuse:
    def test_uses_the_phase5_adapter_not_a_forked_table(self):
        # Plan §14: D002 must reuse normalize_torch_toolkit_relation from
        # diagnosis.facts — never interpret_torch_cuda directly, never a
        # second hand-rolled major/minor comparison.
        source = Path(d002_module.__file__).read_text()
        tree = ast.parse(source)
        imported_from_facts: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "cuda_doctor.diagnosis.facts":
                imported_from_facts.update(alias.name for alias in node.names)
        assert "normalize_torch_toolkit_relation" in imported_from_facts
        assert "normalize_torch_toolkit_relation(" in source  # actually called
        assert "interpret_torch_cuda" not in source  # no direct prose-level fork
        assert ".major" not in source  # no hand-rolled major comparison

    @pytest.mark.parametrize(
        ("torch_version", "toolkit_version"),
        [
            ("12.4", "12.4"),
            ("11.8", "11.8"),
            ("12.4", "12.1"),
            ("12.1", "12.4"),
            ("11.8", "12.4"),
            ("12.4", "11.8"),
            ("12.1", "13.0"),
            ("11.8", "11.9"),
            ("12", "12.0"),
            ("12.4.1", "12.4.2"),
        ],
        ids=[
            "exact-12_4",
            "exact-11_8",
            "minor-forward",
            "minor-backward",
            "major-down",
            "major-up",
            "major-12-vs-13",
            "minor-11",
            "pad-normalized",
            "patch-normalized",
        ],
    )
    def test_emission_matches_the_adapter_relation(self, torch_version, toolkit_version):
        # Behavioral non-forking proof: for every pair, D002 emits exactly
        # when the frozen Phase-5 adapter classifies MAJOR_INCOMPATIBLE.
        relation = normalize_torch_toolkit_relation(
            parse_cuda_version(torch_version), parse_cuda_version(toolkit_version)
        )
        facts = facts_with(
            torch=torch_fact(cuda_version=torch_version),
            reference=ref("NVCC", toolkit_version),
        )
        emitted = RULE.evaluate(facts, []) is not None
        assert emitted is (relation is ToolkitCompatibility.MAJOR_INCOMPATIBLE)


class TestEvidencePrivacy:
    def test_secret_canonical_root_never_reaches_the_candidate(self):
        # Plan §11: ReferenceToolkit carries canonical_root, but the D002
        # candidate must never expose it. A secret-looking root on a
        # genuinely mismatching directory reference still emits D002 —
        # bounded evidence only.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                raw_value="/home/secretuser/toolkits/cuda-11.8",
                canonical_root="/home/secretuser/toolkits/cuda-11.8",
                toolkit_version="11.8",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            )
        )
        facts = built_facts(snapshot)
        assert facts.reference_toolkit is not None
        assert facts.reference_toolkit.canonical_root is not None
        assert "secretuser" in str(asdict(facts))  # meaningful in the fact layer
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert "secretuser" not in str(asdict(candidate.diagnosis))
        assert "secretuser" not in str(candidate.diagnosis.to_dict())
        assert "secretuser" not in str(asdict(candidate))
        assert "toolkits/cuda" not in str(asdict(candidate.diagnosis))


class TestPurityAndDeterminism:
    def test_same_facts_produce_equal_candidates(self):
        facts = mismatch_facts()
        assert RULE.evaluate(facts, []) == RULE.evaluate(facts, [])

    def test_selector_input_order_does_not_change_output(self):
        # Selector order is presentation-only for the fact layer; D002's
        # output over the same reference is order-independent.
        first = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
            selectors=(
                sel("CUDACXX", "12.4", canonical_root="/opt/a"),
                sel("NVCC", "11.8", canonical_root="/opt/b"),
            ),
        )
        second = facts_with(
            torch=torch_fact(cuda_version="12.4"),
            reference=ref("NVCC", "11.8"),
            selectors=(
                sel("NVCC", "11.8", canonical_root="/opt/b"),
                sel("CUDACXX", "12.4", canonical_root="/opt/a"),
            ),
        )
        assert RULE.evaluate(first, []) == RULE.evaluate(second, [])

    def test_host_environment_changes_nothing(self, monkeypatch):
        # The rule must never consult os.environ or other host state:
        # mutating the process environment between evaluations changes
        # nothing.
        facts = mismatch_facts()
        before = RULE.evaluate(facts, [])
        monkeypatch.setenv("CUDA_HOME", "/host/cuda-99.0")
        monkeypatch.setenv("CUDACXX", "/host/bin/nvcc")
        monkeypatch.setenv("PATH", "/host/bin")
        assert RULE.evaluate(facts, []) == before

    @pytest.mark.parametrize("module", [rules_base, d002_module], ids=["base", "d002"])
    def test_imports_no_host_probes(self, module):
        # Structural purity (Phase-7 plan §15): any host probe
        # (environment, filesystem, subprocess, torch, clock, randomness)
        # or collector/snapshot-model dependency requires an import.
        tree = ast.parse(Path(module.__file__).read_text())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert imported.isdisjoint(
            {"os", "subprocess", "shutil", "site", "time", "random", "torch"}
        )
        for name in imported:
            assert not name.startswith("cuda_doctor.collectors"), name
            assert name != "cuda_doctor.core.models", name

    def test_source_never_names_collector_observation_types(self):
        # The rule must not even reference the pre-normalization surface:
        # its text names no snapshot/observation/inventory type.
        for module in (rules_base, d002_module):
            source = Path(module.__file__).read_text()
            for forbidden in (
                "EnvironmentSnapshot",
                "CUDASelectorObservation",
                "RuntimeLibraryInventory",
            ):
                assert forbidden not in source, forbidden


class TestNoEngineIntegration:
    def test_engine_still_emits_no_diagnoses(self):
        # Phase-7 plan §16: the rule fires on the facts, but the engine
        # neither runs it nor populates DiagnosisResult.diagnoses.
        snapshot = snapshot_with_selectors(
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-11.8",
                toolkit_version="11.8",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            )
        )
        result = DiagnosisEngine().run(snapshot)
        assert result.diagnoses == []
        assert result.diagnosis_errors == {}
        assert RULE.evaluate(built_facts(snapshot), []) is not None

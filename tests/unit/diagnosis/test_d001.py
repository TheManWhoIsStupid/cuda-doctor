"""Tests for the D001 CUDA toolkit resolution conflict rule.

The rule is tested over synthetic ``DiagnosisFacts`` (truth tables §31):
collector and normalization behavior is pinned by the Phase-2/Phase-5
suites; here only the diagnosis decision over already-normalized facts
matters. Frozen truth-table row IDs appear in test names/comments
(D001-P01 … D001-P08, D001-N01 … D001-N14, D001-U01 … D001-U04) so the
coverage stays auditable against the frozen tables.
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
from cuda_doctor.core.models import CUDAInstallation, CUDASelectorObservation, EnvironmentSnapshot
from cuda_doctor.diagnosis.engine import DiagnosisEngine
from cuda_doctor.diagnosis.facts import (
    DiagnosisFactBuilder,
    DiagnosisFacts,
    IdentityReliability,
    PlatformApplicability,
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
)
from cuda_doctor.diagnosis.rules import D001ToolkitResolutionConflict, DiagnosisRule
from cuda_doctor.diagnosis.rules import base as rules_base
from cuda_doctor.diagnosis.rules import d001 as d001_module

COMPATIBILITY = load_compatibility_data()

RULE = D001ToolkitResolutionConflict()

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


def base_facts(platform: Platform = Platform.LINUX) -> DiagnosisFacts:
    snapshot = base_snapshot()
    if platform is not Platform.LINUX:
        snapshot.system = replace(snapshot.system, platform=platform)
    return DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)


def facts_with(
    *selectors: ToolkitSelectorFact,
    platform: Platform = Platform.LINUX,
    applicability: PlatformApplicability | None = None,
) -> DiagnosisFacts:
    base = base_facts(platform)
    return replace(
        base,
        toolkit_selectors=selectors,
        applicability=applicability if applicability is not None else base.applicability,
    )


def observation(name: str, **overrides) -> CUDASelectorObservation:
    params: dict = {"name": name}
    params.update(overrides)
    return CUDASelectorObservation(**params)


def snapshot_with_selectors(
    *observations: CUDASelectorObservation, platform: Platform | None = None
) -> EnvironmentSnapshot:
    snapshot = base_snapshot()
    snapshot.cuda = replace(snapshot.cuda, selector_observations=list(observations))
    if platform is not None:
        snapshot.system = replace(snapshot.system, platform=platform)
    return snapshot


def built_facts(snapshot: EnvironmentSnapshot) -> DiagnosisFacts:
    return DiagnosisFactBuilder().build(snapshot, [], COMPATIBILITY)


def selector_fact(facts: DiagnosisFacts, name: str) -> ToolkitSelectorFact:
    matches = [fact for fact in facts.toolkit_selectors if fact.name == name]
    assert len(matches) == 1, f"expected exactly one {name} selector fact"
    return matches[0]


def emitted_diagnosis(facts: DiagnosisFacts) -> Diagnosis:
    candidate = RULE.evaluate(facts, [])
    assert candidate is not None, "expected a D001 candidate"
    return candidate.diagnosis


def linux_conflict_facts() -> DiagnosisFacts:
    return facts_with(
        sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
        sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
    )


def linux_conflict_candidate() -> DiagnosisCandidate:
    candidate = RULE.evaluate(linux_conflict_facts(), [])
    assert candidate is not None
    return candidate


class TestRuleContract:
    def test_code_is_d001(self):
        assert D001ToolkitResolutionConflict.code == "D001"
        assert D001ToolkitResolutionConflict().code == "D001"

    def test_satisfies_the_frozen_rule_protocol(self):
        assert isinstance(RULE, DiagnosisRule)

    def test_issue_and_prior_candidate_context_never_changes_the_result(self):
        # D001 has no frozen issue-code or upstream-candidate dependency
        # (Phase-6 plan §11): both parameters must stay inert.
        facts = linux_conflict_facts()
        issues = [
            Issue(code="CUDA006", severity=Severity.WARNING, title="nvcc root differs"),
            Issue(code="TORCH002", severity=Severity.ERROR, title="torch cuda false"),
        ]
        upstream = DiagnosisCandidate(
            diagnosis=Diagnosis(
                code="D999",
                title="placeholder",
                severity=Severity.WARNING,
                confidence=Confidence.LOW,
                summary="unrelated upstream candidate",
            )
        )
        assert RULE.evaluate(facts, issues) == RULE.evaluate(facts, [])
        assert RULE.evaluate(facts, [], {"D999": upstream}) == RULE.evaluate(facts, [])


class TestPositiveConflictRows:
    def test_p01_linux_cuda_home_11_8_vs_effective_nvcc_12_4(self):
        # D001-P01: explicit CUDA_HOME resolves CUDA 11.8 while the
        # effective nvcc resolves CUDA 12.4 — the canonical conflict.
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
        )
        diagnosis = emitted_diagnosis(facts)
        assert diagnosis.code == "D001"
        assert diagnosis.confidence is Confidence.HIGH
        assert diagnosis.severity is Severity.WARNING

    def test_p02_linux_cudacxx_12_4_vs_effective_nvcc_11_8(self):
        # D001-P02: the explicitly selected compiler (CUDACXX 12.4)
        # disagrees with the PATH-resolved nvcc (11.8).
        facts = facts_with(
            sel("CUDACXX", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
        )
        diagnosis = emitted_diagnosis(facts)
        assert diagnosis.confidence is Confidence.HIGH
        assert diagnosis.severity is Severity.WARNING

    def test_p03_linux_cuda_home_12_1_vs_cudacxx_12_4(self):
        # D001-P03: two explicit selectors disagree (12.1 vs 12.4).
        facts = facts_with(
            sel("CUDA_HOME", "12.1", canonical_root="/opt/cuda-12.1"),
            sel("CUDACXX", "12.4", canonical_root="/opt/cuda-12.4"),
        )
        assert emitted_diagnosis(facts).confidence is Confidence.HIGH

    def test_p04_more_than_two_active_selectors_single_candidate(self):
        # D001-P04: three active selectors (12.4/12.4/11.8) with at least
        # one meaningful disagreement — still exactly one candidate, and
        # every participating selector appears in the evidence exactly
        # once (no pairwise duplicates).
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("CUDACXX", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert [item.key for item in candidate.diagnosis.evidence] == [
            "CUDACXX",
            "CUDA_HOME",
            "NVCC",
        ]

    def test_p05_windows_cuda_path_11_8_vs_nvcc_12_4(self):
        # D001-P05: on Windows CUDA_PATH is an active selector.
        facts = facts_with(
            sel("CUDA_PATH", "11.8", canonical_root="C:/toolkits/v11.8"),
            sel("NVCC", "12.4", canonical_root="C:/toolkits/v12.4"),
            platform=Platform.WINDOWS,
        )
        diagnosis = emitted_diagnosis(facts)
        assert diagnosis.confidence is Confidence.HIGH
        assert diagnosis.severity is Severity.WARNING

    def test_p06_windows_cudacxx_12_4_vs_cuda_path_12_1(self):
        # D001-P06: explicit compiler vs Windows environment selector.
        facts = facts_with(
            sel("CUDA_PATH", "12.1", canonical_root="C:/toolkits/v12.1"),
            sel("CUDACXX", "12.4", canonical_root="C:/toolkits/v12.4"),
            platform=Platform.WINDOWS,
        )
        assert emitted_diagnosis(facts).confidence is Confidence.HIGH

    def test_p07_usr_local_cuda_active_via_cuda_home_genuinely_disagrees(self):
        # D001-P07: CUDA_HOME=/usr/local/cuda (symlink → 11.8) makes
        # USR_LOCAL_CUDA an active participant whose normalized version
        # genuinely disagrees with the effective nvcc (12.4).
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                raw_value="/usr/local/cuda",
                canonical_root="/usr/local/cuda-11.8",
                toolkit_version="11.8",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "usr_local_cuda",
                raw_value="/usr/local/cuda",
                canonical_root="/usr/local/cuda-11.8",
                toolkit_version="11.8",
                exists=True,
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        facts = built_facts(snapshot)
        assert selector_fact(facts, "USR_LOCAL_CUDA").active is True
        diagnosis = emitted_diagnosis(facts)
        assert [item.key for item in diagnosis.evidence] == [
            "CUDA_HOME",
            "NVCC",
            "USR_LOCAL_CUDA",
        ]
        # The direct CUDA_HOME/NVCC disagreement establishes HIGH; the
        # derived_weak USR_LOCAL_CUDA participant does not cap it.
        assert diagnosis.confidence is Confidence.HIGH

    def test_p08_usr_local_cuda_active_via_nvcc_genuinely_disagrees(self):
        # D001-P08: nvcc resolved through /usr/local/cuda/bin/nvcc (11.8)
        # vs an explicit CUDACXX (12.4).
        snapshot = snapshot_with_selectors(
            observation(
                "nvcc",
                raw_value="nvcc",
                resolved_path="/usr/local/cuda/bin/nvcc",
                canonical_path="/usr/local/cuda-11.8/bin/nvcc",
                canonical_root="/usr/local/cuda-11.8",
                toolkit_version="11.8",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "usr_local_cuda",
                raw_value="/usr/local/cuda",
                canonical_root="/usr/local/cuda-11.8",
                toolkit_version="11.8",
                exists=True,
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            ),
            observation(
                "cudacxx",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        facts = built_facts(snapshot)
        assert selector_fact(facts, "USR_LOCAL_CUDA").active is True
        diagnosis = emitted_diagnosis(facts)
        assert [item.key for item in diagnosis.evidence] == [
            "CUDACXX",
            "NVCC",
            "USR_LOCAL_CUDA",
        ]

    def test_three_distinct_versions_still_one_candidate(self):
        # A three-way disagreement stays one D001 family — never pairwise
        # duplicate candidates (Phase-6 plan §6).
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
            sel("CUDACXX", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("NVCC", "13.0", canonical_root="/opt/cuda-13.0"),
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        assert candidate.diagnosis.confidence is Confidence.HIGH
        assert len(candidate.diagnosis.evidence) == 3


class TestNegativeNoDiagnosisRows:
    def test_n01_same_reliable_version_different_roots(self):
        # D001-N01: CUDA_HOME → /opt/cuda-a (12.4) and NVCC → /opt/cuda-b
        # (12.4): roots differ, reliable versions agree — no conflict.
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/opt/cuda-a"),
            sel("NVCC", "12.4", canonical_root="/opt/cuda-b"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_n02_cudacxx_and_nvcc_same_version_different_roots(self):
        # D001-N02
        facts = facts_with(
            sel("CUDACXX", "12.4", canonical_root="/opt/cuda-a"),
            sel("NVCC", "12.4", canonical_root="/opt/cuda-b"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_n03_three_selectors_same_version_distinct_roots(self):
        # D001-N03
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/opt/cuda-a"),
            sel("CUDACXX", "12.4", canonical_root="/opt/cuda-b"),
            sel("NVCC", "12.4", canonical_root="/opt/cuda-c"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_n04_same_canonical_toolkit_through_aliases(self):
        # D001-N04: different spellings resolving to the same normalized
        # toolkit identity do not conflict.
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("NVCC", "12.4", canonical_root="/opt/cuda-12.4"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_n05_builder_symlink_alias_single_identity(self):
        # D001-N05 (builder-based): a symlinked CUDA_HOME spelling and a
        # direct spelling canonicalize to one toolkit identity.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                raw_value="/opt/cuda-latest",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "nvcc",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        facts = built_facts(snapshot)
        assert {fact.canonical_root for fact in facts.toolkit_selectors} == {
            "/opt/cuda-12.4"
        }
        assert RULE.evaluate(facts, []) is None

    def test_n06_usr_local_cuda_alias_agrees_with_active_toolkit(self):
        # D001-N06: USR_LOCAL_CUDA actively participates but describes
        # the same toolkit version as every other active selector.
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/usr/local/cuda-12.4"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            sel(
                "USR_LOCAL_CUDA",
                "12.4",
                IdentityReliability.DERIVED_WEAK,
                canonical_root="/usr/local/cuda-12.4",
            ),
        )
        assert RULE.evaluate(facts, []) is None

    def test_n07_multiple_installed_one_active(self):
        # D001-N07: two installed toolkits (11.8 + 12.4), one active
        # selector (CUDA_HOME 12.4) — installation multiplicity is never
        # a conflict.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            )
        )
        snapshot.cuda = replace(
            snapshot.cuda,
            installations=[
                CUDAInstallation("/usr/local/cuda-11.8", "11.8"),
                CUDAInstallation("/usr/local/cuda-12.4", "12.4"),
            ],
        )
        facts = built_facts(snapshot)
        assert len(facts.toolkit_selectors) == 1
        assert RULE.evaluate(facts, []) is None

    def test_n08_all_active_selectors_agree(self):
        # D001-N08: several active selectors, all resolving 12.4.
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("CUDACXX", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            sel("PATH_CUDA_BIN", "12.4", canonical_root="/usr/local/cuda-12.4/bin"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_n09_incidental_installations_differ_from_agreeing_selectors(self):
        # D001-N09: all known active selectors agree while incidental
        # installed toolkits report other versions — installs never
        # participate in D001.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        snapshot.cuda = replace(
            snapshot.cuda,
            installations=[
                CUDAInstallation("/opt/cuda-11.8", "11.8"),
                CUDAInstallation("/opt/cuda-13.0", "13.0"),
            ],
        )
        facts = built_facts(snapshot)
        assert {fact.toolkit_version for fact in facts.toolkit_selectors} == {"12.4"}
        assert RULE.evaluate(facts, []) is None

    def test_n10_invalid_cuda_home_is_not_an_identity(self):
        # D001-N10: an invalid selector (valid=False, no version) carries
        # no usable identity — it cannot conflict with anything.
        facts = facts_with(
            sel("CUDA_HOME", None, IdentityReliability.UNKNOWN, valid=False),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_n11_invalid_cudacxx_ignored_against_valid_nvcc(self):
        # D001-N11
        facts = facts_with(
            sel("CUDACXX", None, IdentityReliability.UNKNOWN, valid=False),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_n12_active_but_versionless_selector_ignored(self):
        # D001-N12: active and valid, but no derivable version identity —
        # never mismatch evidence.
        facts = facts_with(
            sel("CUDACXX", None, IdentityReliability.UNKNOWN, canonical_root="/opt/w"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_n13_builder_invalid_selector_keeps_no_conflict(self):
        # D001-N13 (builder-based): an unresolvable CUDACXX observation
        # alongside a healthy nvcc yields no D001 from real facts.
        snapshot = snapshot_with_selectors(
            observation(
                "cudacxx",
                raw_value="/no/such/nvcc",
                exists=False,
                valid=False,
                version_source=None,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        assert RULE.evaluate(built_facts(snapshot), []) is None

    def test_n14_stale_usr_local_cuda_symlink_inactive(self):
        # D001-N14: /usr/local/cuda exists (pointing at 11.8) but nothing
        # resolves through it while NVCC resolves 12.4 — no
        # participation, no conflict.
        snapshot = snapshot_with_selectors(
            observation(
                "usr_local_cuda",
                raw_value="/usr/local/cuda",
                canonical_root="/usr/local/cuda-11.8",
                toolkit_version="11.8",
                exists=True,
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        facts = built_facts(snapshot)
        assert selector_fact(facts, "USR_LOCAL_CUDA").active is False
        assert RULE.evaluate(facts, []) is None

    def test_usr_local_cuda_same_target_but_no_passage(self):
        # Phase-6 plan §12: /usr/local/cuda points at the same toolkit
        # as the active CUDA_HOME, but resolution never passed through
        # the symlink — accepted Phase-5 semantics keep it inactive, and
        # D001 trusts that normalized fact.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                raw_value="/usr/local/cuda-12.4",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "usr_local_cuda",
                raw_value="/usr/local/cuda",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                exists=True,
                valid=True,
                version_source=VERSION_SOURCE_DERIVED_WEAK,
            ),
        )
        facts = built_facts(snapshot)
        assert selector_fact(facts, "USR_LOCAL_CUDA").active is False
        assert RULE.evaluate(facts, []) is None

    def test_nvidia_smi_generation_differs_but_selectors_agree(self):
        # The driver's "CUDA Version" (13.0 in the base snapshot) never
        # becomes a selector: agreeing selectors stay healthy even though
        # the driver reports a different CUDA generation.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        assert snapshot.driver.cuda_version == "13.0"
        assert RULE.evaluate(built_facts(snapshot), []) is None

    def test_no_selectors_at_all(self):
        assert RULE.evaluate(facts_with(), []) is None

    def test_single_active_selector(self):
        # One active selector cannot disagree with anything.
        facts = facts_with(sel("CUDA_HOME", "12.4", canonical_root="/opt/cuda-12.4"))
        assert RULE.evaluate(facts, []) is None

    def test_cuda_home_and_nvcc_agree(self):
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_nvcc_and_path_cuda_bin_agree(self):
        facts = facts_with(
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            sel("PATH_CUDA_BIN", "12.4", canonical_root="/usr/local/cuda-12.4/bin"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_linux_cuda_path_is_never_a_selector(self):
        # Truth tables §16.11: on Linux CUDA_PATH is inventory
        # information only — even a disagreeing value stays inactive and
        # cannot conflict.
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            sel("CUDA_PATH", "12.6", canonical_root="/opt/cuda-12.6", active=False),
        )
        assert RULE.evaluate(facts, []) is None

    def test_builder_linux_cuda_path_stays_inactive(self):
        # The fact layer (not D001) owns Linux CUDA_PATH inactivity —
        # proven end-to-end through the builder.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "cuda_path",
                raw_value="/opt/cuda-12.6",
                canonical_root="/opt/cuda-12.6",
                toolkit_version="12.6",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        facts = built_facts(snapshot)
        assert selector_fact(facts, "CUDA_PATH").active is False
        assert RULE.evaluate(facts, []) is None

    def test_shadowed_path_cuda_bin_not_reconstructed(self):
        # A non-winning PATH entry never becomes an active selector: the
        # fact layer represents only the winning PATH_CUDA_BIN, and D001
        # never re-derives PATH precedence.
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            sel(
                "PATH_CUDA_BIN", "11.8", canonical_root="/opt/shadowed/cuda-11.8/bin",
                active=False,
            ),
        )
        assert RULE.evaluate(facts, []) is None


class TestUnknownIdentityRows:
    def test_u01_cuda_home_unknown_vs_nvcc_known(self):
        # D001-U01: one identity-known selector plus one unknown-version
        # selector — UNKNOWN is insufficient comparison evidence.
        facts = facts_with(
            sel("CUDA_HOME", None, IdentityReliability.UNKNOWN, canonical_root="/opt/a"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_u02_cudacxx_known_vs_nvcc_unknown(self):
        # D001-U02
        facts = facts_with(
            sel("CUDACXX", "12.4", canonical_root="/opt/cuda-12.4"),
            sel("NVCC", None, IdentityReliability.UNKNOWN, canonical_root="/opt/b"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_u03_both_unknown(self):
        # D001-U03: all active selectors have unknown identity.
        facts = facts_with(
            sel("CUDA_HOME", None, IdentityReliability.UNKNOWN),
            sel("NVCC", None, IdentityReliability.UNKNOWN),
        )
        assert RULE.evaluate(facts, []) is None

    def test_u04_roots_differ_versions_unknown(self):
        # D001-U04: differing roots with unknown identities never create
        # a mismatch — path inequality is not version evidence.
        facts = facts_with(
            sel("CUDA_HOME", None, IdentityReliability.UNKNOWN, canonical_root="/opt/a"),
            sel("NVCC", None, IdentityReliability.UNKNOWN, canonical_root="/opt/b"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_wrapper_cudacxx_direct_version_is_sufficient(self):
        # Truth tables §16: a wrapper CUDACXX with a DIRECT version but
        # no root identity still carries a comparable version — a
        # genuine version disagreement with NVCC emits D001 (HIGH).
        facts = facts_with(
            sel("CUDACXX", "12.4"),
            sel("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
        )
        diagnosis = emitted_diagnosis(facts)
        assert diagnosis.confidence is Confidence.HIGH
        assert [item.key for item in diagnosis.evidence] == ["CUDACXX", "NVCC"]

    def test_wrapper_cudacxx_unknown_identity_emits_nothing(self):
        # The same wrapper without a derivable version is an unknown
        # identity — no mismatch, whatever the other selector resolves.
        facts = facts_with(
            sel("CUDACXX", None, IdentityReliability.UNKNOWN),
            sel("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_known_version_with_unknown_reliability_excluded(self):
        # Malformed/partial facts degrade conservatively: the fact layer
        # can carry a known version with UNKNOWN reliability (e.g. an
        # unrecognized version source); such a selector is not
        # sufficiently reliable identity evidence and never conflicts.
        facts = facts_with(
            sel("CUDA_HOME", "12.4", IdentityReliability.UNKNOWN,
                canonical_root="/opt/cuda-12.4"),
            sel("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
        )
        assert RULE.evaluate(facts, []) is None

    def test_valid_none_selector_excluded(self):
        # A selector with valid=None (unverified) is not valid evidence.
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/opt/cuda-12.4", valid=None),
            sel("NVCC", "11.8", canonical_root="/usr/local/cuda-11.8"),
        )
        assert RULE.evaluate(facts, []) is None


class TestDuplicateSelectorHandling:
    def test_three_selectors_same_toolkit_no_conflict(self):
        # Co-referent selectors (HOME + NVCC + USR_LOCAL all describing
        # one toolkit) never manufacture a conflict.
        facts = facts_with(
            sel("CUDA_HOME", "12.4", canonical_root="/usr/local/cuda-12.4"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            sel(
                "USR_LOCAL_CUDA",
                "12.4",
                IdentityReliability.DERIVED_WEAK,
                canonical_root="/usr/local/cuda-12.4",
            ),
        )
        assert RULE.evaluate(facts, []) is None

    def test_one_disagreement_across_several_selectors_one_candidate(self):
        # One real disagreement (HOME 11.8 vs the rest 12.4) yields
        # exactly one candidate — never pairwise duplicates.
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/usr/local/cuda-11.8"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            sel("PATH_CUDA_BIN", "12.4", canonical_root="/usr/local/cuda-12.4/bin"),
            sel(
                "USR_LOCAL_CUDA",
                "12.4",
                IdentityReliability.DERIVED_WEAK,
                canonical_root="/usr/local/cuda-12.4",
            ),
        )
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        keys = [item.key for item in candidate.diagnosis.evidence]
        assert keys == ["CUDA_HOME", "NVCC", "PATH_CUDA_BIN", "USR_LOCAL_CUDA"]
        assert len(set(keys)) == len(keys)  # no duplicated evidence


class TestPlatformApplicability:
    @pytest.mark.parametrize("platform", [Platform.MACOS, Platform.OTHER])
    def test_not_evaluated_on_macos_or_other(self, platform):
        # Architecture §33: D001 is evaluated on Linux and Windows only;
        # elsewhere it produces no candidate and no error.
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            platform=platform,
        )
        assert facts.applicability.d001 is False
        assert RULE.evaluate(facts, []) is None

    @pytest.mark.parametrize("platform", [Platform.LINUX, Platform.WINDOWS])
    def test_evaluated_on_linux_and_windows(self, platform):
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            platform=platform,
        )
        assert facts.applicability.d001 is True
        assert RULE.evaluate(facts, []) is not None

    def test_applicability_gate_trusted_over_platform(self):
        # The rule consumes the normalized applicability fact only; even
        # a conflicting environment on Linux stays unevaluated when the
        # gate is False.
        base = base_facts(Platform.LINUX)
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            applicability=replace(base.applicability, d001=False),
        )
        assert RULE.evaluate(facts, []) is None


class TestConfidenceMapping:
    def test_direct_vs_direct_is_high(self):
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-a"),
            sel("NVCC", "12.4", canonical_root="/opt/cuda-b"),
        )
        assert emitted_diagnosis(facts).confidence is Confidence.HIGH

    def test_direct_vs_derived_strong_is_high(self):
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-a"),
            sel("NVCC", "12.4", IdentityReliability.DERIVED_STRONG, canonical_root="/opt/b"),
        )
        assert emitted_diagnosis(facts).confidence is Confidence.HIGH

    def test_strong_vs_strong_is_high(self):
        facts = facts_with(
            sel("CUDA_HOME", "11.8", IdentityReliability.DERIVED_STRONG,
                canonical_root="/opt/cuda-a"),
            sel("NVCC", "12.4", IdentityReliability.DERIVED_STRONG, canonical_root="/opt/b"),
        )
        assert emitted_diagnosis(facts).confidence is Confidence.HIGH

    def test_direct_vs_derived_weak_is_medium(self):
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-a"),
            sel("NVCC", "12.4", IdentityReliability.DERIVED_WEAK, canonical_root="/opt/b"),
        )
        assert emitted_diagnosis(facts).confidence is Confidence.MEDIUM

    def test_weak_vs_weak_is_medium(self):
        facts = facts_with(
            sel("CUDA_HOME", "11.8", IdentityReliability.DERIVED_WEAK,
                canonical_root="/opt/cuda-a"),
            sel("NVCC", "12.4", IdentityReliability.DERIVED_WEAK, canonical_root="/opt/b"),
        )
        assert emitted_diagnosis(facts).confidence is Confidence.MEDIUM

    def test_strong_disagreement_not_downgraded_by_weak_participant(self):
        # Extra derived_weak participants must not dilute an established
        # direct disagreement (truth tables §7: HIGH stays possible).
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-a"),
            sel("NVCC", "12.4", canonical_root="/opt/cuda-b"),
            sel(
                "USR_LOCAL_CUDA",
                "11.8",
                IdentityReliability.DERIVED_WEAK,
                canonical_root="/opt/cuda-a",
            ),
        )
        assert emitted_diagnosis(facts).confidence is Confidence.HIGH


class TestCandidateOutputContent:
    def test_frozen_title_and_summary(self):
        diagnosis = linux_conflict_candidate().diagnosis
        assert diagnosis.code == "D001"
        assert diagnosis.title == "CUDA toolkit resolution conflict"
        assert diagnosis.summary == (
            "Active CUDA development selectors disagree about the selected "
            "CUDA toolkit: CUDA_HOME resolves CUDA 11.8; NVCC resolves CUDA 12.4."
        )

    def test_evidence_bounded_normalized_and_ordered(self):
        diagnosis = linux_conflict_candidate().diagnosis
        assert diagnosis.evidence == [
            EvidenceItem(
                key="CUDA_HOME",
                value="CUDA 11.8 (direct)",
                source="toolkit_selectors",
            ),
            EvidenceItem(
                key="NVCC",
                value="CUDA 12.4 (direct)",
                source="toolkit_selectors",
            ),
        ]

    def test_frozen_impact(self):
        assert linux_conflict_candidate().diagnosis.impact == [
            ImpactItem(scope="system CUDA configuration", status="inconsistent"),
            ImpactItem(scope="CUDA development toolchain", status="inconsistent"),
            ImpactItem(scope="CUDA extension build", status="at risk"),
            ImpactItem(scope="PyTorch runtime", status="not automatically broken"),
        ]

    def test_remediation_safe_actionable_and_linux_native(self):
        diagnosis = linux_conflict_candidate().diagnosis
        steps = diagnosis.remediation
        assert 1 <= len(steps) <= 3
        for step in steps:
            assert step.safety is RemediationSafety.SAFE
            assert step.command is not None
            assert step.command.startswith("export ")
            assert "sudo" not in step.command
            assert step.why  # every step explains itself
        actions = " ".join(step.action for step in steps)
        assert "CUDA_HOME" in actions
        assert "CUDACXX" not in actions  # CUDACXX is not participating here
        assert "intended toolkit" in actions

    def test_remediation_targets_each_participating_selector_class(self):
        facts = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-a"),
            sel("CUDACXX", "12.1", canonical_root="/opt/cuda-b"),
            sel("NVCC", "12.4", canonical_root="/opt/cuda-c"),
        )
        steps = emitted_diagnosis(facts).remediation
        commands = [step.command or "" for step in steps]
        assert any("CUDA_HOME=" in command for command in commands)
        assert any("CUDACXX=" in command for command in commands)
        assert any("PATH=" in command for command in commands)

    def test_windows_remediation_uses_cmd_syntax(self):
        facts = facts_with(
            sel("CUDA_PATH", "11.8", canonical_root="C:/toolkits/v11.8"),
            sel("NVCC", "12.4", canonical_root="C:/toolkits/v12.4"),
            platform=Platform.WINDOWS,
        )
        steps = emitted_diagnosis(facts).remediation
        assert steps
        for step in steps:
            assert step.safety is RemediationSafety.SAFE
            assert step.command is not None
            assert step.command.startswith("set ")
            assert "export" not in step.command

    def test_no_removal_or_destructive_remediation(self):
        rendered = str(asdict(linux_conflict_candidate().diagnosis))
        for forbidden in ("rm -rf", "uninstall", "apt remove", "delete"):
            assert forbidden not in rendered, forbidden

    def test_linux_verification_commands(self):
        assert linux_conflict_candidate().diagnosis.verification == [
            "echo $CUDA_HOME",
            "echo $CUDACXX",
            "which nvcc",
            "nvcc --version",
        ]

    def test_windows_verification_commands(self):
        facts = facts_with(
            sel("CUDA_PATH", "11.8", canonical_root="C:/toolkits/v11.8"),
            sel("NVCC", "12.4", canonical_root="C:/toolkits/v12.4"),
            platform=Platform.WINDOWS,
        )
        assert emitted_diagnosis(facts).verification == [
            "echo %CUDA_PATH%",
            "echo %CUDACXX%",
            "where nvcc",
            "nvcc --version",
        ]

    def test_diagnosis_stays_pre_resolver(self):
        # Phase-6 plan §2/§7: no final role, no resolver-owned fields.
        diagnosis = linux_conflict_candidate().diagnosis
        assert diagnosis.role is None
        assert diagnosis.cause_class is None
        assert diagnosis.related_issues == []
        assert diagnosis.relationships == []
        assert diagnosis.exclusions == []
        assert diagnosis.presentation_suppressed_issue_codes == []

    def test_candidate_is_risk_only_without_causal_parents(self):
        candidate = linux_conflict_candidate()
        assert candidate.risk_only is True
        assert candidate.explains_active_failure is False
        assert candidate.causal_parent_codes == ()


class TestEvidencePrivacy:
    def test_secret_raw_paths_never_reach_the_candidate(self):
        # Phase-6 plan §9: a secret-looking raw path present in source
        # snapshot observations must not appear in the final D001
        # candidate.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                raw_value="/home/secretuser/toolkits/cuda-11.8",
                canonical_root="/home/secretuser/toolkits/cuda-11.8",
                toolkit_version="11.8",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        facts = built_facts(snapshot)
        # The private path genuinely exists in the fact layer's input...
        assert "secretuser" in str(asdict(facts))
        candidate = RULE.evaluate(facts, [])
        assert candidate is not None
        # ...but never reaches the candidate in any rendered form.
        assert "secretuser" not in str(asdict(candidate.diagnosis))
        assert "secretuser" not in str(candidate.diagnosis.to_dict())
        assert "secretuser" not in str(asdict(candidate))
        assert "toolkits/cuda" not in str(asdict(candidate.diagnosis))


class TestPurityAndDeterminism:
    def test_same_facts_produce_equal_candidates(self):
        # Phase-6 plan §13: same DiagnosisFacts → equal D001 result.
        facts = linux_conflict_facts()
        assert RULE.evaluate(facts, []) == RULE.evaluate(facts, [])

    def test_selector_input_order_does_not_change_output(self):
        first = facts_with(
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
        )
        second = facts_with(
            sel("NVCC", "12.4", canonical_root="/usr/local/cuda-12.4"),
            sel("CUDA_HOME", "11.8", canonical_root="/opt/cuda-11.8"),
        )
        assert RULE.evaluate(first, []) == RULE.evaluate(second, [])

    def test_host_environment_changes_nothing(self, monkeypatch):
        # The rule must never consult os.environ or other host state:
        # mutating the process environment between evaluations changes
        # nothing.
        facts = linux_conflict_facts()
        before = RULE.evaluate(facts, [])
        monkeypatch.setenv("CUDA_HOME", "/host/cuda-99.0")
        monkeypatch.setenv("CUDACXX", "/host/bin/nvcc")
        monkeypatch.setenv("PATH", "/host/bin")
        assert RULE.evaluate(facts, []) == before

    @pytest.mark.parametrize("module", [rules_base, d001_module], ids=["base", "d001"])
    def test_imports_no_host_probes(self, module):
        # Structural purity (Phase-6 plan §13): any host probe
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
        for module in (rules_base, d001_module):
            source = Path(module.__file__).read_text()
            for forbidden in (
                "EnvironmentSnapshot",
                "CUDASelectorObservation",
                "RuntimeLibraryInventory",
            ):
                assert forbidden not in source, forbidden


class TestNoEngineIntegration:
    def test_engine_still_emits_no_diagnoses(self):
        # Phase-6 plan §14: the rule fires on the facts, but the engine
        # neither runs it nor populates DiagnosisResult.diagnoses.
        snapshot = snapshot_with_selectors(
            observation(
                "cuda_home",
                canonical_root="/opt/cuda-11.8",
                toolkit_version="11.8",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
            observation(
                "nvcc",
                canonical_root="/usr/local/cuda-12.4",
                toolkit_version="12.4",
                valid=True,
                version_source=VERSION_SOURCE_DIRECT,
            ),
        )
        result = DiagnosisEngine().run(snapshot)
        assert result.diagnoses == []
        assert result.diagnosis_errors == {}
        assert RULE.evaluate(built_facts(snapshot), []) is not None
